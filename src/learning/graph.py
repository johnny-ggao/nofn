"""
LangGraph 交易工作流
"""
from typing import Dict, Optional, Union

from datetime import datetime

from langgraph.graph import StateGraph, END
from termcolor import cprint

from .state import TradingState
from .agents import TradingAgent
from .memory import TradingMemory, TradingCase
from ..engine.trading_engine import TradingEngine
from ..engine.market_snapshot import MarketSnapshot
from ..utils.config import LLMConfig
from ..strategies import BaseStrategy, StrategyFactory


class TradingWorkflowGraph:
    """
    交易工作流图

    使用 StateGraph 定义节点和边，清晰地展示工作流程：
    1. get_market_data -> 获取市场数据
    2. get_recent_trades -> 获取最近交易记录
    3. retrieve_memory -> 检索历史记忆
    4. make_decision -> LLM 决策
    5. execute_trades -> 执行交易
    6. evaluate -> 评估决策
    7. update_memory -> 更新记忆

    每个节点都是一个函数，接收 state 并返回更新后的 state
    """

    def __init__(
        self,
        engine: TradingEngine,
        llm_config: LLMConfig,
        db_path: str = "data/trading_memory.db",
        system_prompt_path: Optional[str] = None,
        strategy: Optional[Union[str, BaseStrategy]] = None,
    ):
        """
        初始化工作流图

        Args:
            engine: 交易引擎
            llm_config: LLM 配置
            db_path: 记忆数据库路径
            system_prompt_path: 系统提示词文件路径 (如果提供策略，则此参数被忽略)
            strategy: 策略名称或策略实例。如果为 None，使用默认策略 (mtf_momentum)
        """
        # 加载策略
        self.strategy = self._load_strategy(strategy)
        cprint(f"📊 使用策略: {self.strategy.name} v{self.strategy.version}", "cyan")

        # 设置引擎的策略（用于指标计算）
        self.engine = engine
        self.engine.strategy = self.strategy
        self.engine._timeframes = self.strategy.get_timeframe_list()
        self.engine._candle_limits = self.strategy.get_candle_limits()
        self.engine._indicator_calculator = self.strategy.get_indicator_calculator()

        # 获取策略的 prompt（策略优先，其次使用传入的路径）
        if system_prompt_path:
            effective_prompt_path = system_prompt_path
        else:
            effective_prompt_path = self.strategy.config.prompt_path

        # 初始化记忆系统（支持向量搜索）
        self.memory = TradingMemory(
            db_path=db_path,
            user_id="nofn_trading",
            vector_store_dir="data/vector_store",
            embedding_provider=llm_config.embedding_provider or llm_config.provider,
            embedding_api_key=llm_config.embedding_api_key or llm_config.api_key,
            embedding_model=llm_config.embedding_model,
            enable_vector_search=True,
        )

        # 初始化 Trading Agent（使用策略的 prompt）
        self.agent = TradingAgent(
            model_provider=llm_config.provider,
            model_id=llm_config.model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=llm_config.temperature,
            system_prompt_path=effective_prompt_path,
            strategy=self.strategy,
        )

        # 创建 StateGraph
        self.graph = self._build_graph()

        # 编译图
        self.compiled_graph = self.graph.compile()

        cprint("✅ TradingWorkflowGraph 初始化完成 (LangGraph)", "green")

    @staticmethod
    def _load_strategy(strategy: Optional[Union[str, BaseStrategy]]) -> BaseStrategy:
        """
        加载策略

        Args:
            strategy: 策略名称、策略实例或 None

        Returns:
            策略实例
        """
        if strategy is None:
            # 使用默认策略
            return StrategyFactory.create_default()
        elif isinstance(strategy, str):
            # 通过名称加载
            return StrategyFactory.get(strategy)
        elif isinstance(strategy, BaseStrategy):
            # 直接使用传入的实例
            return strategy
        else:
            raise ValueError(f"无效的策略类型: {type(strategy)}")

    def _build_graph(self) -> StateGraph:
        """构建工作流图"""

        workflow = StateGraph(TradingState)  # type: ignore[arg-type]

        workflow.add_node("get_market_data", self.get_market_data)  # type: ignore[arg-type]
        workflow.add_node("get_recent_trades", self.get_recent_trades)  # type: ignore[arg-type]
        workflow.add_node("retrieve_memory", self.retrieve_memory)  # type: ignore[arg-type]
        workflow.add_node("make_decision", self.make_decision)  # type: ignore[arg-type]
        workflow.add_node("execute_trades", self.execute_trades)  # type: ignore[arg-type]
        workflow.add_node("evaluate", self.evaluate)  # type: ignore[arg-type]
        workflow.add_node("update_memory", self.update_memory)  # type: ignore[arg-type]

        workflow.set_entry_point("get_market_data")  # 入口节点
        workflow.add_edge("get_market_data", "get_recent_trades")
        workflow.add_edge("get_recent_trades", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "make_decision")

        # 条件边：根据决策类型决定是否执行
        workflow.add_conditional_edges(
            "make_decision",
            self.should_execute,  # 条件函数
            {
                "execute": "execute_trades",  # 如果应该执行
                "skip": "update_memory",      # 如果跳过执行
            }
        )

        workflow.add_edge("execute_trades", "evaluate")
        workflow.add_edge("evaluate", "update_memory")
        workflow.add_edge("update_memory", END)  # 结束节点

        return workflow

    # ========== 节点函数 ==========

    async def get_market_data(self, state: TradingState) -> dict:
        """节点1: 获取市场数据"""
        cprint("\n" + "=" * 70, "cyan")
        cprint(f"📊 迭代 {state.get('iteration', 0) + 1}: 获取市场数据", "cyan")
        cprint("=" * 70, "cyan")

        symbols = state.get('symbols', [])
        snapshot = await self.engine.get_market_snapshot(symbols)

        self._print_market_data(snapshot)

        return {
            'market_snapshot': snapshot,
            'timestamp': datetime.now(),
        }

    async def get_recent_trades(self, state: TradingState) -> dict:
        """节点2: 获取最近交易记录"""
        cprint("\n📜 获取最近交易记录...", "cyan")

        symbols = state.get('symbols', [])
        all_trades = []

        try:
            # 遍历每个交易对获取交易记录
            for symbol in symbols:
                try:
                    trades = await self.engine.adapter.get_trades(symbol=symbol, limit=10)
                    for trade in trades:
                        all_trades.append({
                            'id': trade.trade_id,
                            'order_id': trade.order_id,
                            'symbol': trade.symbol,
                            'side': trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
                            'trade_type': trade.trade_type,  # open/close/add/reduce
                            'price': float(trade.price),
                            'amount': float(trade.amount),
                            'closed_pnl': float(trade.closed_pnl) if trade.closed_pnl else None,
                            'fee': float(trade.fee) if trade.fee else None,
                            'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
                        })
                except Exception as e:
                    cprint(f"⚠️ 获取 {symbol} 交易记录失败: {e}", "yellow")

            # 按时间排序，取最近 10 笔
            all_trades.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
            trades_data = all_trades[:10]

            if trades_data:
                cprint(f"✅ 获取到 {len(trades_data)} 笔最近交易", "green")
                self._print_recent_trades(trades_data)
            else:
                cprint("ℹ️ 暂无交易记录", "yellow")

            return {'recent_trades': trades_data}

        except Exception as e:
            cprint(f"⚠️ 获取交易记录失败: {e}", "yellow")
            return {'recent_trades': []}

    async def retrieve_memory(self, state: TradingState) -> dict:
        """节点3: 检索历史记忆"""
        cprint("\n🧠 检索历史记忆...", "cyan")

        snapshot = state.get('market_snapshot')
        if not snapshot:
            return {'memory_context': "无市场数据"}

        market_conditions = snapshot.to_dict()

        # 获取记忆上下文
        memory_context = self.memory.get_context(
            market_conditions=market_conditions,
            recent_days=7,
        )

        # 搜索相似案例
        similar = self.memory.search_similar(market_conditions, limit=3)

        if similar:
            cprint(f"✅ 找到 {len(similar)} 个相似案例", "green")
            self._print_historical_cases(similar)
        else:
            cprint("ℹ️ 没有找到相似案例", "yellow")

        return {
            'memory_context': memory_context,
            'similar_cases': [c.to_dict() for c in similar],
        }

    def _print_historical_cases(self, cases: list) -> None:
        """打印历史案例详情"""
        cprint("\n" + "-" * 50, "cyan")
        cprint("📚 历史案例与经验教训", "cyan")
        cprint("-" * 50, "cyan")

        for i, case in enumerate(cases, 1):
            # 案例标题
            timestamp_str = case.timestamp.strftime('%Y-%m-%d %H:%M') if case.timestamp else 'N/A'
            cprint(f"\n[案例 {i}] {case.case_id}", "white", attrs=["bold"])
            cprint(f"  时间: {timestamp_str}", "white")

            # 相似度（如果有）
            similarity = getattr(case, 'similarity', None)
            if similarity is not None:
                sim_percent = similarity * 100
                sim_color = "green" if sim_percent >= 70 else ("yellow" if sim_percent >= 50 else "white")
                cprint(f"  相似度: {sim_percent:.1f}%", sim_color)

            # 质量评分
            if case.quality_score is not None:
                score_color = "green" if case.quality_score >= 70 else ("yellow" if case.quality_score >= 50 else "red")
                cprint(f"  质量评分: {case.quality_score}/100", score_color)

            # 已实现盈亏
            if case.realized_pnl is not None:
                pnl_color = "green" if case.realized_pnl >= 0 else "red"
                pnl_sign = "+" if case.realized_pnl >= 0 else ""
                cprint(f"  盈亏: {pnl_sign}${case.realized_pnl:.2f}", pnl_color)

            # 决策摘要
            if case.decision:
                decision_summary = case.decision[:100] + "..." if len(case.decision) > 100 else case.decision
                cprint(f"  决策: {decision_summary}", "white")

            # 反思/评估
            if case.reflection:
                reflection_summary = case.reflection[:100] + "..." if len(case.reflection) > 100 else case.reflection
                cprint(f"  评估: {reflection_summary}", "magenta")

            # 经验教训
            if case.lessons_learned:
                cprint("  经验教训:", "yellow")
                for lesson in case.lessons_learned[:3]:  # 最多显示3条
                    cprint(f"    • {lesson}", "yellow")

        cprint("\n" + "-" * 50, "cyan")

    async def make_decision(self, state: TradingState) -> dict:
        """节点4: 做出交易决策"""
        cprint("\n" + "=" * 70, "cyan")
        cprint("🧠 LLM 开始分析决策...", "cyan")
        cprint("=" * 70, "cyan")

        snapshot = state.get('market_snapshot')
        memory_context = state.get('memory_context')
        recent_trades = state.get('recent_trades', [])

        # 调用 Agent 做决策
        decision = await self.agent.make_decision(
            market_snapshot=snapshot,
            memory_context=memory_context,
            recent_trades=recent_trades,
        )

        # 打印决策
        if decision.get('analysis'):
            cprint(f"\n{decision['analysis']}\n", "white")

        cprint("=" * 70, "green")
        cprint(f"✅ 决策完成: {decision.get('decision_type', 'wait')}", "green")
        cprint("=" * 70, "green")

        self._print_signals(decision.get('signals', []), snapshot)

        return {
            'decision': decision,
            'decision_raw_response': decision.get('raw_response', ''),
        }

    async def execute_trades(self, state: TradingState) -> dict:
        """节点4: 执行交易"""
        decision = state.get('decision', {})
        signals = decision.get('signals', [])

        cprint("\n" + "=" * 70, "green")
        cprint("⚡ 执行交易信号...", "green")
        cprint("=" * 70, "green")

        results = []
        for signal in signals:
            action = signal.get('action', 'wait')

            if action == 'wait':
                continue

            cprint(f"\n执行: {action} {signal.get('symbol', '')}", "cyan")

            # 执行交易
            result = await self.engine.execute_signal({
                'action': action,
                'symbol': signal.get('symbol'),
                'amount': signal.get('amount'),
                'leverage': signal.get('leverage'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
            })

            results.append({
                'signal': signal,
                'result': result,
                'timestamp': datetime.now(),
            })

            if result.get('success'):
                cprint(f"✅ {action} 执行成功", "green")
            else:
                cprint(f"❌ {action} 执行失败: {result.get('error')}", "red")

        return {
            'execution_results': results,
            'should_execute': True,
        }

    async def evaluate(self, state: TradingState) -> dict:
        """节点5: 评估决策"""
        execution_results = state.get('execution_results', [])

        if not execution_results:
            return {
                'evaluation': {
                    'analysis': "本次无交易执行，无需评估",
                    'quality_score': 50,
                    'lessons': [],
                },
                'quality_score': 50,
            }

        cprint("\n" + "=" * 70, "magenta")
        cprint("🔍 Trading Agent 自我评估并学习...", "magenta")
        cprint("=" * 70, "magenta")

        # 获取账户信息
        account_info = await self._get_account_info()

        # 调用 Agent 评估
        evaluation = await self.agent.evaluate_and_learn(
            decision=state.get('decision', {}),
            execution_results=execution_results,
            account_info=account_info,
            market_snapshot=state.get('market_snapshot'),
        )

        cprint(f"\n{evaluation.get('analysis', '')}\n", "white")

        cprint("=" * 70, "magenta")
        cprint("✅ 评估完成", "magenta")
        if evaluation.get('lessons'):
            cprint(f"   学到了 {len(evaluation['lessons'])} 条经验", "magenta")
        cprint(f"   决策质量: {evaluation.get('quality_score', 50)}/100", "magenta")
        cprint("=" * 70, "magenta")

        return {
            'evaluation': evaluation,
            'evaluation_raw_response': evaluation.get('raw_response', ''),
            'quality_score': evaluation.get('quality_score', 50),
            'lessons_learned': evaluation.get('lessons', []),
        }

    async def update_memory(self, state: TradingState) -> dict:
        """节点6: 更新记忆"""
        cprint("\n💾 更新记忆库...", "cyan")

        snapshot = state.get('market_snapshot')
        decision = state.get('decision', {})
        evaluation = state.get('evaluation', {})

        # 序列化执行结果（将 Pydantic 模型转换为字典）
        execution_results = state.get('execution_results', [])
        serialized_results = self._serialize_execution_results(execution_results)

        # 创建交易案例
        case = TradingCase(
            market_conditions=snapshot.to_dict() if snapshot else {},
            decision=decision.get('analysis', ''),
            execution_result=serialized_results,
            reflection=evaluation.get('analysis', ''),
            lessons_learned=state.get('lessons_learned', []),
            quality_score=state.get('quality_score'),
            timestamp=state.get('timestamp', datetime.now()),
        )

        # 保存案例
        self.memory.add_case(case)

        cprint(f"✅ 案例已保存: {case.case_id}", "green")

        return {}

    # ========== 条件函数 ==========

    @staticmethod
    def should_execute(state: TradingState) -> str:
        """
        条件函数：判断是否应该执行交易

        Returns:
            "execute" 或 "skip"
        """
        decision = state.get('decision', {})
        signals = decision.get('signals', [])

        # 过滤掉 wait 信号
        executable = [s for s in signals if s.get('action') != 'wait']

        if executable:
            return "execute"
        else:
            cprint("\n✋ 无可执行信号，跳过执行", "yellow")
            return "skip"

    # ========== 辅助方法 ==========

    async def run_iteration(
        self,
        symbols: list,
        iteration: int = 0
    ) -> TradingState:
        """
        运行一次迭代
        """
        # 初始化状态
        initial_state: TradingState = {
            'symbols': symbols,
            'iteration': iteration,
            'timestamp': datetime.now(),
            'should_execute': False,
            'should_analyze': False,
            'human_approved': True,
            'lessons_learned': [],
            'errors': [],
            'warnings': [],
        }

        try:
            # 运行图
            final_state = await self.compiled_graph.ainvoke(initial_state)  # type: ignore[arg-type]

            return final_state

        except Exception as e:
            cprint(f"❌ 迭代执行失败: {e}", "red")
            import traceback
            traceback.print_exc()
            return initial_state

    async def _get_account_info(self) -> Optional[Dict]:
        """获取账户信息"""
        try:
            balance = await self.engine.adapter.get_balance()
            stats = self.engine.trade_history.get_statistics() if self.engine.trade_history else {}
            positions = await self.engine.adapter.get_positions()

            positions_data = []
            for pos in positions:
                positions_data.append({
                    'symbol': pos.symbol,
                    'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    'unrealized_pnl': float(pos.unrealized_pnl) if pos.unrealized_pnl else 0,
                    'entry_price': float(pos.entry_price),
                    'amount': float(pos.amount),
                })

            return {
                'balance': {
                    'total': float(balance.total),
                    'available': float(balance.available),
                    'frozen': float(balance.frozen),
                },
                'statistics': stats,
                'open_positions': positions_data,
            }
        except Exception as e:
            cprint(f"⚠️  获取账户信息失败: {e}", "yellow")
            return None

    @staticmethod
    def _serialize_execution_results(results: list) -> list:
        """
        序列化执行结果，将 Pydantic 模型转换为字典

        Args:
            results: 执行结果列表

        Returns:
            可 JSON 序列化的结果列表
        """
        serialized = []
        for item in results:
            serialized_item = {}

            # 处理 signal
            if 'signal' in item:
                serialized_item['signal'] = item['signal']

            # 处理 result（可能包含 ExecutionResult 对象）
            if 'result' in item:
                result = item['result']
                if isinstance(result, dict):
                    # 递归处理嵌套的 Pydantic 模型
                    serialized_result = {}
                    for key, value in result.items():
                        if hasattr(value, 'model_dump'):
                            # Pydantic 模型转换为字典
                            serialized_result[key] = value.model_dump(mode='json')
                        elif hasattr(value, 'dict'):
                            # 旧版 Pydantic 模型
                            serialized_result[key] = value.dict()
                        else:
                            serialized_result[key] = value
                    serialized_item['result'] = serialized_result
                elif hasattr(result, 'model_dump'):
                    serialized_item['result'] = result.model_dump(mode='json')
                else:
                    serialized_item['result'] = result

            # 处理 timestamp
            if 'timestamp' in item:
                ts = item['timestamp']
                if hasattr(ts, 'isoformat'):
                    serialized_item['timestamp'] = ts.isoformat()
                else:
                    serialized_item['timestamp'] = str(ts)

            serialized.append(serialized_item)

        return serialized

    @staticmethod
    def _print_market_data(snapshot: MarketSnapshot) -> None:
        """打印市场数据"""
        if not snapshot or not snapshot.assets:
            return

        for symbol, asset in snapshot.assets.items():
            cprint(f"\n{'─' * 70}", "yellow")
            cprint(f"📈 {symbol}", "yellow")
            cprint(f"{'─' * 70}", "yellow")
            cprint(f"💰 价格: ${float(asset.current_price):.2f}", "white")

            if asset.change_24h_percent is not None:
                cprint(f"📊 24H变化: {asset.change_24h_percent:+.2f}%", "white")

    @staticmethod
    def _print_signals(signals: list, snapshot: Optional[MarketSnapshot]) -> None:
        """打印交易信号"""
        if not signals:
            return

        cprint(f"\n📋 交易信号 ({len(signals)} 个):", "cyan")

        for i, signal in enumerate(signals, 1):
            action = signal.get('action', 'N/A')
            symbol = signal.get('symbol', 'N/A')

            color = "green" if action in ['open_long', 'close_short'] else "red"
            cprint(f"\n  [{i}] {action.upper()} {symbol}", color)
            cprint(f"      数量: {signal.get('amount', 'N/A')}", "white")
            cprint(f"      止损: ${signal.get('stop_loss', 'N/A')}", "white")
            cprint(f"      止盈: ${signal.get('take_profit', 'N/A')}", "white")
            cprint(f"      置信度: {signal.get('confidence', 'N/A')}%", "white")
            cprint(f"      原因: {signal.get('reason', 'N/A')}", "white")

    @staticmethod
    def _print_recent_trades(trades: list) -> None:
        """打印最近交易记录"""
        if not trades:
            return

        # 交易类型中文映射
        trade_type_map = {
            'open': '开仓',
            'close': '平仓',
            'add': '加仓',
            'reduce': '减仓',
        }

        cprint(f"\n📜 最近 {len(trades)} 笔交易:", "cyan")

        for i, trade in enumerate(trades, 1):
            side = trade.get('side', 'N/A')
            trade_type = trade.get('trade_type', 'N/A')
            trade_type_cn = trade_type_map.get(trade_type, trade_type)
            closed_pnl = trade.get('closed_pnl')

            # 根据方向和类型决定颜色
            if trade_type == 'close' and closed_pnl is not None:
                color = "green" if closed_pnl >= 0 else "red"
            else:
                color = "green" if side in ['buy', 'long'] else "red"

            # 标题行
            cprint(f"\n  [{i}] {trade.get('symbol', 'N/A')} | {trade_type_cn} | {side.upper()}", color)
            cprint(f"      价格: ${trade.get('price', 0):.2f}", "white")
            cprint(f"      数量: {trade.get('amount', 'N/A')}", "white")

            # 如果是平仓，显示盈亏
            if trade_type == 'close' and closed_pnl is not None:
                pnl_color = "green" if closed_pnl >= 0 else "red"
                pnl_sign = "+" if closed_pnl >= 0 else ""
                cprint(f"      盈亏: {pnl_sign}${closed_pnl:.2f}", pnl_color)

            if trade.get('fee'):
                cprint(f"      手续费: ${trade.get('fee'):.4f}", "white")

            if trade.get('timestamp'):
                cprint(f"      时间: {trade.get('timestamp')}", "white")