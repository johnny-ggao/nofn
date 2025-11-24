"""
学习图 (Layer 3)

使用LangGraph实现的学习和进化系统
"""
import asyncio
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from termcolor import cprint

from ..engine.trading_engine import TradingEngine
from ..engine.market_snapshot import MarketSnapshot
from ..decision.decision_maker import DecisionMaker, Decision
from .memory_manager import MemoryManager, TradingCase


class TradingState(TypedDict):
    """交易状态"""
    # 输入
    symbols: List[str]

    # Layer 1数据
    market_snapshot: Optional[MarketSnapshot]

    # Layer 3记忆
    memory_context: Optional[str]
    similar_cases: Optional[List[TradingCase]]

    # Layer 2决策
    decision: Optional[Decision]

    # 执行结果
    execution_results: Optional[List[Dict]]

    # 反思
    reflection: Optional[str]
    lessons_learned: Optional[List[str]]

    # 元数据
    iteration: int
    timestamp: datetime


class LearningGraph:
    """
    学习图

    核心创新：将LangGraph用在真正重要的地方
    - 状态管理
    - 记忆检索
    - 反思学习
    - 策略优化

    而不是用于简单的工具调用
    """

    def __init__(
        self,
        engine: TradingEngine,
        decision_maker: DecisionMaker,
        memory_manager: MemoryManager,
        checkpoint_db: str = "data/checkpoint.db"
    ):
        self.engine = engine
        self.decision_maker = decision_maker
        self.memory = memory_manager
        self.checkpoint_db = checkpoint_db

        self.graph = None
        self._checkpointer = None

    def _build_graph(self) -> StateGraph:
        """构建学习图"""
        # 创建图
        workflow = StateGraph(TradingState)

        # 添加节点
        workflow.add_node("retrieve_memory", self._retrieve_memory)
        workflow.add_node("get_market_data", self._get_market_data)
        workflow.add_node("decide", self._decide)
        workflow.add_node("execute", self._execute)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("update_memory", self._update_memory)

        # 设置入口
        workflow.set_entry_point("get_market_data")

        # 添加边
        workflow.add_edge("get_market_data", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", "update_memory")
        workflow.add_edge("update_memory", END)

        # 异步初始化 checkpointer
        # 注意：checkpointer 将在 _ensure_graph_initialized 中创建
        return workflow

    async def _get_market_data(self, state: TradingState) -> Dict[str, Any]:
        """获取市场数据（Layer 1）"""
        cprint("\n" + "=" * 70, "cyan")
        cprint(f"📊 迭代 {state.get('iteration', 0) + 1}: 获取市场数据", "cyan")
        cprint("=" * 70, "cyan")

        snapshot = await self.engine.get_market_snapshot(state['symbols'])

        return {
            'market_snapshot': snapshot,
            'timestamp': datetime.now()
        }

    async def _retrieve_memory(self, state: TradingState) -> Dict[str, Any]:
        """检索相关记忆（Layer 3）"""
        cprint("\n🧠 检索历史记忆...", "cyan")

        snapshot = state['market_snapshot']

        # 检索相似案例
        similar_cases = self.memory.search_similar(
            snapshot.to_dict() if snapshot else {},
            k=5
        )

        # 生成记忆上下文
        memory_context = self.memory.to_context(
            recent_days=7,
            similar_cases=similar_cases
        )

        if similar_cases:
            cprint(f"✅ 找到 {len(similar_cases)} 个相似案例", "green")
        else:
            cprint("ℹ️ 没有找到相似案例", "yellow")

        return {
            'memory_context': memory_context,
            'similar_cases': similar_cases
        }

    async def _decide(self, state: TradingState) -> Dict[str, Any]:
        """做出决策（Layer 2）"""
        snapshot = state['market_snapshot']
        memory_context = state.get('memory_context')

        decision = await self.decision_maker.analyze_and_decide(
            snapshot,
            memory_context
        )

        return {'decision': decision}

    async def _execute(self, state: TradingState) -> Dict[str, Any]:
        """执行交易（Layer 1）"""
        decision = state.get('decision')
        if not decision or not decision.signals:
            cprint("\n✋ 无决策信号，跳过执行", "yellow")
            return {'execution_results': []}

        market_snapshot = state.get('market_snapshot')
        results = []

        # 检查是否有需要执行的信号（非wait的信号）
        executable_signals = [s for s in decision.signals if s.action != 'wait']
        if not executable_signals:
            cprint("\n✋ 仅有观望信号，无需执行", "yellow")
            return {'execution_results': []}

        cprint("\n" + "=" * 70, "green")
        cprint("⚡ 执行交易信号...", "green")
        cprint("=" * 70, "green")

        for signal in decision.signals:
            # 完全跳过 'wait' 信号
            if signal.action == 'wait':
                continue

            # 处理 'hold' 信号 - 检查是否需要更新止损止盈
            if signal.action == 'hold':
                # 获取当前持仓信息
                if market_snapshot and signal.symbol in market_snapshot.assets:
                    asset = market_snapshot.assets[signal.symbol]

                    # 检查是否有持仓
                    if asset.position_size > 0:
                        current_sl = float(asset.stop_loss) if asset.stop_loss else None
                        current_tp = float(asset.take_profit) if asset.take_profit else None
                        signal_sl = float(signal.stop_loss) if signal.stop_loss else None
                        signal_tp = float(signal.take_profit) if signal.take_profit else None

                        # 判断是否需要更新止损止盈
                        # 使用更大的阈值避免频繁微调：
                        # - 绝对差异 > 1.0 美元（避免噪音）
                        # - 或相对差异 > 0.5%（对于大价格）
                        needs_update = False
                        update_reason = []

                        # 计算合理的阈值（价格的0.3%，但至少1美元）
                        current_price = float(asset.current_price) if asset.current_price else 0
                        min_threshold = max(1.0, current_price * 0.003)

                        if signal_sl is not None:
                            if current_sl is None:
                                # 缺失止损，必须添加
                                needs_update = True
                                update_reason.append(f"止损: 未设置 → {signal_sl}")
                            elif abs(signal_sl - current_sl) > min_threshold:
                                # 差异显著，需要更新
                                needs_update = True
                                update_reason.append(f"止损: {current_sl} → {signal_sl}")

                        if signal_tp is not None:
                            if current_tp is None:
                                # 缺失止盈，必须添加
                                needs_update = True
                                update_reason.append(f"止盈: 未设置 → {signal_tp}")
                            elif abs(signal_tp - current_tp) > min_threshold:
                                # 差异显著，需要更新
                                needs_update = True
                                update_reason.append(f"止盈: {current_tp} → {signal_tp}")

                        if needs_update:
                            cprint(f"\n🔧 更新 {signal.symbol} 止损止盈", "cyan")
                            cprint(f"   {', '.join(update_reason)}", "yellow")

                            # 执行更新
                            result = await self.engine.execute_signal({
                                'action': 'set_stop_loss_take_profit',
                                'symbol': signal.symbol,
                                'stop_loss': signal_sl,
                                'take_profit': signal_tp,
                            })

                            results.append({
                                'signal': signal,
                                'result': result,
                                'timestamp': datetime.now(),
                                'action_detail': 'update_sl_tp'
                            })

                            if result.get('success'):
                                cprint(f"✅ 止损止盈更新成功", "green")
                            else:
                                cprint(f"❌ 止损止盈更新失败: {result.get('error')}", "red")
                        else:
                            cprint(f"\n✋ {signal.symbol} 止损止盈无需更新", "yellow")

                continue

            # 执行其他交易信号（开仓、平仓等）
            cprint(f"\n执行: {signal.action} {signal.symbol}", "cyan")

            result = await self.engine.execute_signal({
                'action': signal.action,
                'symbol': signal.symbol,
                'amount': signal.amount,
                'leverage': signal.leverage,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
            })

            results.append({
                'signal': signal,
                'result': result,
                'timestamp': datetime.now()
            })

            if result.get('success'):
                cprint(f"✅ {signal.action} 执行成功", "green")
            else:
                cprint(f"❌ {signal.action} 执行失败: {result.get('error')}", "red")

        return {'execution_results': results}

    async def _reflect(self, state: TradingState) -> Dict[str, Any]:
        """反思和学习（Layer 3 + LLM）"""
        decision = state.get('decision')
        execution_results = state.get('execution_results', [])

        if not execution_results:
            return {
                'reflection': "本次无交易执行，无需反思",
                'lessons_learned': []
            }

        cprint("\n" + "=" * 70, "magenta")
        cprint("🤔 反思本次决策...", "magenta")
        cprint("=" * 70, "magenta")

        # 获取账户信息
        account_info = await self._get_account_info()

        # 构建反思提示（包含账户信息）
        reflection_prompt = self._build_reflection_prompt(state, account_info)

        # LLM反思
        llm = self.decision_maker.llm
        response = await llm.ainvoke([
            {"role": "system", "content": "你是一个交易系统的反思模块，负责分析交易决策和结果，提取经验教训。"},
            {"role": "user", "content": reflection_prompt}
        ])

        reflection_text = response.content
        cprint(f"\n{reflection_text}\n", "white")

        # 提取经验教训
        lessons = self._extract_lessons(reflection_text)

        cprint("=" * 70, "magenta")
        cprint("✅ 反思完成", "magenta")
        if lessons:
            cprint(f"   学到了 {len(lessons)} 条经验", "magenta")
        cprint("=" * 70, "magenta")

        return {
            'reflection': reflection_text,
            'lessons_learned': lessons
        }

    async def _get_account_info(self) -> dict:
        """获取账户信息（用于反思和总结）"""
        try:
            # 1. 获取账户余额
            balance = await self.engine.adapter.get_balance()

            # 2. 获取交易统计数据
            stats = self.engine.trade_history.get_statistics()

            # 3. 获取当前持仓（从 adapter 获取实时持仓，包含 unrealized_pnl）
            open_positions = await self.engine.adapter.get_positions()

            # 4. 格式化持仓数据（包含未实现盈亏）
            positions_data = []
            for pos in open_positions:
                positions_data.append({
                    'symbol': pos.symbol,
                    'side': pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    'unrealized_pnl': float(pos.unrealized_pnl) if pos.unrealized_pnl else 0,
                    'entry_price': float(pos.entry_price),
                    'amount': float(pos.amount)
                })

            return {
                'balance': {
                    'total': float(balance.total),
                    'available': float(balance.available),
                    'frozen': float(balance.frozen)
                },
                'statistics': stats,
                'open_positions': positions_data
            }
        except Exception as e:
            cprint(f"⚠️  获取账户信息失败: {e}", "yellow")
            return None

    async def _update_memory(self, state: TradingState) -> Dict[str, Any]:
        """更新记忆库（Layer 3）"""
        cprint("\n💾 更新记忆库...", "cyan")

        # 序列化 execution_results，将复杂对象转换为字典
        execution_results = state.get('execution_results', [])
        serializable_results = []

        for result in execution_results:
            # 处理 result 字段 - 可能是 dict 或 ExecutionResult 对象
            result_data = result['result']
            if hasattr(result_data, 'model_dump'):
                # ExecutionResult 对象，使用 Pydantic 的 model_dump
                result_dict = result_data.model_dump(mode='json')
            elif isinstance(result_data, dict):
                # 已经是字典，检查是否包含 ExecutionResult 对象
                result_dict = {}
                for key, value in result_data.items():
                    if hasattr(value, 'model_dump'):
                        result_dict[key] = value.model_dump(mode='json')
                    else:
                        result_dict[key] = value
            else:
                result_dict = {'raw': str(result_data)}

            serializable_result = {
                'signal': {
                    'action': result['signal'].action,
                    'symbol': result['signal'].symbol,
                    'amount': float(result['signal'].amount) if result['signal'].amount else None,
                    'leverage': result['signal'].leverage,
                    'stop_loss': float(result['signal'].stop_loss) if result['signal'].stop_loss else None,
                    'take_profit': float(result['signal'].take_profit) if result['signal'].take_profit else None,
                    'confidence': result['signal'].confidence,
                    'reason': result['signal'].reason,
                },
                'result': result_dict,
                'timestamp': result['timestamp'].isoformat()
            }
            serializable_results.append(serializable_result)

        # 创建交易案例
        case = TradingCase(
            market_conditions=state['market_snapshot'].to_dict() if state.get('market_snapshot') else {},
            decision=state['decision'].analysis if state.get('decision') else "",
            execution_result=serializable_results,
            reflection=state.get('reflection'),
            lessons_learned=state.get('lessons_learned', []),
            timestamp=state.get('timestamp', datetime.now())
        )

        # 添加到记忆（后台异步执行，避免阻塞）
        asyncio.create_task(self._save_case_background(case))

        cprint(f"✅ 案例已提交保存（后台执行）: {case.case_id}", "green")

        return {}

    @staticmethod
    def _build_reflection_prompt(state: TradingState, account_info: dict = None) -> str:
        """构建反思提示"""
        lines = [
            "请反思以下交易过程：",
            "",
            "## 账户状态"
        ]

        # 添加账户信息
        if account_info:
            balance = account_info.get('balance', {})
            stats = account_info.get('statistics', {})
            positions = account_info.get('open_positions', [])

            lines.append(f"- 账户余额: ${balance.get('total', 0):.2f}")
            lines.append(f"- 可用资金: ${balance.get('available', 0):.2f}")
            lines.append(f"- 冻结保证金: ${balance.get('frozen', 0):.2f}")

            if stats:
                lines.append(f"- 累计平仓次数: {stats.get('total_positions', 0)} 次")
                lines.append(f"  - 盈利次数: {stats.get('win_count', 0)} 次")
                lines.append(f"  - 亏损次数: {stats.get('loss_count', 0)} 次")
                lines.append(f"  - 胜率: {stats.get('win_rate', 0) * 100:.1f}%")
                lines.append(f"- 已实现总盈亏: ${stats.get('total_pnl', 0):.2f}")
                lines.append(f"  - 最大盈利: ${stats.get('max_profit', 0):.2f}")
                lines.append(f"  - 最大亏损: ${stats.get('max_loss', 0):.2f}")

            if positions:
                unrealized_total = sum(float(p.get('unrealized_pnl', 0)) for p in positions)
                lines.append(f"- 当前持仓: {len(positions)} 个")
                lines.append(f"- 未实现盈亏: ${unrealized_total:.2f}")
                for pos in positions:
                    side_name = "做多" if pos.get('side') == 'long' else "做空"
                    lines.append(f"  - {pos.get('symbol')}: {side_name}, 盈亏 ${pos.get('unrealized_pnl', 0):.2f}")
        else:
            lines.append("（账户信息获取失败）")

        lines.extend([
            "",
            "## 市场条件",
            state['market_snapshot'].to_text() if state.get('market_snapshot') else "N/A",
            "",
            "## 决策",
            state['decision'].analysis if state.get('decision') else "N/A",
            "",
            "## 执行结果"
        ])

        for result in state.get('execution_results', []):
            signal = result['signal']
            lines.append(f"- {signal.action} {signal.symbol}: {'成功' if result['result'].get('success') else '失败'}")
            if not result['result'].get('success'):
                lines.append(f"  错误: {result['result'].get('error')}")

        lines.append("")
        lines.append("## 请分析：")
        lines.append("1. 结合账户状态，这次决策合理吗？")
        lines.append("2. 当前盈亏情况如何影响下次决策？")
        lines.append("3. 学到了什么经验？（请用简短的一句话总结，每条经验单独一行）")

        return "\n".join(lines)

    @staticmethod
    def _extract_lessons(reflection_text: str) -> List[str]:
        """从反思文本中提取经验教训"""
        lessons = []

        # 简单的文本解析
        lines = reflection_text.split('\n')
        in_lessons_section = False

        for line in lines:
            line = line.strip()
            if '学到' in line or '经验' in line or 'lesson' in line.lower():
                in_lessons_section = True
                continue

            if in_lessons_section and line:
                # 提取列表项
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    lesson = line.lstrip('-*• ').strip()
                    if lesson:
                        lessons.append(lesson)
                elif line[0].isdigit() and '.' in line[:3]:
                    lesson = line.split('.', 1)[1].strip()
                    if lesson:
                        lessons.append(lesson)

        return lessons[:5]  # 最多保留5条

    async def _ensure_graph_initialized(self):
        """确保 graph 已初始化（异步）"""
        if self.graph is None:
            import aiosqlite

            # 创建异步 SQLite 连接（独立的 checkpoint 数据库）
            conn = await aiosqlite.connect(self.checkpoint_db, timeout=30.0)

            # 启用 WAL 模式以支持并发读写
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA busy_timeout=30000')  # 30秒
            await conn.commit()

            self._checkpointer = AsyncSqliteSaver(conn)

            # 编译 graph
            workflow = self._build_graph()
            self.graph = workflow.compile(checkpointer=self._checkpointer)

            cprint(f"✅ Graph 和 Checkpointer (AsyncSqliteSaver) 初始化完成: {self.checkpoint_db}", "green")

    async def run_iteration(self, symbols: List[str], iteration: int = 0):
        """运行一次迭代"""
        # 确保 graph 已初始化
        await self._ensure_graph_initialized()

        initial_state = {
            'symbols': symbols,
            'iteration': iteration,
            'timestamp': datetime.now(),
        }

        config = {"configurable": {"thread_id": "trading_thread"}}

        try:
            result = await self.graph.ainvoke(initial_state, config)
            return result
        except Exception as e:
            cprint(f"❌ 迭代执行失败: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    async def _save_case_background(self, case: TradingCase):
        """后台保存案例（避免阻塞主流程）"""
        try:
            # 在线程池中执行同步的 add_case
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.memory.add_case, case)

            cprint(f"✅ 案例保存成功: {case.case_id}", "green")

            # 保存成功后，检查是否需要生成摘要
            await self._check_and_generate_summary()

        except Exception as e:
            cprint(f"⚠️  案例保存失败: {e}", "red")
            # 失败不影响主流程，只记录错误

    async def _check_and_generate_summary(self):
        """
        智能检查是否需要生成摘要

        触发条件（满足任一即触发）：
        1. 首次摘要：累积 20 个案例
        2. 时间周期：距离上次摘要 ≥ 7 天（且有至少 10 个新案例）
        3. 案例数量：每 50 个案例
        """
        try:
            stats = self.memory._get_db_stats()
            cases_count = stats['cases_count']

            # 获取最近的摘要
            recent_summaries = self.memory._get_recent_summaries(limit=1)

            should_generate = False
            reason = ""

            if not recent_summaries:
                # 首次摘要：累积 20 个案例
                if cases_count >= 20:
                    should_generate = True
                    reason = f"首次摘要（累积 {cases_count} 个案例）"
            else:
                last_summary = recent_summaries[0]
                days_since = (datetime.now() - last_summary.period_end).days
                cases_since = cases_count - last_summary.total_cases

                # 条件1：时间周期（7天）
                if days_since >= 7 and cases_since >= 10:
                    should_generate = True
                    reason = f"距离上次摘要 {days_since} 天（新增 {cases_since} 个案例）"

                # 条件2：案例数阈值（每50个）
                elif cases_count > 0 and cases_count % 50 == 0:
                    should_generate = True
                    reason = f"案例数达到 {cases_count}"

            if should_generate:
                cprint(f"\n📝 触发摘要生成: {reason}", "yellow")
                await self._generate_summary_background()

        except Exception as e:
            # 静默失败，不影响主流程
            cprint(f"⚠️  摘要检查失败: {e}", "red")

    async def _generate_summary_background(self):
        """后台生成摘要（不阻塞主流程）"""
        try:
            cprint("🔄 开始生成摘要（后台任务）...", "cyan")

            # 获取账户信息
            account_info = await self._get_account_info()

            # 生成摘要（包含账户信息）
            summary = await self.memory.generate_weekly_summary(account_info)

            if summary:
                cprint(f"✅ 摘要生成完成: {summary.summary_id}", "green")
                cprint(f"   覆盖案例: {summary.total_cases} 个", "green")
                cprint(f"   真实交易: {summary.total_trades} 笔", "green")
                cprint(f"   关键模式: {len(summary.key_patterns)} 条", "green")
                cprint(f"   成功策略: {len(summary.successful_strategies)} 条", "green")
                cprint(f"   核心经验: {len(summary.lessons)} 条", "green")
            else:
                cprint("ℹ️  案例数不足，跳过摘要生成", "yellow")

        except Exception as e:
            cprint(f"❌ 摘要生成失败: {e}", "red")
            import traceback
            traceback.print_exc()

    async def run_loop(
        self,
        symbols: List[str],
        interval_seconds: int = 180,
        max_iterations: Optional[int] = None
    ):
        """运行交易循环"""
        iteration = 0

        cprint(f"📊 监控币种: {', '.join(symbols)}", "cyan")
        cprint(f"⏱️ 循环间隔: {interval_seconds}秒 ({interval_seconds / 60:.1f}分钟)", "cyan")
        cprint(f"🔄 最大迭代: {max_iterations or '无限'}", "cyan")
        cprint("")

        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    cprint(f"\n✅ 达到最大迭代次数 ({max_iterations})", "green")
                    break

                await self.run_iteration(symbols, iteration)

                iteration += 1

                if max_iterations is None or iteration < max_iterations:
                    cprint(f"\n⏳ 等待 {interval_seconds} 秒进入下一轮...\n", "cyan")
                    await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            cprint("\n⚠️  收到中断信号，正在停止...", "yellow")
        except Exception as e:
            cprint(f"\n❌ 循环出错: {e}", "red")
            raise
        finally:
            cprint("\n👋 交易系统已停止\n", "yellow")
