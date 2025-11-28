"""
学习图 (Layer 3) - 完全基于 Agno

混合架构:
- Trading Agent (确定性): 快速决策执行，使用动态 prompt
- Analyst Agent (代理式): 深度分析，生成规则

工作流程:
1. Trading Agent 执行交易 → 记录到 Knowledge Base
2. Evaluation Agent 即时评估 → 小幅调整 prompt
3. Analyst Agent 定期深度分析 → 生成规则 → 更新 Trading Agent
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

from agno.db.sqlite import SqliteDb
from termcolor import cprint

from ..engine.trading_engine import TradingEngine
from ..engine.market_snapshot import MarketSnapshot
from .trading_memory import TradingMemory, TradingCase
from .agents import TradingAgents
from .analyst import AnalystAgent


# ============================================================================
# Trading State
# ============================================================================

@dataclass
class TradingState:
    """交易状态 - 贯穿整个迭代周期"""

    # 输入
    symbols: List[str] = field(default_factory=list)

    # Layer 1 数据
    market_snapshot: Optional[MarketSnapshot] = None

    # Layer 3 记忆
    memory_context: Optional[str] = None

    # Layer 2 决策
    decision: Optional[Dict[str, Any]] = None

    # 执行结果
    execution_results: Optional[List[Dict]] = None

    # 反思
    reflection: Optional[Dict[str, Any]] = None

    # 元数据
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Learning Graph
# ============================================================================

class LearningGraph:
    """
    学习图 (完全基于 Agno) - 混合架构

    ┌─────────────────────────────────────────────────────────────┐
    │                    Hybrid Architecture                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Trading Agent (确定性)     ──▶  Knowledge Base              │
    │  - 快速决策执行                      │                       │
    │  - 动态 prompt                       ▼                       │
    │       ▲                      Analyst Agent (代理式)          │
    │       │                      - 自主检索历史                  │
    │       │                      - 深度分析                      │
    │       └──── Config Store ◀── - 生成规则                      │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

    核心组件：
    - TradingAgents: 决策 + 即时评估
    - AnalystAgent: 深度分析（独立于交易循环）
    - TradingMemory: 记忆系统
    - TradingKnowledge: 向量知识库
    """

    def __init__(
        self,
        engine: TradingEngine,
        db_path: str = "data/agno_trading.db",
        model_provider: str = "openai",
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt_path: Optional[str] = None,
        # 知识库配置
        knowledge_db_path: str = "data/trading_knowledge",
        knowledge_enabled: bool = True,
        embedder_model: str = "text-embedding-3-small",
        # 深度分析配置
        analyst_enabled: bool = True,
        analyst_model_id: str = "gpt-4o",
        analyst_interval_hours: int = 24,
    ):
        self.engine = engine
        self.model_provider = model_provider
        self.api_key = api_key
        self.base_url = base_url

        # 初始化组件
        self._init_database(db_path)
        self._init_memory(db_path, model_provider, model_id, api_key, base_url)
        self._init_agents(
            db_path, model_provider, model_id, api_key, base_url,
            temperature, system_prompt_path, knowledge_db_path,
            knowledge_enabled, embedder_model
        )
        self._init_analyst(
            analyst_enabled, analyst_model_id,
            analyst_interval_hours, model_provider, api_key, base_url
        )

        # 迭代计数器
        self.iteration_count = 0
        self.analyst_trigger_iterations = 20

        cprint("✅ LearningGraph 初始化完成 (混合架构)", "green")

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def _init_database(self, db_path: str) -> None:
        """初始化数据库"""
        self.db = SqliteDb(db_file=db_path)

    def _init_memory(
        self,
        db_path: str,
        model_provider: str,
        model_id: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> None:
        """初始化记忆系统"""
        self.memory = TradingMemory(
            db_path=db_path,
            user_id="nofn_trading",
            model_provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )

    def _init_agents(
        self,
        db_path: str,
        model_provider: str,
        model_id: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        system_prompt_path: Optional[str],
        knowledge_db_path: str,
        knowledge_enabled: bool,
        embedder_model: str,
    ) -> None:
        """初始化 Trading Agents"""
        if system_prompt_path is None:
            from pathlib import Path
            system_prompt_path = str(
                Path(__file__).parent.parent / "prompts" / "nofn_v2.txt"
            )

        self.agents = TradingAgents(
            db=self.db,
            model_provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            system_prompt_path=system_prompt_path,
            knowledge_db_path=knowledge_db_path,
            knowledge_enabled=knowledge_enabled,
            embedder_model=embedder_model,
            embedder_api_key=api_key,
            embedder_base_url=base_url,
        )

    def _init_analyst(
        self,
        analyst_enabled: bool,
        analyst_model_id: str,
        analyst_interval_hours: int,
        model_provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> None:
        """初始化 Analyst Agent"""
        self.analyst: Optional[AnalystAgent] = None
        self.analyst_enabled = analyst_enabled

        if analyst_enabled and self.agents.knowledge:
            self.analyst = AnalystAgent(
                knowledge=self.agents.knowledge,
                prompt_config=self.agents.prompt_config,
                model_provider=model_provider,
                model_id=analyst_model_id,
                api_key=api_key,
                base_url=base_url,
            )
            self.analyst.analysis_interval_hours = analyst_interval_hours

    # --------------------------------------------------------------------------
    # Main Loop
    # --------------------------------------------------------------------------

    async def run_loop(
        self,
        symbols: List[str],
        interval_seconds: int = 180,
        max_iterations: Optional[int] = None,
    ) -> None:
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

    async def run_iteration(self, symbols: List[str], iteration: int = 0) -> TradingState:
        """运行一次迭代"""
        state = TradingState(
            symbols=symbols,
            iteration=iteration,
            timestamp=datetime.now(),
        )

        try:
            # 核心交易流程
            state = await self._step_get_market_data(state)
            state = await self._step_retrieve_memory(state)
            state = await self._step_decide(state)
            state = await self._step_execute(state)
            state = await self._step_reflect(state)
            state = await self._step_update_memory(state)

            # 检查深度分析
            self.iteration_count += 1
            await self._maybe_run_deep_analysis()

            return state

        except Exception as e:
            cprint(f"❌ 迭代执行失败: {e}", "red")
            import traceback
            traceback.print_exc()
            return state

    # --------------------------------------------------------------------------
    # Trading Pipeline Steps
    # --------------------------------------------------------------------------

    async def _step_get_market_data(self, state: TradingState) -> TradingState:
        """Step 1: 获取市场数据"""
        cprint("\n" + "=" * 70, "cyan")
        cprint(f"📊 迭代 {state.iteration + 1}: 获取市场数据", "cyan")
        cprint("=" * 70, "cyan")

        self._print_performance_stats()

        snapshot = await self.engine.get_market_snapshot(state.symbols)
        state.market_snapshot = snapshot
        state.timestamp = datetime.now()

        self._print_market_data(snapshot)

        return state

    async def _step_retrieve_memory(self, state: TradingState) -> TradingState:
        """Step 2: 检索历史记忆"""
        cprint("\n🧠 检索历史记忆...", "cyan")

        snapshot = state.market_snapshot
        market_conditions = snapshot.to_dict() if snapshot else {}

        memory_context = self.memory.get_context(
            market_conditions=market_conditions,
            recent_days=7,
        )
        state.memory_context = memory_context

        similar = self.memory.search_similar(market_conditions, limit=3)
        if similar:
            cprint(f"✅ 找到 {len(similar)} 个相似案例", "green")
        else:
            cprint("ℹ️ 没有找到相似案例", "yellow")

        return state

    async def _step_decide(self, state: TradingState) -> TradingState:
        """Step 3: 做出决策"""
        cprint("\n" + "=" * 70, "cyan")
        cprint("🧠 LLM 开始分析决策...", "cyan")
        cprint("=" * 70, "cyan")

        decision = await self.agents.make_decision(
            market_snapshot=state.market_snapshot,
            memory_context=state.memory_context,
        )
        state.decision = decision

        # 打印分析和决策
        if decision.get('analysis'):
            cprint(f"\n{decision['analysis']}\n", "white")

        cprint("=" * 70, "green")
        cprint(f"✅ 决策完成: {decision.get('decision_type', 'wait')}", "green")
        cprint("=" * 70, "green")

        self._print_signals(decision.get('signals', []))

        return state

    async def _step_execute(self, state: TradingState) -> TradingState:
        """Step 4: 执行交易"""
        decision = state.decision
        if not decision or not decision.get('signals'):
            cprint("\n✋ 无决策信号，跳过执行", "yellow")
            state.execution_results = []
            return state

        signals = decision['signals']
        executable = [s for s in signals if s.get('action') != 'wait']

        if not executable:
            cprint("\n✋ 仅有观望信号，无需执行", "yellow")
            state.execution_results = []
            return state

        cprint("\n" + "=" * 70, "green")
        cprint("⚡ 执行交易信号...", "green")
        cprint("=" * 70, "green")

        results = []
        for signal in signals:
            action = signal.get('action', 'wait')

            if action == 'wait':
                continue

            if action == 'hold':
                result = await self._handle_hold_signal(signal, state.market_snapshot)
                if result:
                    results.append(result)
                continue

            # 执行交易
            cprint(f"\n执行: {action} {signal.get('symbol', '')}", "cyan")

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

        state.execution_results = results
        return state

    async def _step_reflect(self, state: TradingState) -> TradingState:
        """Step 5: 评估决策并调整策略"""
        execution_results = state.execution_results or []

        if not execution_results:
            state.reflection = {
                'reflection': "本次无交易执行，无需反思",
                'lessons': [],
                'quality_score': 50,
            }
            return state

        cprint("\n" + "=" * 70, "magenta")
        cprint("🔍 评估决策并调整策略...", "magenta")
        cprint("=" * 70, "magenta")

        account_info = await self._get_account_info()

        evaluation = await self.agents.evaluate_and_adjust(
            decision=state.decision,
            execution_results=execution_results,
            account_info=account_info,
            market_snapshot=state.market_snapshot,
        )

        state.reflection = {
            'reflection': evaluation.get('analysis', ''),
            'lessons': evaluation.get('lessons', []),
            'quality_score': evaluation.get('quality_score', 50),
        }

        cprint(f"\n{evaluation.get('analysis', '')}\n", "white")

        cprint("=" * 70, "magenta")
        cprint("✅ 评估完成", "magenta")
        if evaluation.get('lessons'):
            cprint(f"   学到了 {len(evaluation['lessons'])} 条经验", "magenta")
        cprint(f"   决策质量: {evaluation.get('quality_score', 50)}/100", "magenta")

        if evaluation.get('prompt_updated'):
            cprint("   🔄 决策策略已动态调整", "yellow")
        cprint("=" * 70, "magenta")

        return state

    async def _step_update_memory(self, state: TradingState) -> TradingState:
        """Step 6: 更新记忆"""
        cprint("\n💾 更新记忆库...", "cyan")

        # 序列化执行结果
        serializable_results = self._serialize_execution_results(
            state.execution_results or []
        )

        # 创建交易案例
        reflection = state.reflection or {}
        case = TradingCase(
            market_conditions=state.market_snapshot.to_dict() if state.market_snapshot else {},
            decision=state.decision.get('analysis', '') if state.decision else '',
            execution_result=serializable_results,
            reflection=reflection.get('reflection', ''),
            lessons_learned=reflection.get('lessons', []),
            timestamp=state.timestamp,
        )

        # 后台保存
        asyncio.create_task(self._save_case_background(case, state))

        cprint(f"✅ 案例已提交保存: {case.case_id}", "green")

        return state

    # --------------------------------------------------------------------------
    # Deep Analysis
    # --------------------------------------------------------------------------

    async def run_deep_analysis(self, force: bool = True) -> Optional[Any]:
        """手动触发深度分析"""
        if not self.analyst:
            cprint("⚠️  AnalystAgent 未启用", "yellow")
            return None

        stats = self.engine.trade_history.get_statistics()
        return await self.analyst.run_analysis(
            period_days=7,
            account_stats=stats,
            force=force,
        )

    async def _maybe_run_deep_analysis(self) -> None:
        """检查并执行深度分析（后台）"""
        if not self.analyst or not self.analyst_enabled:
            return

        should_analyze = (
            self.iteration_count % self.analyst_trigger_iterations == 0 or
            self.analyst._should_analyze()
        )

        if should_analyze:
            asyncio.create_task(self._run_deep_analysis_background())

    async def _run_deep_analysis_background(self) -> None:
        """后台执行深度分析"""
        try:
            stats = self.engine.trade_history.get_statistics()

            report = await self.analyst.run_analysis(
                period_days=7,
                account_stats=stats,
                force=False,
            )

            if report and report.config_changes:
                self.agents.update_decision_agent_prompt()
                cprint("🔄 Trading Agent 已应用新策略", "yellow")

        except Exception as e:
            cprint(f"⚠️  深度分析失败: {e}", "yellow")

    # --------------------------------------------------------------------------
    # Signal Handlers
    # --------------------------------------------------------------------------

    async def _handle_hold_signal(
        self,
        signal: Dict,
        market_snapshot: MarketSnapshot,
    ) -> Optional[Dict]:
        """处理 hold 信号 - 检查是否需要更新止损止盈"""
        signal_symbol = signal.get('symbol', '')
        if not signal_symbol:
            return None

        # 匹配资产
        base_symbol = signal_symbol.split('/')[0]
        matched_asset = None

        for asset_symbol, asset in market_snapshot.assets.items():
            if asset_symbol.startswith(base_symbol + '/'):
                matched_asset = asset
                break

        if not matched_asset:
            cprint(f"⚠️  {signal_symbol} 未在市场快照中找到对应资产", "yellow")
            return None

        # 获取当前止损止盈
        if matched_asset.position_size <= 0:
            position = await self.engine.adapter.get_position(signal_symbol)
            if not position or position.amount <= 0:
                return None
            current_sl = float(position.stop_loss) if position.stop_loss else None
            current_tp = float(position.take_profit) if position.take_profit else None
            current_price = float(matched_asset.current_price) if matched_asset.current_price else 0
        else:
            current_sl = float(matched_asset.stop_loss) if matched_asset.stop_loss else None
            current_tp = float(matched_asset.take_profit) if matched_asset.take_profit else None
            current_price = float(matched_asset.current_price) if matched_asset.current_price else 0

        # 检查是否需要更新
        signal_sl = signal.get('stop_loss')
        signal_tp = signal.get('take_profit')
        min_threshold = max(1.0, current_price * 0.003)

        needs_update, update_reason = self._check_sl_tp_update(
            current_sl, current_tp, signal_sl, signal_tp, min_threshold
        )

        if not needs_update:
            cprint(f"\n✋ {signal_symbol} 止损止盈无需更新", "yellow")
            return None

        cprint(f"\n🔧 更新 {signal_symbol} 止损止盈", "cyan")
        cprint(f"   {', '.join(update_reason)}", "yellow")

        result = await self.engine.execute_signal({
            'action': 'set_stop_loss_take_profit',
            'symbol': signal_symbol,
            'stop_loss': signal_sl,
            'take_profit': signal_tp,
        })

        if result.get('success'):
            cprint("✅ 止损止盈更新成功", "green")
        else:
            cprint(f"❌ 止损止盈更新失败: {result.get('error')}", "red")

        return {
            'signal': signal,
            'result': result,
            'timestamp': datetime.now(),
            'action_detail': 'update_sl_tp',
        }

    def _check_sl_tp_update(
        self,
        current_sl: Optional[float],
        current_tp: Optional[float],
        signal_sl: Optional[float],
        signal_tp: Optional[float],
        threshold: float,
    ) -> tuple[bool, List[str]]:
        """检查是否需要更新止损止盈"""
        needs_update = False
        update_reason = []

        if signal_sl is not None:
            if current_sl is None:
                needs_update = True
                update_reason.append(f"止损: 未设置 → {signal_sl}")
            elif abs(float(signal_sl) - current_sl) > threshold:
                needs_update = True
                update_reason.append(f"止损: {current_sl} → {signal_sl}")

        if signal_tp is not None:
            if current_tp is None:
                needs_update = True
                update_reason.append(f"止盈: 未设置 → {signal_tp}")
            elif abs(float(signal_tp) - current_tp) > threshold:
                needs_update = True
                update_reason.append(f"止盈: {current_tp} → {signal_tp}")

        return needs_update, update_reason

    # --------------------------------------------------------------------------
    # Memory Helpers
    # --------------------------------------------------------------------------

    def _serialize_execution_results(self, results: List[Dict]) -> List[Dict]:
        """序列化执行结果"""
        serializable = []

        for result in results:
            result_data = result.get('result', {})

            if hasattr(result_data, 'model_dump'):
                result_dict = result_data.model_dump(mode='json')
            elif isinstance(result_data, dict):
                result_dict = {
                    k: (v.model_dump(mode='json') if hasattr(v, 'model_dump') else v)
                    for k, v in result_data.items()
                }
            else:
                result_dict = {'raw': str(result_data)}

            signal = result.get('signal', {})
            serializable.append({
                'signal': signal if isinstance(signal, dict) else {
                    'action': getattr(signal, 'action', 'N/A'),
                    'symbol': getattr(signal, 'symbol', ''),
                },
                'result': result_dict,
                'timestamp': result.get('timestamp', datetime.now()).isoformat(),
            })

        return serializable

    async def _save_case_background(self, case: TradingCase, state: TradingState) -> None:
        """后台保存案例"""
        try:
            # 保存到 Memory
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.memory.add_case, case)
            cprint(f"✅ 案例保存成功 (Memory): {case.case_id}", "green")

            # 保存到 Knowledge
            if self.agents.knowledge and self.agents.knowledge_enabled and state.market_snapshot:
                await self._save_to_knowledge(case, state)

            # 检查摘要
            await self._check_and_generate_summary()

        except Exception as e:
            cprint(f"⚠️  案例保存失败: {e}", "red")

    async def _save_to_knowledge(self, case: TradingCase, state: TradingState) -> None:
        """保存到知识库"""
        try:
            # 获取主要标的
            primary_symbol = ""
            if state.execution_results:
                for r in state.execution_results:
                    sig = r.get('signal', {})
                    if isinstance(sig, dict) and sig.get('symbol'):
                        primary_symbol = sig.get('symbol')
                        break
            if not primary_symbol and state.market_snapshot.assets:
                primary_symbol = list(state.market_snapshot.assets.keys())[0]

            await self.agents.save_case_to_knowledge(
                case_id=case.case_id,
                symbol=primary_symbol,
                market_snapshot=state.market_snapshot,
                decision=state.decision or {},
                execution_results=state.execution_results or [],
                reflection=state.reflection,
            )
        except Exception as e:
            cprint(f"⚠️  保存到知识库失败: {e}", "yellow")

    async def _check_and_generate_summary(self) -> None:
        """检查是否需要生成摘要"""
        try:
            stats = self.memory.get_statistics()
            recent_cases = stats.get('recent_cases', 0)

            if recent_cases > 0 and recent_cases % 20 == 0:
                cprint(f"\n📝 触发摘要生成 (案例数: {recent_cases})", "yellow")
                await self._generate_summary_background()

        except Exception as e:
            cprint(f"⚠️  摘要检查失败: {e}", "red")

    async def _generate_summary_background(self) -> None:
        """后台生成摘要"""
        try:
            cprint("🔄 开始生成摘要...", "cyan")
            account_info = await self._get_account_info()
            summary = await self.memory.generate_summary(account_info)

            if summary:
                cprint("✅ 摘要生成完成", "green")
            else:
                cprint("ℹ️  案例数不足，跳过摘要", "yellow")

        except Exception as e:
            cprint(f"❌ 摘要生成失败: {e}", "red")

    async def _get_account_info(self) -> Optional[Dict]:
        """获取账户信息"""
        try:
            balance = await self.engine.adapter.get_balance()
            stats = self.engine.trade_history.get_statistics()
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

    # --------------------------------------------------------------------------
    # Display Helpers
    # --------------------------------------------------------------------------

    def _print_performance_stats(self) -> None:
        """打印绩效统计"""
        stats_7d = self.engine.trade_history.get_statistics(days=7)
        stats_all = self.engine.trade_history.get_statistics()

        if stats_all['total_positions'] == 0:
            cprint("📊 暂无历史交易数据", "yellow")
            return

        cprint("\n📈 绩效统计:", "magenta")

        if stats_7d['total_positions'] > 0:
            cprint(
                f"  【7天】交易:{stats_7d['total_positions']}笔 | "
                f"胜率:{stats_7d['win_rate']*100:.1f}% | "
                f"盈亏:${stats_7d['total_pnl']:.2f} | "
                f"夏普:{stats_7d['sharpe_ratio']:.2f} | "
                f"最大回撤:${stats_7d['max_drawdown']:.2f}",
                "white"
            )

        cprint(
            f"  【总计】交易:{stats_all['total_positions']}笔 | "
            f"胜率:{stats_all['win_rate']*100:.1f}% | "
            f"盈亏:${stats_all['total_pnl']:.2f} | "
            f"夏普:{stats_all['sharpe_ratio']:.2f} | "
            f"索提诺:{stats_all['sortino_ratio']:.2f}",
            "white"
        )

        cprint(
            f"  【风险】利润因子:{stats_all['profit_factor']:.2f} | "
            f"期望值:${stats_all['expectancy']:.2f} | "
            f"盈亏比:{stats_all['risk_reward_ratio']:.2f} | "
            f"最大回撤:{stats_all['max_drawdown_percent']:.1f}%",
            "white"
        )

        cprint(
            f"  【盈亏】平均盈利:${stats_all['avg_win']:.2f} | "
            f"平均亏损:${stats_all['avg_loss']:.2f} | "
            f"最大盈利:${stats_all['max_profit']:.2f} | "
            f"最大亏损:${stats_all['max_loss']:.2f}",
            "white"
        )

    def _print_market_data(self, snapshot: MarketSnapshot) -> None:
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

            if asset.funding_rate is not None:
                cprint(
                    f"📋 资金费率: {asset.funding_rate * 100:+.4f}%  |  "
                    f"持仓量: ${(asset.open_interest or 0) / 1e6:.2f}M",
                    "white"
                )

            if asset.tf_4h:
                self._print_tf_indicators("4H", asset.tf_4h)
            if asset.tf_1h:
                self._print_tf_indicators("1H", asset.tf_1h)
            if asset.tf_15m:
                self._print_tf_indicators("15M", asset.tf_15m)

    def _print_tf_indicators(self, name: str, tf: Any) -> None:
        """打印时间框架指标"""
        # EMA 序列
        ema_lines = []
        if tf.ema8_series:
            ema_lines.append(f"EMA8:[{','.join([f'{v:.1f}' for v in tf.ema8_series])}]")
        if tf.ema21_series:
            ema_lines.append(f"EMA21:[{','.join([f'{v:.1f}' for v in tf.ema21_series])}]")
        if tf.ema50_series:
            ema_lines.append(f"EMA50:[{','.join([f'{v:.1f}' for v in tf.ema50_series])}]")

        # 其他指标
        parts = []
        if tf.rsi is not None:
            parts.append(f"RSI:{tf.rsi:.1f}")
        if tf.macd_value is not None and tf.macd_signal is not None:
            parts.append(f"MACD:{tf.macd_value:.2f}/{tf.macd_signal:.2f}/{tf.macd_histogram:.2f}")
        if tf.adx is not None:
            parts.append(f"ADX:{tf.adx:.1f}")
        if tf.stoch_k is not None and tf.stoch_d is not None:
            parts.append(f"Stoch:{tf.stoch_k:.1f}/{tf.stoch_d:.1f}")
        if tf.atr is not None:
            parts.append(f"ATR:{tf.atr:.2f}")
        if tf.bb_upper is not None and tf.bb_lower is not None:
            parts.append(f"BB:{tf.bb_lower:.2f}-{tf.bb_upper:.2f}")

        # OI
        oi_parts = []
        if hasattr(tf, 'oi_current') and tf.oi_current is not None:
            oi_parts.append(f"OI:${tf.oi_current / 1_000_000:.1f}M")
        if hasattr(tf, 'oi_change_4h') and tf.oi_change_4h is not None:
            oi_parts.append(f"4H:{tf.oi_change_4h:+.2f}%")
        if hasattr(tf, 'oi_change_24h') and tf.oi_change_24h is not None:
            oi_parts.append(f"24H:{tf.oi_change_24h:+.2f}%")

        # 输出
        cprint(f"⏱️  {name}:", "white")
        for line in ema_lines:
            cprint(f"    {line}", "white")
        if parts:
            cprint(f"    {' | '.join(parts)}", "white")
        if oi_parts:
            cprint(f"    📊 {' | '.join(oi_parts)}", "cyan")

    def _print_signals(self, signals: List[Dict]) -> None:
        """打印交易信号"""
        if not signals:
            return

        cprint(f"\n📋 交易信号 ({len(signals)} 个):", "cyan")

        for i, signal in enumerate(signals, 1):
            action = signal.get('action', 'N/A')
            symbol = signal.get('symbol', 'N/A')

            # 颜色
            if action in ['open_long', 'close_short']:
                color = "green"
            elif action in ['open_short', 'close_long']:
                color = "red"
            elif action == 'close_position':
                color = "yellow"
            else:
                color = "white"

            cprint(f"\n  [{i}] {action.upper()} {symbol}", color)
            if signal.get('amount'):
                cprint(f"      数量: {signal['amount']}", "white")
            if signal.get('leverage'):
                cprint(f"      杠杆: {signal['leverage']}x", "white")
            if signal.get('stop_loss'):
                cprint(f"      止损: ${signal['stop_loss']}", "white")
            if signal.get('take_profit'):
                cprint(f"      止盈: ${signal['take_profit']}", "white")
            cprint(f"      置信度: {signal.get('confidence', 'N/A')}%", "white")
            cprint(f"      原因: {signal.get('reason', 'N/A')}", "white")


# ============================================================================
# Backward Compatibility
# ============================================================================

TradingWorkflow = LearningGraph
