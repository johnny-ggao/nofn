"""
学习图 (Layer 3) - 完全基于 Agno

使用 Agno 的原生能力实现学习和进化系统：
- TradingAgents: 专用的决策和反思 Agent
- TradingMemory: 基于 Agno 的记忆系统
- Session State: Agno 原生会话状态管理
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


@dataclass
class TradingState:
    """交易状态"""
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


class LearningGraph:
    """
    学习图 (完全基于 Agno)

    核心组件：
    - TradingAgents: Agno Agent 集合 (决策、反思、摘要)
    - TradingMemory: Agno 原生记忆系统
    - SqliteDb: Agno 持久化存储

    工作流程：
    1. 获取市场数据 (Layer 1)
    2. 检索历史记忆 (TradingMemory)
    3. 做出决策 (TradingAgents.decision_agent)
    4. 执行交易 (Layer 1)
    5. 反思学习 (TradingAgents.reflection_agent)
    6. 更新记忆 (TradingMemory)
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
    ):
        self.engine = engine

        # 创建 Agno SqliteDb
        self.db = SqliteDb(db_file=db_path)

        # 创建 TradingMemory (基于 Agno)
        self.memory = TradingMemory(
            db_path=db_path,
            user_id="nofn_trading",
            model_provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )

        # 加载系统提示词路径
        if system_prompt_path is None:
            from pathlib import Path
            system_prompt_path = str(
                Path(__file__).parent.parent / "prompts" / "nofn_v2.txt"
            )

        # 创建 TradingAgents
        self.agents = TradingAgents(
            db=self.db,
            model_provider=model_provider,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            system_prompt_path=system_prompt_path,
        )

        cprint("✅ LearningGraph 初始化完成", "green")

    async def run_iteration(self, symbols: List[str], iteration: int = 0) -> TradingState:
        """运行一次迭代"""
        state = TradingState(
            symbols=symbols,
            iteration=iteration,
            timestamp=datetime.now(),
        )

        try:
            # Step 1: 获取市场数据
            state = await self._get_market_data(state)

            # Step 2: 检索记忆
            state = await self._retrieve_memory(state)

            # Step 3: 决策
            state = await self._decide(state)

            # Step 4: 执行
            state = await self._execute(state)

            # Step 5: 反思
            state = await self._reflect(state)

            # Step 6: 更新记忆
            state = await self._update_memory(state)

            return state

        except Exception as e:
            cprint(f"❌ 迭代执行失败: {e}", "red")
            import traceback
            traceback.print_exc()
            return state

    async def _get_market_data(self, state: TradingState) -> TradingState:
        """获取市场数据 (Layer 1)"""
        cprint("\n" + "=" * 70, "cyan")
        cprint(f"📊 迭代 {state.iteration + 1}: 获取市场数据", "cyan")
        cprint("=" * 70, "cyan")

        # 打印绩效统计（夏普率等）
        self._print_performance_stats()

        snapshot = await self.engine.get_market_snapshot(state.symbols)
        state.market_snapshot = snapshot
        state.timestamp = datetime.now()

        # 打印详细的市场数据和指标
        self._print_market_data(snapshot)

        return state

    def _print_performance_stats(self):
        """打印绩效统计（夏普率等风险调整指标）"""
        # 获取统计数据
        stats_7d = self.engine.trade_history.get_statistics(days=7)
        stats_all = self.engine.trade_history.get_statistics()

        # 只有有交易记录时才打印
        if stats_all['total_positions'] == 0:
            cprint("📊 暂无历史交易数据", "yellow")
            return

        cprint("\n📈 绩效统计:", "magenta")

        # 7天统计
        if stats_7d['total_positions'] > 0:
            cprint(f"  【7天】交易:{stats_7d['total_positions']}笔 | "
                   f"胜率:{stats_7d['win_rate']*100:.1f}% | "
                   f"盈亏:${stats_7d['total_pnl']:.2f} | "
                   f"夏普:{stats_7d['sharpe_ratio']:.2f} | "
                   f"最大回撤:${stats_7d['max_drawdown']:.2f}", "white")

        # 全部统计
        cprint(f"  【总计】交易:{stats_all['total_positions']}笔 | "
               f"胜率:{stats_all['win_rate']*100:.1f}% | "
               f"盈亏:${stats_all['total_pnl']:.2f} | "
               f"夏普:{stats_all['sharpe_ratio']:.2f} | "
               f"索提诺:{stats_all['sortino_ratio']:.2f}", "white")

        # 详细风险指标
        cprint(f"  【风险】利润因子:{stats_all['profit_factor']:.2f} | "
               f"期望值:${stats_all['expectancy']:.2f} | "
               f"盈亏比:{stats_all['risk_reward_ratio']:.2f} | "
               f"最大回撤:{stats_all['max_drawdown_percent']:.1f}%", "white")

        # 盈亏分析
        cprint(f"  【盈亏】平均盈利:${stats_all['avg_win']:.2f} | "
               f"平均亏损:${stats_all['avg_loss']:.2f} | "
               f"最大盈利:${stats_all['max_profit']:.2f} | "
               f"最大亏损:${stats_all['max_loss']:.2f}", "white")

    def _print_market_data(self, snapshot: MarketSnapshot):
        """打印市场数据和指标（仅数值，不做趋势判断）"""
        if not snapshot or not snapshot.assets:
            return

        for symbol, asset in snapshot.assets.items():
            cprint(f"\n{'─' * 70}", "yellow")
            cprint(f"📈 {symbol}", "yellow")
            cprint(f"{'─' * 70}", "yellow")

            # 价格
            cprint(f"💰 价格: ${float(asset.current_price):.2f}", "white")

            # 24小时统计
            if asset.change_24h_percent is not None:
                cprint(f"📊 24H变化: {asset.change_24h_percent:+.2f}%", "white")

            # 永续合约指标
            if asset.funding_rate is not None:
                cprint(f"📋 资金费率: {asset.funding_rate * 100:+.4f}%  |  持仓量: ${(asset.open_interest or 0) / 1e6:.2f}M", "white")

            # 4小时指标
            if asset.tf_4h:
                self._print_tf_indicators("4H", asset.tf_4h)

            # 1小时指标
            if asset.tf_1h:
                self._print_tf_indicators("1H", asset.tf_1h)

            # 15分钟指标
            if asset.tf_15m:
                self._print_tf_indicators("15M", asset.tf_15m)

    def _print_tf_indicators(self, name: str, tf):
        """打印单个时间框架指标（仅数值，不做趋势判断）"""
        # EMA 序列（最新10个点）
        ema_lines = []
        if tf.ema8_series:
            ema8_str = ",".join([f"{v:.1f}" for v in tf.ema8_series])
            ema_lines.append(f"EMA8:[{ema8_str}]")
        if tf.ema21_series:
            ema21_str = ",".join([f"{v:.1f}" for v in tf.ema21_series])
            ema_lines.append(f"EMA21:[{ema21_str}]")
        if tf.ema50_series:
            ema50_str = ",".join([f"{v:.1f}" for v in tf.ema50_series])
            ema_lines.append(f"EMA50:[{ema50_str}]")

        # 其他指标
        other_parts = []

        # RSI
        if tf.rsi is not None:
            other_parts.append(f"RSI:{tf.rsi:.1f}")

        # MACD (使用 macd_value 属性名)
        if tf.macd_value is not None and tf.macd_signal is not None and tf.macd_histogram is not None:
            other_parts.append(f"MACD:{tf.macd_value:.2f}/{tf.macd_signal:.2f}/{tf.macd_histogram:.2f}")

        # ADX
        if tf.adx is not None:
            other_parts.append(f"ADX:{tf.adx:.1f}")

        # Stochastic
        if tf.stoch_k is not None and tf.stoch_d is not None:
            other_parts.append(f"Stoch:{tf.stoch_k:.1f}/{tf.stoch_d:.1f}")

        # ATR
        if tf.atr is not None:
            other_parts.append(f"ATR:{tf.atr:.2f}")

        # Bollinger Bands
        if tf.bb_upper is not None and tf.bb_lower is not None:
            other_parts.append(f"BB:{tf.bb_lower:.2f}-{tf.bb_upper:.2f}")

        # 打印输出
        cprint(f"⏱️  {name}:", "white")
        for ema_line in ema_lines:
            cprint(f"    {ema_line}", "white")
        if other_parts:
            cprint(f"    {' | '.join(other_parts)}", "white")

    async def _retrieve_memory(self, state: TradingState) -> TradingState:
        """检索历史记忆 (TradingMemory)"""
        cprint("\n🧠 检索历史记忆...", "cyan")

        snapshot = state.market_snapshot
        market_conditions = snapshot.to_dict() if snapshot else {}

        # 使用 TradingMemory 获取上下文
        memory_context = self.memory.get_context(
            market_conditions=market_conditions,
            recent_days=7,
        )

        state.memory_context = memory_context

        # 搜索相似案例
        similar = self.memory.search_similar(market_conditions, limit=3)
        if similar:
            cprint(f"✅ 找到 {len(similar)} 个相似案例", "green")
        else:
            cprint("ℹ️ 没有找到相似案例", "yellow")

        return state

    async def _decide(self, state: TradingState) -> TradingState:
        """做出决策 (TradingAgents.decision_agent)"""
        cprint("\n" + "=" * 70, "cyan")
        cprint("🧠 LLM 开始分析决策...", "cyan")
        cprint("=" * 70, "cyan")

        decision = await self.agents.make_decision(
            market_snapshot=state.market_snapshot,
            memory_context=state.memory_context,
        )

        state.decision = decision

        # 打印 LLM 分析内容
        analysis = decision.get('analysis', '')
        if analysis:
            cprint(f"\n{analysis}\n", "white")

        # 打印决策结果
        cprint("=" * 70, "green")
        cprint(f"✅ 决策完成: {decision.get('decision_type', 'wait')}", "green")
        cprint("=" * 70, "green")

        # 打印每个信号的详细内容
        signals = decision.get('signals', [])
        if signals:
            cprint(f"\n📋 交易信号 ({len(signals)} 个):", "cyan")
            for i, signal in enumerate(signals, 1):
                action = signal.get('action', 'N/A')
                symbol = signal.get('symbol', 'N/A')
                confidence = signal.get('confidence', 'N/A')
                reason = signal.get('reason', 'N/A')

                # 根据动作类型选择颜色
                if action in ['open_long', 'close_short']:
                    action_color = "green"
                elif action in ['open_short', 'close_long']:
                    action_color = "red"
                elif action == 'close_position':
                    action_color = "yellow"
                else:
                    action_color = "white"

                cprint(f"\n  [{i}] {action.upper()} {symbol}", action_color)
                if signal.get('amount'):
                    cprint(f"      数量: {signal.get('amount')}", "white")
                if signal.get('leverage'):
                    cprint(f"      杠杆: {signal.get('leverage')}x", "white")
                if signal.get('stop_loss'):
                    cprint(f"      止损: ${signal.get('stop_loss')}", "white")
                if signal.get('take_profit'):
                    cprint(f"      止盈: ${signal.get('take_profit')}", "white")
                cprint(f"      置信度: {confidence}%", "white")
                cprint(f"      原因: {reason}", "white")

        return state

    async def _execute(self, state: TradingState) -> TradingState:
        """执行交易 (Layer 1)"""
        decision = state.decision
        if not decision or not decision.get('signals'):
            cprint("\n✋ 无决策信号，跳过执行", "yellow")
            state.execution_results = []
            return state

        signals = decision['signals']
        market_snapshot = state.market_snapshot
        results = []

        # 检查是否有需要执行的信号
        executable_signals = [s for s in signals if s.get('action') != 'wait']
        if not executable_signals:
            cprint("\n✋ 仅有观望信号，无需执行", "yellow")
            state.execution_results = []
            return state

        cprint("\n" + "=" * 70, "green")
        cprint("⚡ 执行交易信号...", "green")
        cprint("=" * 70, "green")

        for signal in signals:
            action = signal.get('action', 'wait')

            # 跳过 wait 信号
            if action == 'wait':
                continue

            # 处理 hold 信号
            if action == 'hold':
                result = await self._handle_hold_signal(signal, market_snapshot)
                if result:
                    results.append(result)
                continue

            # 执行其他交易信号
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

    async def _handle_hold_signal(
        self,
        signal: Dict,
        market_snapshot: MarketSnapshot,
    ) -> Optional[Dict]:
        """处理 hold 信号 - 检查是否需要更新止损止盈"""
        signal_symbol = signal.get('symbol', '')
        if not signal_symbol:
            return None

        # 尝试匹配 symbol（处理 USDT/USDC 不同交易对的情况）
        # 例如 signal 返回 BTC/USDC:USDC，但 market_snapshot 中是 BTC/USDT:USDT
        base_symbol = signal_symbol.split('/')[0]  # 提取基础币种如 BTC
        matched_asset = None
        matched_symbol = None

        for asset_symbol, asset in market_snapshot.assets.items():
            if asset_symbol.startswith(base_symbol + '/'):
                matched_asset = asset
                matched_symbol = asset_symbol
                break

        if not matched_asset:
            cprint(f"⚠️  {signal_symbol} 未在市场快照中找到对应资产", "yellow")
            return None

        # 检查是否有持仓（可能在不同交易对上）
        # 如果 market_snapshot 中的持仓为 0，尝试直接查询该 symbol 的持仓
        if matched_asset.position_size <= 0:
            # 直接查询 signal 指定的 symbol 的持仓
            position = await self.engine.adapter.get_position(signal_symbol)
            if not position or position.amount <= 0:
                return None
            # 使用 signal_symbol 而不是 matched_symbol
            current_sl = float(position.stop_loss) if position.stop_loss else None
            current_tp = float(position.take_profit) if position.take_profit else None
            current_price = float(matched_asset.current_price) if matched_asset.current_price else 0
        else:
            current_sl = float(matched_asset.stop_loss) if matched_asset.stop_loss else None
            current_tp = float(matched_asset.take_profit) if matched_asset.take_profit else None
            current_price = float(matched_asset.current_price) if matched_asset.current_price else 0

        # 获取信号中的止损止盈
        signal_sl = signal.get('stop_loss')
        signal_tp = signal.get('take_profit')

        needs_update = False
        update_reason = []

        # 计算阈值（价格变化超过 0.3% 才更新）
        min_threshold = max(1.0, current_price * 0.003)

        if signal_sl is not None:
            if current_sl is None:
                needs_update = True
                update_reason.append(f"止损: 未设置 → {signal_sl}")
            elif abs(float(signal_sl) - current_sl) > min_threshold:
                needs_update = True
                update_reason.append(f"止损: {current_sl} → {signal_sl}")

        if signal_tp is not None:
            if current_tp is None:
                needs_update = True
                update_reason.append(f"止盈: 未设置 → {signal_tp}")
            elif abs(float(signal_tp) - current_tp) > min_threshold:
                needs_update = True
                update_reason.append(f"止盈: {current_tp} → {signal_tp}")

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
            cprint(f"✅ 止损止盈更新成功", "green")
        else:
            cprint(f"❌ 止损止盈更新失败: {result.get('error')}", "red")

        return {
            'signal': signal,
            'result': result,
            'timestamp': datetime.now(),
            'action_detail': 'update_sl_tp',
        }

    async def _reflect(self, state: TradingState) -> TradingState:
        """反思学习 (TradingAgents.reflection_agent)"""
        execution_results = state.execution_results or []

        if not execution_results:
            state.reflection = {
                'reflection': "本次无交易执行，无需反思",
                'lessons': [],
                'quality_score': 50,
            }
            return state

        cprint("\n" + "=" * 70, "magenta")
        cprint("🤔 反思本次决策...", "magenta")
        cprint("=" * 70, "magenta")

        # 获取账户信息
        account_info = await self._get_account_info()

        # 使用 ReflectionAgent 进行反思
        reflection = await self.agents.reflect(
            decision=state.decision,
            execution_results=execution_results,
            account_info=account_info,
            market_snapshot=state.market_snapshot,
        )

        state.reflection = reflection

        cprint(f"\n{reflection.get('reflection', '')}\n", "white")

        cprint("=" * 70, "magenta")
        cprint("✅ 反思完成", "magenta")
        if reflection.get('lessons'):
            cprint(f"   学到了 {len(reflection['lessons'])} 条经验", "magenta")
        cprint(f"   决策质量: {reflection.get('quality_score', 50)}/100", "magenta")
        cprint("=" * 70, "magenta")

        return state

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

    async def _update_memory(self, state: TradingState) -> TradingState:
        """更新记忆 (TradingMemory)"""
        cprint("\n💾 更新记忆库...", "cyan")

        # 序列化执行结果
        execution_results = state.execution_results or []
        serializable_results = []

        for result in execution_results:
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
            serializable_results.append({
                'signal': signal if isinstance(signal, dict) else {
                    'action': getattr(signal, 'action', 'N/A'),
                    'symbol': getattr(signal, 'symbol', ''),
                },
                'result': result_dict,
                'timestamp': result.get('timestamp', datetime.now()).isoformat(),
            })

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

        # 添加到 TradingMemory (后台执行)
        asyncio.create_task(self._save_case_background(case))

        cprint(f"✅ 案例已提交保存: {case.case_id}", "green")

        return state

    async def _save_case_background(self, case: TradingCase):
        """后台保存案例"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.memory.add_case, case)
            cprint(f"✅ 案例保存成功: {case.case_id}", "green")

            # 检查是否需要生成摘要
            await self._check_and_generate_summary()

        except Exception as e:
            cprint(f"⚠️  案例保存失败: {e}", "red")

    async def _check_and_generate_summary(self):
        """检查是否需要生成摘要"""
        try:
            stats = self.memory.get_statistics()
            recent_cases = stats.get('recent_cases', 0)

            # 每20个案例或每周生成摘要
            if recent_cases > 0 and recent_cases % 20 == 0:
                cprint(f"\n📝 触发摘要生成 (案例数: {recent_cases})", "yellow")
                await self._generate_summary_background()

        except Exception as e:
            cprint(f"⚠️  摘要检查失败: {e}", "red")

    async def _generate_summary_background(self):
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

    async def run_loop(
        self,
        symbols: List[str],
        interval_seconds: int = 180,
        max_iterations: Optional[int] = None,
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


# 为了向后兼容，保留旧的 TradingWorkflow 名称
TradingWorkflow = LearningGraph
