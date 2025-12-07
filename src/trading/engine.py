"""交易引擎模块。

提供策略运行时和决策协调器:
- StrategyRuntime: 策略运行时容器
- DecisionCoordinator: 决策协调器
- create_strategy_runtime: 工厂函数
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from .db import PersistenceService, get_persistence_service
from .decision import BaseComposer, LlmComposer
from .execution import BaseExecutionGateway, create_execution_gateway
from .market import BaseFeaturesPipeline, DefaultFeaturesPipeline
from .history import BaseDigestBuilder, BaseHistoryRecorder, InMemoryHistoryRecorder, RollingDigestBuilder
from .memory import ShortTermMemory
from .models import (
    ComposeContext,
    Constraints,
    DecisionCycleResult,
    FeatureVector,
    HistoryRecord,
    MarketType,
    PriceMode,
    StrategySummary,
    StrategyStatus,
    TradeDecisionAction,
    TradeHistoryEntry,
    TradeInstruction,
    TradeSide,
    TradeType,
    TradingMode,
    TxResult,
    TxStatus,
    UserRequest,
    get_current_timestamp_ms,
)
from .portfolio import BasePortfolioService, InMemoryPortfolioService


def generate_uuid(prefix: str = "") -> str:
    """生成带前缀的 UUID。"""
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}-{uid}" if prefix else uid


# =============================================================================
# 决策协调器
# =============================================================================


class DecisionCoordinator(ABC):
    """决策协调器抽象基类。"""

    @abstractmethod
    async def run_once(self) -> DecisionCycleResult:
        """执行一个决策周期。"""
        raise NotImplementedError

    @abstractmethod
    async def close_all_positions(self) -> List[TradeHistoryEntry]:
        """平掉所有持仓。"""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """释放资源。"""
        raise NotImplementedError


class DefaultDecisionCoordinator(DecisionCoordinator):
    """默认决策协调器实现。

    协调完整的决策流程:
    1. 获取仓位视图
    2. 拉取数据并计算特征
    3. 构建决策上下文
    4. 执行决策(LLM + 风控) -> 交易指令
    5. 执行指令
    6. 记录检查点并更新摘要
    7. 持久化交易和持仓数据
    """

    def __init__(
        self,
        *,
        request: UserRequest,
        strategy_id: str,
        portfolio_service: BasePortfolioService,
        features_pipeline: BaseFeaturesPipeline,
        composer: BaseComposer,
        execution_gateway: BaseExecutionGateway,
        history_recorder: BaseHistoryRecorder,
        digest_builder: BaseDigestBuilder,
        persistence_service: Optional[PersistenceService] = None,
        short_term_memory: Optional[ShortTermMemory] = None,
    ) -> None:
        self._request = request
        self.strategy_id = strategy_id
        self.portfolio_service = portfolio_service
        self._features_pipeline = features_pipeline
        self._composer = composer
        self._execution_gateway = execution_gateway
        self._history_recorder = history_recorder
        self._digest_builder = digest_builder
        self._persistence_service = persistence_service
        self._short_term_memory = short_term_memory or ShortTermMemory(max_records=10)
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self.cycle_index: int = 0
        self._strategy_name = request.trading_config.strategy_name or strategy_id

    async def run_once(self) -> DecisionCycleResult:
        """执行一个决策周期。"""
        timestamp_ms = get_current_timestamp_ms()
        compose_id = generate_uuid("compose")

        # 获取仓位视图
        portfolio = self.portfolio_service.get_view()

        # LIVE 模式: 从交易所同步资金
        try:
            if self._request.exchange_config.trading_mode == TradingMode.LIVE:
                balance = await self._execution_gateway.fetch_balance()
                free = balance.get("free", {})
                total = balance.get("total", {})
                free_cash = float(free.get("USDT", 0.0) or free.get("USD", 0.0))
                total_cash = float(total.get("USDT", 0.0) or total.get("USD", 0.0))

                if self._request.exchange_config.market_type == MarketType.SPOT:
                    portfolio.account_balance = float(free_cash)
                    portfolio.buying_power = max(0.0, float(portfolio.account_balance))
                else:
                    portfolio.account_balance = float(total_cash)
                    portfolio.buying_power = float(free_cash)
                    portfolio.free_cash = float(free_cash)
        except Exception:
            logger.warning("无法从交易所同步余额，使用缓存视图")

        # VIRTUAL 模式: 现货只能用可用资金
        if self._request.exchange_config.trading_mode == TradingMode.VIRTUAL:
            if self._request.exchange_config.market_type == MarketType.SPOT:
                portfolio.buying_power = max(0.0, float(portfolio.account_balance))

        # 构建特征
        pipeline_result = await self._features_pipeline.build()
        features = list(pipeline_result.features or [])
        market_features = self._extract_market_features(features)

        # 构建摘要
        digest = self._digest_builder.build(self._history_recorder.get_records())

        # 获取短期记忆（最近决策历史）
        recent_decisions = [
            r.to_dict() for r in self._short_term_memory.get_recent(5)
        ]
        pending_signals = self._short_term_memory.get_all_pending_signals()

        # 构建决策上下文（包含历史记忆）
        context = ComposeContext(
            ts=timestamp_ms,
            compose_id=compose_id,
            strategy_id=self.strategy_id,
            features=features,
            portfolio=portfolio,
            digest=digest,
            recent_decisions=recent_decisions,
            pending_signals=pending_signals,
        )

        # 执行决策
        compose_result = await self._composer.compose(context)
        instructions = compose_result.instructions
        rationale = compose_result.rationale
        logger.info(f"决策器返回 {len(instructions)} 条指令")

        # 执行指令
        logger.info(f"执行 {len(instructions)} 条指令")
        tx_results = await self._execution_gateway.execute(
            instructions, market_features=market_features
        )
        logger.info(f"执行网关返回 {len(tx_results)} 个结果")

        # 过滤失败的指令
        failed_ids = set()
        failure_msgs = []
        for tx in tx_results:
            if tx.status in (TxStatus.REJECTED, TxStatus.ERROR):
                failed_ids.add(tx.instruction_id)
                reason = tx.reason or "未知错误"
                msg = f"跳过 {tx.instrument.symbol} {tx.side.value}: {reason}"
                failure_msgs.append(msg)
                logger.warning(f"订单被拒: {msg}")

        if failure_msgs:
            prefix = "\n\n**执行警告:**\n"
            rationale = (
                (rationale or "")
                + prefix
                + "\n".join(f"- {msg}" for msg in failure_msgs)
            )

        if failed_ids:
            instructions = [
                inst for inst in instructions if inst.instruction_id not in failed_ids
            ]

        # 从结果创建交易记录
        trades = self._create_trades(tx_results, compose_id, timestamp_ms)

        # 应用交易到仓位
        self.portfolio_service.apply_trades(trades, market_features)

        # 构建摘要
        summary = self._build_summary(timestamp_ms, trades)

        # 创建历史记录
        history_records = self._create_history_records(
            timestamp_ms, compose_id, features, instructions, trades, summary
        )

        for record in history_records:
            self._history_recorder.record(record)

        # 重建摘要
        digest = self._digest_builder.build(self._history_recorder.get_records())
        self.cycle_index += 1

        portfolio = self.portfolio_service.get_view()

        # 记录到短期记忆
        self._short_term_memory.record_decision(
            compose_id=compose_id,
            cycle_index=self.cycle_index,
            timestamp_ms=timestamp_ms,
            instructions=instructions,
            trades=trades,
            rationale=rationale,
        )

        # 持久化交易和持仓数据
        if self._persistence_service:
            self._persist_cycle_data(trades, portfolio, timestamp_ms)

        return DecisionCycleResult(
            compose_id=compose_id,
            timestamp_ms=timestamp_ms,
            cycle_index=self.cycle_index,
            rationale=rationale,
            strategy_summary=summary,
            instructions=instructions,
            trades=trades,
            history_records=history_records,
            digest=digest,
            portfolio_view=portfolio,
        )

    def _extract_market_features(
        self, features: List[FeatureVector]
    ) -> List[FeatureVector]:
        """提取市场快照特征。"""
        return [
            f for f in features
            if (f.meta or {}).get("source") == "market_snapshot"
        ]

    def _create_trades(
        self,
        tx_results: List[TxResult],
        compose_id: str,
        timestamp_ms: int,
    ) -> List[TradeHistoryEntry]:
        """从执行结果创建交易历史记录。"""
        trades: List[TradeHistoryEntry] = []

        for tx in tx_results:
            if tx.status in (TxStatus.ERROR, TxStatus.REJECTED):
                continue

            qty = float(tx.filled_qty or 0.0)
            if qty == 0:
                continue

            price = float(tx.avg_exec_price or 0.0)
            notional = (price * qty) if price and qty else None
            fee = float(tx.fee_cost or 0.0)
            realized_pnl = -fee if notional else None

            trade = TradeHistoryEntry(
                trade_id=generate_uuid("trade"),
                compose_id=compose_id,
                instruction_id=tx.instruction_id,
                strategy_id=self.strategy_id,
                instrument=tx.instrument,
                side=tx.side,
                type=TradeType.LONG if tx.side == TradeSide.BUY else TradeType.SHORT,
                quantity=qty,
                entry_price=price or None,
                avg_exec_price=tx.avg_exec_price,
                notional_entry=notional,
                entry_ts=timestamp_ms,
                trade_ts=timestamp_ms,
                realized_pnl=realized_pnl,
                leverage=tx.leverage,
                fee_cost=fee or None,
            )
            trades.append(trade)

        return trades

    def _build_summary(
        self,
        timestamp_ms: int,
        trades: List[TradeHistoryEntry],
    ) -> StrategySummary:
        """构建周期后的策略摘要。"""
        realized_delta = sum(trade.realized_pnl or 0.0 for trade in trades)
        self._realized_pnl += realized_delta

        try:
            view = self.portfolio_service.get_view()
            unrealized = float(view.total_unrealized_pnl or 0.0)
            equity = float(view.total_value or 0.0)
        except Exception:
            unrealized = float(self._unrealized_pnl or 0.0)
            equity = float(self._request.trading_config.initial_capital or 0.0)

        self._unrealized_pnl = float(unrealized)

        initial_capital = self._request.trading_config.initial_capital or 0.0
        pnl_pct = (
            (self._realized_pnl + self._unrealized_pnl) / initial_capital
            if initial_capital
            else None
        )

        unrealized_pnl_pct = (self._unrealized_pnl / equity * 100.0) if equity else None

        return StrategySummary(
            strategy_id=self.strategy_id,
            name=self._strategy_name,
            model_provider=self._request.llm_model_config.provider,
            model_id=self._request.llm_model_config.model_id,
            exchange_id=self._request.exchange_config.exchange_id,
            mode=self._request.exchange_config.trading_mode,
            status=StrategyStatus.RUNNING,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=self._unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            pnl_pct=pnl_pct,
            total_value=equity,
            last_updated_ts=timestamp_ms,
        )

    def _create_history_records(
        self,
        timestamp_ms: int,
        compose_id: str,
        features: List[FeatureVector],
        instructions: List[TradeInstruction],
        trades: List[TradeHistoryEntry],
        summary: StrategySummary,
    ) -> List[HistoryRecord]:
        """为本周期创建历史记录。"""
        feature_payload = [v.model_dump(mode="json") for v in features]
        instruction_payload = [i.model_dump(mode="json") for i in instructions]
        trade_payload = [t.model_dump(mode="json") for t in trades]

        return [
            HistoryRecord(
                ts=timestamp_ms,
                kind="features",
                reference_id=compose_id,
                payload={"features": feature_payload},
            ),
            HistoryRecord(
                ts=timestamp_ms,
                kind="compose",
                reference_id=compose_id,
                payload={"summary": summary.model_dump(mode="json")},
            ),
            HistoryRecord(
                ts=timestamp_ms,
                kind="instructions",
                reference_id=compose_id,
                payload={"instructions": instruction_payload},
            ),
            HistoryRecord(
                ts=timestamp_ms,
                kind="execution",
                reference_id=compose_id,
                payload={"trades": trade_payload},
            ),
        ]

    def _persist_memory(self) -> None:
        """持久化短期记忆到数据库。"""
        if not self._persistence_service:
            return

        try:
            state = self._short_term_memory.to_state()
            self._persistence_service.save_memory(
                strategy_id=self.strategy_id,
                decisions=state["decisions"],
                pending_signals=state["pending_signals"],
                cycle_index=self._short_term_memory.get_last_cycle_index(),
            )
        except Exception:
            logger.exception("持久化记忆失败")

    def _persist_cycle_data(
        self,
        trades: List[TradeHistoryEntry],
        portfolio: Any,
        timestamp_ms: int,
    ) -> None:
        """持久化决策周期数据到数据库。"""
        if not self._persistence_service:
            return

        try:
            # 持久化短期记忆
            self._persist_memory()

            # 持久化交易记录
            if trades:
                trade_data = []
                for trade in trades:
                    trade_data.append({
                        "strategy_id": self.strategy_id,
                        "trade_id": trade.trade_id,
                        "compose_id": trade.compose_id,
                        "instruction_id": trade.instruction_id,
                        "symbol": trade.instrument.symbol,
                        "trade_type": trade.type.value if trade.type else "LONG",
                        "side": trade.side.value if trade.side else "BUY",
                        "leverage": trade.leverage,
                        "quantity": trade.quantity,
                        "entry_price": trade.entry_price,
                        "avg_exec_price": trade.avg_exec_price,
                        "notional_entry": trade.notional_entry,
                        "realized_pnl": trade.realized_pnl,
                        "fee_cost": trade.fee_cost,
                        "entry_time": datetime.fromtimestamp(
                            timestamp_ms / 1000, tz=timezone.utc
                        ) if timestamp_ms else None,
                    })
                self._persistence_service.save_trades_batch(trade_data)

            # 持久化持仓快照
            if portfolio and portfolio.positions:
                holdings_data = []
                snapshot_ts = datetime.fromtimestamp(
                    timestamp_ms / 1000, tz=timezone.utc
                )
                for symbol, pos in portfolio.positions.items():
                    qty = float(pos.quantity or 0)
                    if qty == 0:
                        continue
                    holdings_data.append({
                        "strategy_id": self.strategy_id,
                        "symbol": symbol,
                        "holding_type": "LONG" if qty > 0 else "SHORT",
                        "quantity": abs(qty),
                        "leverage": getattr(pos, "leverage", None),
                        "entry_price": getattr(pos, "avg_entry_price", None),
                        "unrealized_pnl": getattr(pos, "unrealized_pnl", None),
                        "unrealized_pnl_pct": getattr(pos, "unrealized_pnl_pct", None),
                        "snapshot_ts": snapshot_ts,
                    })
                if holdings_data:
                    self._persistence_service.save_holdings_batch(holdings_data)

        except Exception:
            logger.exception("持久化周期数据失败")

    async def close_all_positions(self) -> List[TradeHistoryEntry]:
        """平掉所有持仓。"""
        try:
            logger.info(f"正在平掉策略 {self.strategy_id} 的所有持仓")

            portfolio = self.portfolio_service.get_view()

            if not portfolio.positions:
                logger.info("没有持仓需要平掉")
                return []

            instructions = []
            compose_id = generate_uuid("close_all")
            timestamp_ms = get_current_timestamp_ms()

            for symbol, pos in portfolio.positions.items():
                quantity = float(pos.quantity)
                if quantity == 0:
                    continue

                side = TradeSide.SELL if quantity > 0 else TradeSide.BUY
                action = (
                    TradeDecisionAction.CLOSE_LONG
                    if quantity > 0
                    else TradeDecisionAction.CLOSE_SHORT
                )

                inst = TradeInstruction(
                    instruction_id=generate_uuid("inst"),
                    compose_id=compose_id,
                    instrument=pos.instrument,
                    action=action,
                    side=side,
                    quantity=abs(quantity),
                    price_mode=PriceMode.MARKET,
                    meta={
                        "rationale": "策略停止: 平掉所有持仓",
                        "reduceOnly": True,
                    },
                )
                instructions.append(inst)

            if not instructions:
                return []

            logger.info(f"执行 {len(instructions)} 条平仓指令")

            # 获取市场特征用于定价
            market_features: List[FeatureVector] = []
            if self._request.exchange_config.trading_mode == TradingMode.VIRTUAL:
                try:
                    pipeline_result = await self._features_pipeline.build()
                    market_features = self._extract_market_features(
                        pipeline_result.features or []
                    )
                except Exception:
                    logger.exception("构建平仓市场特征失败")

            # 执行
            tx_results = await self._execution_gateway.execute(
                instructions, market_features=market_features
            )

            # 创建交易记录
            trades = self._create_trades(tx_results, compose_id, timestamp_ms)
            self.portfolio_service.apply_trades(trades, market_features=[])

            # 记录历史
            for trade in trades:
                self._history_recorder.record(
                    HistoryRecord(
                        ts=timestamp_ms,
                        kind="execution",
                        reference_id=compose_id,
                        payload={"trades": [trade.model_dump(mode="json")]},
                    )
                )

            logger.info(f"成功平仓，生成 {len(trades)} 笔交易")
            return trades

        except Exception:
            logger.exception(f"平仓失败: 策略 {self.strategy_id}")
            return []

    async def close(self) -> None:
        """释放资源。"""
        # 关闭前保存记忆
        self._persist_memory()

        try:
            close_fn = getattr(self._execution_gateway, "close", None)
            if callable(close_fn):
                await close_fn()
        except Exception:
            pass


# =============================================================================
# 策略运行时
# =============================================================================


@dataclass
class StrategyRuntime:
    """策略运行时容器。"""

    request: UserRequest
    strategy_id: str
    coordinator: DefaultDecisionCoordinator
    history_recorder: Optional[InMemoryHistoryRecorder] = None

    async def run_cycle(self) -> DecisionCycleResult:
        """运行一个决策周期。"""
        return await self.coordinator.run_once()


async def _fetch_free_cash(
    gateway: BaseExecutionGateway,
    symbols: list,
) -> tuple[float, float]:
    """从执行网关获取可用资金。"""
    try:
        balance = await gateway.fetch_balance()
        free = balance.get("free", {})
        total = balance.get("total", {})

        free_cash = 0.0
        total_cash = 0.0
        for ccy in ["USDT", "USD", "USDC", "BUSD"]:
            if ccy in free:
                free_cash = float(free[ccy] or 0.0)
                total_cash = float(total.get(ccy, 0.0) or 0.0)
                if free_cash > 0:
                    break

        return free_cash, total_cash
    except Exception as e:
        logger.warning(f"获取余额失败: {e}")
        return 0.0, 0.0


async def create_strategy_runtime(
    request: UserRequest,
    composer: Optional[BaseComposer] = None,
    features_pipeline: Optional[BaseFeaturesPipeline] = None,
    strategy_id_override: Optional[str] = None,
) -> StrategyRuntime:
    """创建策略运行时（异步初始化）。

    Args:
        request: 用户请求配置
        composer: 自定义决策器
        features_pipeline: 自定义特征管道
        strategy_id_override: 复用的策略 ID

    Returns:
        初始化后的 StrategyRuntime
    """
    # 创建执行网关（交易所）
    execution_gateway = await create_execution_gateway(request.exchange_config)

    # LIVE 模式: 从交易所获取初始资金
    try:
        if request.exchange_config.trading_mode == TradingMode.LIVE:
            free_cash, _ = await _fetch_free_cash(
                execution_gateway,
                request.trading_config.symbols,
            )
            if free_cash > 0:
                request.trading_config.initial_capital = float(free_cash)
    except Exception:
        logger.exception(
            "LIVE 模式获取交易所余额失败，将使用配置的 initial_capital"
        )

    # 验证 LIVE 模式的初始资金
    if request.exchange_config.trading_mode == TradingMode.LIVE:
        initial_cap = request.trading_config.initial_capital or 0.0
        if initial_cap <= 0:
            logger.error(
                f"LIVE 模式 initial_capital={initial_cap}，"
                "没有资金将无法交易"
            )

    # 生成或使用提供的策略 ID
    strategy_id = strategy_id_override or generate_uuid("strategy")

    # 初始化资金
    initial_capital = request.trading_config.initial_capital or 0.0

    # 创建约束
    constraints = Constraints(
        max_positions=request.trading_config.max_positions,
        max_leverage=request.trading_config.max_leverage,
    )

    # 创建仓位服务
    portfolio_service = InMemoryPortfolioService(
        initial_capital=initial_capital,
        trading_mode=request.exchange_config.trading_mode,
        market_type=request.exchange_config.market_type,
        constraints=constraints,
        strategy_id=strategy_id,
    )

    # 使用自定义决策器或默认 LlmComposer
    if composer is None:
        composer = LlmComposer(request=request)

    # 使用自定义特征管道或默认
    if features_pipeline is None:
        features_pipeline = DefaultFeaturesPipeline.from_request(request)

    # 创建历史和摘要组件
    history_recorder = InMemoryHistoryRecorder()
    digest_builder = RollingDigestBuilder()

    # 初始化持久化服务
    persistence_service = get_persistence_service()

    # 尝试恢复短期记忆（如果是复用策略）
    short_term_memory: Optional[ShortTermMemory] = None
    if strategy_id_override:
        try:
            memory_state = persistence_service.load_memory(strategy_id)
            if memory_state:
                short_term_memory = ShortTermMemory.from_state(memory_state)
                logger.info(
                    f"从数据库恢复策略记忆: {strategy_id}, "
                    f"{len(short_term_memory)} 条决策记录"
                )
        except Exception:
            logger.exception(f"恢复策略记忆失败: {strategy_id}")

    # 创建策略记录（新策略）
    if not strategy_id_override:
        try:
            config_dict = request.model_dump(mode="json")
            persistence_service.create_strategy(
                strategy_id=strategy_id,
                name=request.trading_config.strategy_name,
                description=f"策略 {strategy_id}",
                config=config_dict,
                metadata={
                    "symbols": request.trading_config.symbols,
                    "mode": request.exchange_config.trading_mode.value,
                    "market_type": request.exchange_config.market_type.value,
                },
            )
            logger.info(f"策略 {strategy_id} 已持久化到数据库")
        except Exception:
            logger.exception("创建策略记录失败")

    # 创建协调器
    coordinator = DefaultDecisionCoordinator(
        request=request,
        strategy_id=strategy_id,
        portfolio_service=portfolio_service,
        features_pipeline=features_pipeline,
        composer=composer,
        execution_gateway=execution_gateway,
        history_recorder=history_recorder,
        digest_builder=digest_builder,
        persistence_service=persistence_service,
        short_term_memory=short_term_memory,
    )

    return StrategyRuntime(
        request=request,
        strategy_id=strategy_id,
        coordinator=coordinator,
        history_recorder=history_recorder,
    )
