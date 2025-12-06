"""基于 LangGraph 的决策协调器。

使用 LangGraph StateGraph 管理决策流程，
通过 Checkpointer 自动持久化状态。
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from loguru import logger

from .state import (
    TradingState,
    create_initial_state,
    state_to_compose_context,
    create_memory_from_result,
)
from .workflow import (
    create_trading_graph,
    get_sqlite_checkpointer_context,
    get_thread_config,
)

from ..db import PersistenceService, get_persistence_service
from ..decision import BaseComposer
from ..execution import BaseExecutionGateway
from ..history import BaseDigestBuilder, BaseHistoryRecorder
from ..market import BaseFeaturesPipeline
from ..models import (
    ComposeContext,
    DecisionCycleResult,
    FeatureVector,
    HistoryRecord,
    MarketType,
    PriceMode,
    PortfolioView,
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
from ..portfolio import BasePortfolioService


def _generate_uuid(prefix: str = "") -> str:
    """生成带前缀的 UUID。"""
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}-{uid}" if prefix else uid


@dataclass
class GraphCoordinatorConfig:
    """Graph 协调器配置。"""

    # Checkpointer 数据库路径
    checkpointer_db_path: str = "data/langgraph_checkpoints.db"

    # 是否启用持久化
    enable_persistence: bool = True


class GraphDecisionCoordinator:
    """基于 LangGraph 的决策协调器。

    使用 LangGraph 管理状态：
    - State 自动持久化到 SQLite
    - 策略重启时自动恢复状态
    - 决策历史（memories）由 LangGraph 管理
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
        config: Optional[GraphCoordinatorConfig] = None,
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
        self._config = config or GraphCoordinatorConfig()

        self._symbols = list(dict.fromkeys(request.trading_config.symbols))
        self._realized_pnl: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._strategy_name = request.trading_config.strategy_name or strategy_id

        # LangGraph 相关
        self._graph = None
        self._checkpointer_context = None
        self._thread_config = get_thread_config(strategy_id)

        # 从 checkpoint 恢复的 cycle_index
        self._last_cycle_index: int = 0

    async def _ensure_graph(self):
        """确保 graph 已初始化。"""
        if self._graph is not None:
            return

        # 创建 checkpointer context
        self._checkpointer_context = get_sqlite_checkpointer_context(
            self._config.checkpointer_db_path
        )
        checkpointer = await self._checkpointer_context.__aenter__()

        # 创建决策和执行回调
        async def decide_callback(context: Dict[str, Any]) -> Dict[str, Any]:
            """决策回调：调用 LLM Composer。"""
            # 从 dict 重建 ComposeContext
            compose_context = ComposeContext(
                ts=context["ts"],
                compose_id=context["compose_id"],
                strategy_id=context.get("strategy_id"),
                features=[FeatureVector(**f) for f in context.get("features", [])],
                portfolio=PortfolioView(**context["portfolio"]),
                digest=context["digest"],
                recent_decisions=context.get("recent_decisions", []),
                pending_signals=context.get("pending_signals", {}),
            )

            # 调用 composer
            result = await self._composer.compose(compose_context)

            # 转换为 dict
            return {
                "instructions": [
                    inst.model_dump(mode="json") for inst in result.instructions
                ],
                "rationale": result.rationale,
            }

        async def execute_callback(
            instructions: List[Dict[str, Any]],
            market_features: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """执行回调：调用执行网关。"""
            if not instructions:
                return []

            # 重建 TradeInstruction
            trade_instructions = [
                TradeInstruction(**inst) for inst in instructions
            ]

            # 重建 FeatureVector
            features = [FeatureVector(**f) for f in market_features]

            # 执行
            tx_results = await self._execution_gateway.execute(
                trade_instructions, market_features=features
            )

            # 创建交易记录
            trades = self._create_trades(
                tx_results,
                instructions[0].get("compose_id", "") if instructions else "",
                get_current_timestamp_ms(),
            )

            # 应用交易到仓位
            self.portfolio_service.apply_trades(trades, features)

            # 返回 dict 形式
            return [t.model_dump(mode="json") for t in trades]

        # 创建 graph
        self._graph = create_trading_graph(
            decide_callback=decide_callback,
            execute_callback=execute_callback,
            checkpointer=checkpointer,
        )

        # 尝试恢复状态
        try:
            saved_state = await self._graph.aget_state(self._thread_config)
            if saved_state.values:
                self._last_cycle_index = saved_state.values.get("cycle_index", 0)
                memories_count = len(saved_state.values.get("memories", []))
                logger.info(
                    f"从 LangGraph checkpoint 恢复状态: "
                    f"cycle={self._last_cycle_index}, memories={memories_count}"
                )
        except Exception:
            logger.debug("没有找到已保存的 checkpoint 状态")

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
                trade_id=_generate_uuid("trade"),
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

    async def run_once(self) -> DecisionCycleResult:
        """执行一个决策周期。"""
        await self._ensure_graph()

        timestamp_ms = get_current_timestamp_ms()
        compose_id = _generate_uuid("compose")

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

        # 构建摘要
        digest = self._digest_builder.build(self._history_recorder.get_records())

        # 获取当前保存的状态（用于获取 cycle_index 和 memories）
        saved_state = await self._graph.aget_state(self._thread_config)
        current_cycle = saved_state.values.get("cycle_index", 0) if saved_state.values else 0

        # 构建 graph 输入状态
        input_state = {
            "strategy_id": self.strategy_id,
            "compose_id": compose_id,
            "timestamp_ms": timestamp_ms,
            "features": [f.model_dump(mode="json") for f in features],
            "portfolio": portfolio.model_dump(mode="json"),
            "digest": digest.model_dump(mode="json"),
            "instructions": [],
            "rationale": None,
            "trades": [],
            # 记忆数据：保留 cycle_index，memories 由 reducer 管理
            "memories": [],  # 空列表，reducer 会保留旧的
            "pending_signals": {},
            "cycle_index": current_cycle,
        }

        # 运行 graph
        result = await self._graph.ainvoke(input_state, self._thread_config)

        # 提取结果
        instructions = [
            TradeInstruction(**inst) for inst in result.get("instructions", [])
        ]
        trades = [
            TradeHistoryEntry(**t) for t in result.get("trades", [])
        ]
        rationale = result.get("rationale")
        cycle_index = result.get("cycle_index", current_cycle + 1)

        self._last_cycle_index = cycle_index

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
        portfolio = self.portfolio_service.get_view()

        # 持久化交易和持仓数据（可选的额外持久化）
        if self._persistence_service:
            self._persist_cycle_data(trades, portfolio, timestamp_ms)

        return DecisionCycleResult(
            compose_id=compose_id,
            timestamp_ms=timestamp_ms,
            cycle_index=cycle_index,
            rationale=rationale,
            strategy_summary=summary,
            instructions=instructions,
            trades=trades,
            history_records=history_records,
            digest=digest,
            portfolio_view=portfolio,
        )

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

    def _persist_cycle_data(
        self,
        trades: List[TradeHistoryEntry],
        portfolio: Any,
        timestamp_ms: int,
    ) -> None:
        """持久化决策周期数据到数据库（额外持久化，可选）。"""
        if not self._persistence_service:
            return

        try:
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
            compose_id = _generate_uuid("close_all")
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
                    instruction_id=_generate_uuid("inst"),
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
                    market_features = [
                        f for f in (pipeline_result.features or [])
                        if (f.meta or {}).get("source") == "market_snapshot"
                    ]
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
        # 关闭 checkpointer context
        if self._checkpointer_context is not None:
            try:
                await self._checkpointer_context.__aexit__(None, None, None)
            except Exception:
                pass

        # 关闭执行网关
        try:
            close_fn = getattr(self._execution_gateway, "close", None)
            if callable(close_fn):
                await close_fn()
        except Exception:
            pass

    @property
    def cycle_index(self) -> int:
        """获取当前周期索引。"""
        return self._last_cycle_index
