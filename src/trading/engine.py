"""交易引擎模块。

提供策略运行时和决策协调器:
- StrategyRuntime: 策略运行时容器
- DecisionCoordinator: 决策协调器（基于 LangGraph）
- create_strategy_runtime: 工厂函数

记忆管理：
- 短期记忆由 LangGraph State + Checkpointer 自动管理
- 无需手动维护 ShortTermMemory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional
import uuid

from termcolor import cprint

from .db import PersistenceService, get_persistence_service
from .decision import BaseComposer, LlmComposer
from .execution import BaseExecutionGateway, create_execution_gateway
from .market import BaseFeaturesPipeline, DefaultFeaturesPipeline
from .history import BaseDigestBuilder, BaseHistoryRecorder, InMemoryHistoryRecorder, RollingDigestBuilder
from .graph.coordinator import GraphDecisionCoordinator, GraphCoordinatorConfig
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


# DefaultDecisionCoordinator 已移除
# 短期记忆现在由 LangGraph State + Checkpointer 自动管理
# 详见 src/trading/graph/coordinator.py 中的 GraphDecisionCoordinator


# =============================================================================
# 策略运行时
# =============================================================================


@dataclass
class StrategyRuntime:
    """策略运行时容器。"""

    request: UserRequest
    strategy_id: str
    coordinator: GraphDecisionCoordinator  # 使用基于 LangGraph 的协调器
    history_recorder: Optional[InMemoryHistoryRecorder] = None

    async def run_cycle(self) -> DecisionCycleResult:
        """运行一个决策周期。"""
        return await self.coordinator.run_once()


async def _fetch_free_cash(
    gateway: BaseExecutionGateway,
    symbols: list,
    settle_coin: str = "USDT",
) -> tuple[float, float]:
    """从执行网关获取可用资金。

    Args:
        gateway: 执行网关
        symbols: 交易对列表
        settle_coin: 结算货币 (USDT, USDC, USD 等)

    Returns:
        (free_cash, total_cash) 元组
    """
    try:
        balance = await gateway.fetch_balance()
        free = balance.get("free", {})
        total = balance.get("total", {})

        # 按配置的 settle_coin 优先级查询
        settle_coin = settle_coin.upper()
        quote_priority = [settle_coin]
        if settle_coin == "USDT":
            quote_priority.extend(["USD", "BUSD"])
        elif settle_coin == "USDC":
            quote_priority.extend(["USD"])
        elif settle_coin == "USD":
            quote_priority.extend(["USDT", "USDC"])
        else:
            # 默认备选
            quote_priority.extend(["USDT", "USDC", "USD"])

        free_cash = 0.0
        total_cash = 0.0
        for ccy in quote_priority:
            val = float(free.get(ccy, 0) or 0)
            if val > 0:
                free_cash = val
                total_cash = float(total.get(ccy, 0) or 0)
                cprint(f"使用 {ccy} 余额: free={free_cash}, total={total_cash}", "white")
                break

        return free_cash, total_cash
    except Exception as e:
        cprint(f"获取余额失败: {e}", "yellow")
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
                settle_coin=request.exchange_config.settle_coin,
            )
            if free_cash > 0:
                request.trading_config.initial_capital = float(free_cash)
                cprint(f"LIVE 模式检测到 {request.exchange_config.settle_coin} 余额: {free_cash}", "white")
    except Exception:
        import traceback
        cprint("LIVE 模式获取交易所余额失败，将使用配置的 initial_capital", "red")
        traceback.print_exc()

    # 验证 LIVE 模式的初始资金
    if request.exchange_config.trading_mode == TradingMode.LIVE:
        initial_cap = request.trading_config.initial_capital or 0.0
        if initial_cap <= 0:
            cprint(
                f"LIVE 模式 initial_capital={initial_cap}，没有资金将无法交易",
                "red"
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

    # 从数据库加载历史交易记录（用于夏普率等统计）
    try:
        db_trades = persistence_service.get_trades_for_digest(
            strategy_id=strategy_id,
            lookback_days=7,
        )
        if db_trades:
            loaded = history_recorder.load_from_db_trades(db_trades)
            cprint(
                f"从数据库恢复 {loaded} 条历史交易记录 (策略: {strategy_id})",
                "white"
            )
    except Exception:
        import traceback
        cprint(f"恢复历史交易记录失败: {strategy_id}", "yellow")
        traceback.print_exc()

    # 短期记忆现在由 LangGraph Checkpointer 自动管理
    # 无需手动恢复 ShortTermMemory

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
            cprint(f"策略 {strategy_id} 已持久化到数据库", "white")
        except Exception:
            import traceback
            cprint("创建策略记录失败", "red")
            traceback.print_exc()

    # 创建基于 LangGraph 的协调器
    # 短期记忆由 LangGraph State + Checkpointer 自动管理
    graph_config = GraphCoordinatorConfig(
        checkpointer_db_path=f"data/langgraph_{strategy_id}.db",
        enable_persistence=True,
    )

    coordinator = GraphDecisionCoordinator(
        request=request,
        strategy_id=strategy_id,
        portfolio_service=portfolio_service,
        features_pipeline=features_pipeline,
        composer=composer,
        execution_gateway=execution_gateway,
        history_recorder=history_recorder,
        digest_builder=digest_builder,
        persistence_service=persistence_service,
        config=graph_config,
    )

    return StrategyRuntime(
        request=request,
        strategy_id=strategy_id,
        coordinator=coordinator,
        history_recorder=history_recorder,
    )
