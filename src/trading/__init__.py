"""Trading components - models, execution, decision, portfolio, database, memory, graph.

记忆管理说明：
- 短期记忆由 LangGraph State + Checkpointer 自动管理
- DecisionMemory (graph/state.py) 替代了 ShortTermMemory
- 旧的 ShortTermMemory 类仍可用于兼容性，但不推荐使用
"""

from .db import (
    PersistenceService,
    get_persistence_service,
)
from .graph import (
    GraphDecisionCoordinator,
    GraphCoordinatorConfig,
    TradingState,
)
from .models import (
    TradingMode,
    TradeType,
    TradeSide,
    MarketType,
    MarginMode,
    TradeDecisionAction,
    TxStatus,
    PriceMode,
    StrategyStatus,
    StopReason,
    SummaryLLMConfig,
    LLMModelConfig,
    ExchangeConfig,
    TradingConfig,
    UserRequest,
    InstrumentRef,
    Candle,
    CandleConfig,
    FeatureVector,
    Constraints,
    PositionSnapshot,
    PortfolioView,
    TradeDecisionItem,
    TradePlanProposal,
    TradeInstruction,
    TxResult,
    TradeDigestEntry,
    TradeHistoryEntry,
    TradeDigest,
    StrategySummary,
    ComposeContext,
    ComposeResult,
    HistoryRecord,
    DecisionCycleResult,
    FeaturesPipelineResult,
    get_current_timestamp_ms,
)

__all__ = [
    # Database
    "PersistenceService",
    "get_persistence_service",
    # Graph (LangGraph) - 记忆管理由 LangGraph State 自动处理
    "GraphDecisionCoordinator",
    "GraphCoordinatorConfig",
    "TradingState",
    # Models
    "TradingMode",
    "TradeType",
    "TradeSide",
    "MarketType",
    "MarginMode",
    "TradeDecisionAction",
    "TxStatus",
    "PriceMode",
    "StrategyStatus",
    "StopReason",
    "SummaryLLMConfig",
    "LLMModelConfig",
    "ExchangeConfig",
    "TradingConfig",
    "UserRequest",
    "InstrumentRef",
    "Candle",
    "CandleConfig",
    "FeatureVector",
    "Constraints",
    "PositionSnapshot",
    "PortfolioView",
    "TradeDecisionItem",
    "TradePlanProposal",
    "TradeInstruction",
    "TxResult",
    "TradeDigestEntry",
    "TradeHistoryEntry",
    "TradeDigest",
    "StrategySummary",
    "ComposeContext",
    "ComposeResult",
    "HistoryRecord",
    "DecisionCycleResult",
    "FeaturesPipelineResult",
    "get_current_timestamp_ms",
]
