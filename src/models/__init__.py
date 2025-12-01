"""
数据模型模块
"""
from .enums import (
    TradingAction,
    OrderType,
    PositionSide,
    ExchangeName,
    ExecutionStatus,
)
from .trading import (
    TradingIntent,
    Position,
    ExecutionResult,
    Balance,
    Order,
    Trade,
    OrderStatus,
)
from .market import (
    Candle,
    Ticker24h,
    FundingRate,
    LatestPrice,
    OrderBook,
)
from .strategy import (
    TimeFrame,
    TrendDirection,
    MarketIndicators,
    TrendAnalysis,
    MarketAnalysis,
    StrategyDecision,
    StrategySignal,
    PerformanceMetrics,
    StrategyState,
)
from .trade_history import (
    TradeRecord,
    PositionRecord,
)
from .trade_history_manager import (
    TradeHistoryManager,
)
from .websocket_events import (
    WebSocketEventType,
    StopTriggerReason,
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeUpdateEvent,
    AccountUpdateEvent,
)

__all__ = [
    # Enums
    "TradingAction",
    "OrderType",
    "PositionSide",
    "ExchangeName",
    "ExecutionStatus",
    "OrderStatus",
    # Trading
    "TradingIntent",
    "Position",
    "ExecutionResult",
    "Balance",
    "Order",
    "Trade",
    # Market
    "Candle",
    "Ticker24h",
    "FundingRate",
    "LatestPrice",
    "OrderBook",
    # Strategy
    "TimeFrame",
    "TrendDirection",
    "MarketIndicators",
    "TrendAnalysis",
    "MarketAnalysis",
    "StrategyDecision",
    "StrategySignal",
    "PerformanceMetrics",
    "StrategyState",
    # Trade History
    "TradeRecord",
    "PositionRecord",
    "TradeHistoryManager",
    # WebSocket Events
    "WebSocketEventType",
    "StopTriggerReason",
    "OrderUpdateEvent",
    "PositionUpdateEvent",
    "TradeUpdateEvent",
    "AccountUpdateEvent",
]