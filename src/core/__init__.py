"""
NoFn Core Module

核心模块包含：
- store: 存储抽象层（支持多种后端）
- event: 事件系统（交易事件、系统事件）
- types: 核心类型定义
"""
from .store import (
    BaseStore,
    OrderStore,
    TradeStore,
    PositionStore,
    InMemoryOrderStore,
    InMemoryTradeStore,
    InMemoryPositionStore,
    SQLiteOrderStore,
    SQLiteTradeStore,
    SQLitePositionStore,
)
from .event import (
    TradingEvent,
    TradingEventType,
    EventEmitter,
    EventFactory,
)
from .types import (
    OrderStatus,
    PositionSide,
    TradingAction,
)

__all__ = [
    # Store
    "BaseStore",
    "OrderStore",
    "TradeStore",
    "PositionStore",
    "InMemoryOrderStore",
    "InMemoryTradeStore",
    "InMemoryPositionStore",
    "SQLiteOrderStore",
    "SQLiteTradeStore",
    "SQLitePositionStore",
    # Event
    "TradingEvent",
    "TradingEventType",
    "EventEmitter",
    "EventFactory",
    # Types
    "OrderStatus",
    "PositionSide",
    "TradingAction",
]
