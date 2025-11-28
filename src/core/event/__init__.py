"""
事件系统

参考 ValueCell 的事件设计，提供：
- 类型安全的事件定义
- 事件发射器（发布/订阅模式）
- 事件工厂（统一创建事件）
"""
from .types import TradingEvent, TradingEventType, SystemEvent, SystemEventType
from .emitter import EventEmitter
from .factory import EventFactory

__all__ = [
    "TradingEvent",
    "TradingEventType",
    "SystemEvent",
    "SystemEventType",
    "EventEmitter",
    "EventFactory",
]
