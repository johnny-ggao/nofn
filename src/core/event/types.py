"""
事件类型定义

定义交易系统中所有事件类型，参考 ValueCell 的事件分类设计
"""
from enum import Enum
from typing import Optional, Dict, Any, Union
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field
import uuid


class TradingEventType(str, Enum):
    """交易事件类型"""
    # 订单生命周期
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"

    # 持仓变化
    POSITION_OPENED = "position_opened"
    POSITION_INCREASED = "position_increased"
    POSITION_DECREASED = "position_decreased"
    POSITION_CLOSED = "position_closed"
    POSITION_LIQUIDATED = "position_liquidated"

    # 止盈止损
    STOP_LOSS_SET = "stop_loss_set"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_SET = "take_profit_set"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"

    # 风险事件
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    MARGIN_WARNING = "margin_warning"
    LEVERAGE_CHANGED = "leverage_changed"


class SystemEventType(str, Enum):
    """系统事件类型"""
    # 系统状态
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"

    # 连接状态
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    EXCHANGE_ERROR = "exchange_error"

    # 策略状态
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_RESUMED = "strategy_resumed"
    STRATEGY_STOPPED = "strategy_stopped"

    # 决策事件
    DECISION_MADE = "decision_made"
    DECISION_EXECUTED = "decision_executed"
    DECISION_FAILED = "decision_failed"

    # 学习事件
    REFLECTION_COMPLETED = "reflection_completed"
    MEMORY_UPDATED = "memory_updated"


class BaseEvent(BaseModel):
    """事件基类"""
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = False


class TradingEvent(BaseEvent):
    """交易事件"""
    event_type: TradingEventType
    symbol: str
    order_id: Optional[str] = None
    position_id: Optional[str] = None

    # 订单相关
    side: Optional[str] = None  # "long" / "short"
    amount: Optional[Decimal] = None
    price: Optional[Decimal] = None
    filled_amount: Optional[Decimal] = None
    average_price: Optional[Decimal] = None

    # 持仓相关
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None

    # 止盈止损
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # 风险相关
    leverage: Optional[int] = None
    margin_ratio: Optional[float] = None

    # 错误信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    def __str__(self) -> str:
        """格式化输出"""
        parts = [f"[{self.event_type.value}]", f"symbol={self.symbol}"]
        if self.order_id:
            parts.append(f"order={self.order_id[:8]}")
        if self.amount:
            parts.append(f"amount={self.amount}")
        if self.price:
            parts.append(f"price={self.price}")
        if self.realized_pnl:
            parts.append(f"pnl={self.realized_pnl}")
        return " ".join(parts)


class SystemEvent(BaseEvent):
    """系统事件"""
    event_type: SystemEventType
    component: str = "system"  # 产生事件的组件
    message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    # 错误信息
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def __str__(self) -> str:
        """格式化输出"""
        parts = [f"[{self.event_type.value}]", f"component={self.component}"]
        if self.message:
            parts.append(f"msg={self.message}")
        if self.error_message:
            parts.append(f"error={self.error_message}")
        return " ".join(parts)


# 联合类型，方便类型检查
AnyEvent = Union[TradingEvent, SystemEvent]
AnyEventType = Union[TradingEventType, SystemEventType]
