"""
WebSocket 事件模型

用于处理交易所推送的实时事件（订单成交、仓位变化等）
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum

from .enums import PositionSide
from .trading import OrderStatus


class WebSocketEventType(Enum):
    """WebSocket 事件类型"""
    ORDER_UPDATE = "order_update"          # 订单更新
    POSITION_UPDATE = "position_update"    # 仓位更新
    ACCOUNT_UPDATE = "account_update"      # 账户更新
    TRADE_UPDATE = "trade_update"          # 成交更新


class StopTriggerReason(Enum):
    """止损/止盈触发原因"""
    STOP_LOSS = "stop_loss"       # 止损触发
    TAKE_PROFIT = "take_profit"   # 止盈触发
    LIQUIDATION = "liquidation"   # 强平
    MANUAL = "manual"             # 手动平仓
    UNKNOWN = "unknown"           # 未知原因


@dataclass
class OrderUpdateEvent:
    """
    订单更新事件

    当订单状态变化时触发（如新订单、部分成交、完全成交、取消等）
    """
    event_type: WebSocketEventType = field(default=WebSocketEventType.ORDER_UPDATE)
    timestamp: datetime = field(default_factory=datetime.now)

    # 订单基本信息
    order_id: str = ""
    symbol: str = ""
    side: Optional[PositionSide] = None
    order_type: str = ""  # limit, market, stop_loss, take_profit
    status: Optional[OrderStatus] = None

    # 价格和数量
    price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    filled: Optional[Decimal] = None
    remaining: Optional[Decimal] = None

    # 止损/止盈相关
    stop_price: Optional[Decimal] = None
    is_stop_loss: bool = False
    is_take_profit: bool = False
    trigger_reason: Optional[StopTriggerReason] = None

    # 原始数据
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def is_stop_triggered(self) -> bool:
        """判断是否是止损/止盈触发"""
        return self.is_stop_loss or self.is_take_profit

    def is_filled(self) -> bool:
        """判断订单是否完全成交"""
        return self.status == OrderStatus.CLOSED


@dataclass
class PositionUpdateEvent:
    """
    仓位更新事件

    当持仓发生变化时触发（开仓、加仓、减仓、平仓）
    """
    event_type: WebSocketEventType = field(default=WebSocketEventType.POSITION_UPDATE)
    timestamp: datetime = field(default_factory=datetime.now)

    # 仓位基本信息
    symbol: str = ""
    side: Optional[PositionSide] = None

    # 仓位变化
    position_amount: Optional[Decimal] = None  # 当前持仓量
    previous_amount: Optional[Decimal] = None  # 之前持仓量
    amount_change: Optional[Decimal] = None    # 持仓变化量

    # 价格信息
    entry_price: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None

    # 止损/止盈
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # 平仓原因（如果是平仓）
    is_closed: bool = False
    close_reason: Optional[StopTriggerReason] = None

    # 原始数据
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def is_stop_triggered(self) -> bool:
        """判断是否因止损/止盈触发平仓"""
        return (
            self.is_closed and
            self.close_reason in [StopTriggerReason.STOP_LOSS, StopTriggerReason.TAKE_PROFIT]
        )

    def get_close_reason_text(self) -> str:
        """获取平仓原因的文本描述"""
        if not self.close_reason:
            return "未知"

        reason_map = {
            StopTriggerReason.STOP_LOSS: "止损触发",
            StopTriggerReason.TAKE_PROFIT: "止盈触发",
            StopTriggerReason.LIQUIDATION: "强制平仓",
            StopTriggerReason.MANUAL: "手动平仓",
            StopTriggerReason.UNKNOWN: "未知原因",
        }
        return reason_map.get(self.close_reason, "未知")


@dataclass
class TradeUpdateEvent:
    """
    成交更新事件

    当有新的成交时触发
    """
    event_type: WebSocketEventType = field(default=WebSocketEventType.TRADE_UPDATE)
    timestamp: datetime = field(default_factory=datetime.now)

    # 成交信息
    trade_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: Optional[PositionSide] = None

    # 价格和数量
    price: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    fee: Optional[Decimal] = None
    fee_currency: str = ""

    # 是否是平仓成交
    is_closing: bool = False
    realized_pnl: Optional[Decimal] = None

    # 原始数据
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountUpdateEvent:
    """
    账户更新事件

    当账户余额变化时触发
    """
    event_type: WebSocketEventType = field(default=WebSocketEventType.ACCOUNT_UPDATE)
    timestamp: datetime = field(default_factory=datetime.now)

    # 余额信息
    total_balance: Optional[Decimal] = None
    available_balance: Optional[Decimal] = None
    frozen_balance: Optional[Decimal] = None

    # 权益变化
    equity_change: Optional[Decimal] = None

    # 原始数据
    raw_data: Dict[str, Any] = field(default_factory=dict)