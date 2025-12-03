"""
交易相关数据模型
"""
from typing import Optional, Dict, Any, List
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from .enums import TradingAction, OrderType, PositionSide, ExecutionStatus
from enum import Enum


class OrderStatus(str, Enum):
    """订单状态"""
    OPEN = "open"  # 未成交
    CLOSED = "closed"  # 已成交
    CANCELED = "canceled"  # 已取消
    EXPIRED = "expired"  # 已过期
    REJECTED = "rejected"  # 已拒绝
    PARTIALLY_FILLED = "partially_filled"  # 部分成交


class TradingIntent(BaseModel):
    """
    交易意图模型 - 从 LLM 解析出的结构化交易意图
    """
    action: TradingAction = Field(..., description="交易动作类型")
    symbol: str = Field(..., description="交易对，如 BTC/USDT")
    side: Optional[PositionSide] = Field(None, description="持仓方向（开仓时必需）")
    order_type: OrderType = Field(default=OrderType.MARKET, description="订单类型")

    # 数量和价格
    amount: Optional[Decimal] = Field(None, description="交易数量")
    price: Optional[Decimal] = Field(None, description="限价单价格")

    # 风险管理参数
    stop_loss: Optional[Decimal] = Field(None, description="止损价格")
    take_profit: Optional[Decimal] = Field(None, description="止盈价格")
    leverage: Optional[int] = Field(default=1, ge=1, le=125, description="杠杆倍数")

    # 其他参数
    position_id: Optional[str] = Field(None, description="持仓ID（平仓或修改时需要）")
    reduce_only: bool = Field(default=False, description="只减仓模式")

    # 元数据
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM 解析置信度")
    raw_params: Dict[str, Any] = Field(default_factory=dict, description="原始解析参数")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """验证交易对格式"""
        if '/' not in v:
            raise ValueError(f"交易对格式错误: {v}，应为 BASE/QUOTE 格式，如 BTC/USDT")
        return v.upper()

    @field_validator('amount', 'price', 'stop_loss', 'take_profit')
    @classmethod
    def validate_positive(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """验证金额必须为正数"""
        if v is not None and v <= 0:
            raise ValueError(f"数值必须大于 0，当前值: {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": "open_long",
                "symbol": "BTC/USDT",
                "side": "long",
                "order_type": "market",
                "amount": "0.1",
                "stop_loss": "60000",
                "take_profit": "70000",
                "leverage": 10
            }
        }
    }


class TakeProfitTarget(BaseModel):
    """
    止盈目标模型

    支持多重止盈策略，例如 TurboTrader 的四重止盈：
    - 第一目标 (1:1): 止盈 25%
    - 第二目标 (1:2): 止盈 35%
    - 第三目标 (1:3): 止盈 25%
    - 第四目标 (1:5+): 止盈 15%
    """
    price: Decimal = Field(..., description="止盈价格")
    percent: float = Field(..., ge=0.0, le=100.0, description="该目标平仓比例 (%)")
    order_id: Optional[str] = Field(None, description="对应的订单ID（已创建时）")
    triggered: bool = Field(default=False, description="是否已触发")
    new_stop_loss: Optional[Decimal] = Field(None, description="触发后的新止损价格")

    model_config = {
        "json_schema_extra": {
            "example": {
                "price": "70000",
                "percent": 25.0,
                "order_id": "123456",
                "triggered": False,
                "new_stop_loss": "65000"
            }
        }
    }


class MultiTargetSLTP(BaseModel):
    """
    多重止盈止损配置模型

    用于支持复杂的风控策略：
    - 单一止损（或移动止损）
    - 多重止盈目标（阶梯式止盈）

    示例：TurboTrader 四重止盈策略
    - 止损: 入场价 - 1.5%
    - 止盈1: 入场价 + 1.5% (平仓 25%，止损移至入场价)
    - 止盈2: 入场价 + 3.0% (平仓 35%，止损移至盈利 1.2%)
    - 止盈3: 入场价 + 4.5% (平仓 25%，动态跟踪止损)
    - 止盈4: 入场价 + 7.5% (平仓 15%，SAR 跟踪)
    """
    stop_loss: Optional[Decimal] = Field(None, description="止损价格")
    take_profits: List[TakeProfitTarget] = Field(default_factory=list, description="多重止盈目标列表")
    trailing_stop: Optional[float] = Field(None, description="移动止损百分比 (%)")
    trailing_stop_trigger: Optional[Decimal] = Field(None, description="移动止损触发价格")

    def get_active_take_profits(self) -> List[TakeProfitTarget]:
        """获取未触发的止盈目标"""
        return [tp for tp in self.take_profits if not tp.triggered]

    def get_total_tp_percent(self) -> float:
        """获取所有止盈目标的总百分比"""
        return sum(tp.percent for tp in self.take_profits)

    def validate_tp_percent(self) -> bool:
        """验证止盈百分比之和是否为100%"""
        total = self.get_total_tp_percent()
        return 99.0 <= total <= 101.0  # 允许小误差

    model_config = {
        "json_schema_extra": {
            "example": {
                "stop_loss": "63000",
                "take_profits": [
                    {"price": "66000", "percent": 25.0, "new_stop_loss": "65000"},
                    {"price": "68000", "percent": 35.0, "new_stop_loss": "66200"},
                    {"price": "70000", "percent": 25.0},
                    {"price": "75000", "percent": 15.0}
                ],
                "trailing_stop": 1.0
            }
        }
    }


class Position(BaseModel):
    """
    持仓信息模型
    """
    position_id: str = Field(..., description="持仓ID")
    symbol: str = Field(..., description="交易对")
    side: PositionSide = Field(..., description="持仓方向")
    amount: Decimal = Field(..., description="持仓数量")
    entry_price: Decimal = Field(..., description="开仓均价")
    mark_price: Optional[Decimal] = Field(None, description="标记价格")
    liquidation_price: Optional[Decimal] = Field(None, description="强平价格")

    # 盈亏信息
    unrealized_pnl: Optional[Decimal] = Field(None, description="未实现盈亏")
    realized_pnl: Optional[Decimal] = Field(None, description="已实现盈亏")
    pnl_percentage: Optional[float] = Field(None, description="盈亏百分比")

    # 风险管理 (基础单一止盈止损)
    stop_loss: Optional[Decimal] = Field(None, description="止损价格")
    take_profit: Optional[Decimal] = Field(None, description="止盈价格（单一）")
    leverage: int = Field(default=1, description="杠杆倍数")

    # 多重止盈止损配置
    multi_sltp: Optional[MultiTargetSLTP] = Field(None, description="多重止盈止损配置")

    # 时间信息
    opened_at: datetime = Field(..., description="开仓时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

    # 原始数据
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="交易所原始数据")

    def has_multi_tp(self) -> bool:
        """是否配置了多重止盈"""
        return self.multi_sltp is not None and len(self.multi_sltp.take_profits) > 0


class ExecutionResult(BaseModel):
    """
    交易执行结果模型
    """
    status: ExecutionStatus = Field(..., description="执行状态")
    action: TradingAction = Field(..., description="执行的动作")

    # 订单信息
    order_id: Optional[str] = Field(None, description="订单ID")
    symbol: str = Field(..., description="交易对")

    # 执行详情
    executed_amount: Optional[Decimal] = Field(None, description="已成交数量")
    executed_price: Optional[Decimal] = Field(None, description="成交均价")
    fee: Optional[Decimal] = Field(None, description="手续费")

    # 结果信息
    message: str = Field(default="", description="执行消息")
    error: Optional[str] = Field(None, description="错误信息")

    # 持仓信息（如果适用）
    position: Optional[Position] = Field(None, description="相关持仓信息")

    # 元数据
    timestamp: datetime = Field(default_factory=datetime.now, description="执行时间")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="交易所原始响应")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "action": "open_long",
                "order_id": "123456789",
                "symbol": "BTC/USDT",
                "executed_amount": "0.1",
                "executed_price": "65000",
                "fee": "6.5",
                "message": "订单已成功执行"
            }
        }
    }


class Balance(BaseModel):
    """
    账户余额模型
    """
    currency: str = Field(..., description="币种")
    total: Decimal = Field(..., description="总额")
    available: Decimal = Field(..., description="可用余额")
    frozen: Decimal = Field(default=Decimal("0"), description="冻结金额")

    # 额外信息
    equity: Optional[Decimal] = Field(None, description="权益")
    margin: Optional[Decimal] = Field(None, description="保证金")
    unrealized_pnl: Optional[Decimal] = Field(None, description="未实现盈亏")

    timestamp: datetime = Field(default_factory=datetime.now, description="查询时间")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="原始数据")


class Order(BaseModel):
    """
    订单信息模型
    """
    order_id: str = Field(..., description="订单ID")
    symbol: str = Field(..., description="交易对")
    order_type: OrderType = Field(..., description="订单类型")
    side: PositionSide = Field(..., description="订单方向")

    # 数量和价格
    amount: Decimal = Field(..., description="订单数量")
    price: Optional[Decimal] = Field(None, description="订单价格（限价单）")
    average_price: Optional[Decimal] = Field(None, description="成交均价")

    # 成交信息
    filled: Decimal = Field(default=Decimal("0"), description="已成交数量")
    remaining: Optional[Decimal] = Field(None, description="剩余数量")
    status: OrderStatus = Field(..., description="订单状态")

    # 费用
    fee: Optional[Decimal] = Field(None, description="手续费")
    fee_currency: Optional[str] = Field(None, description="手续费币种")

    # 其他参数
    reduce_only: bool = Field(default=False, description="只减仓模式")
    post_only: bool = Field(default=False, description="只做maker")

    # 时间信息
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

    # 原始数据
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="交易所原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "order_id": "123456789",
                "symbol": "BTC/USDT",
                "order_type": "limit",
                "side": "long",
                "amount": "0.1",
                "price": "65000",
                "filled": "0.05",
                "status": "partially_filled",
                "created_at": "2024-01-01T12:00:00"
            }
        }
    }


class Trade(BaseModel):
    """
    成交记录模型
    """
    trade_id: str = Field(..., description="成交ID")
    order_id: str = Field(..., description="关联的订单ID")
    symbol: str = Field(..., description="交易对")
    side: PositionSide = Field(..., description="成交方向")

    # 成交信息
    amount: Decimal = Field(..., description="成交数量")
    price: Decimal = Field(..., description="成交价格")

    # 开仓/平仓信息
    trade_type: Optional[str] = Field(None, description="交易类型: open(开仓), close(平仓), add(加仓), reduce(减仓)")
    closed_pnl: Optional[Decimal] = Field(None, description="平仓盈亏（仅平仓时有值）")
    start_position: Optional[Decimal] = Field(None, description="成交前持仓量")

    # 费用
    fee: Optional[Decimal] = Field(None, description="手续费")
    fee_currency: Optional[str] = Field(None, description="手续费币种")

    # 角色
    is_maker: Optional[bool] = Field(None, description="是否为maker")

    # 时间信息
    timestamp: datetime = Field(..., description="成交时间")

    # 原始数据
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="交易所原始数据")

    model_config = {
        "json_schema_extra": {
            "example": {
                "trade_id": "987654321",
                "order_id": "123456789",
                "symbol": "BTC/USDT",
                "side": "long",
                "amount": "0.05",
                "price": "65000",
                "fee": "3.25",
                "fee_currency": "USDT",
                "is_maker": False,
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    }