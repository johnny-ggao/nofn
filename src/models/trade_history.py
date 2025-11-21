"""
交易历史数据模型

用于记录和管理交易记录、持仓历史
"""
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import json


@dataclass
class TradeRecord:
    """单笔交易记录"""
    # 基本信息
    trade_id: str
    timestamp: datetime
    symbol: str

    # 交易类型
    side: str  # "long" or "short"
    action: str  # "open", "close", "stop_loss", "take_profit", "add", "reduce"

    # 交易详情
    price: Decimal
    amount: Decimal
    leverage: Optional[int] = None

    # 费用和结果
    fee: Decimal = Decimal('0')
    realized_pnl: Optional[Decimal] = None  # 仅对平仓交易有效

    # 关联信息
    position_id: Optional[str] = None
    order_id: Optional[str] = None  # 交易所订单ID

    # 备注
    note: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'action': self.action,
            'price': float(self.price),
            'amount': float(self.amount),
            'leverage': self.leverage,
            'fee': float(self.fee),
            'realized_pnl': float(self.realized_pnl) if self.realized_pnl else None,
            'position_id': self.position_id,
            'order_id': self.order_id,
            'note': self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TradeRecord':
        """从字典创建"""
        return cls(
            trade_id=data['trade_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            side=data['side'],
            action=data['action'],
            price=Decimal(str(data['price'])),
            amount=Decimal(str(data['amount'])),
            leverage=data.get('leverage'),
            fee=Decimal(str(data.get('fee', 0))),
            realized_pnl=Decimal(str(data['realized_pnl'])) if data.get('realized_pnl') else None,
            position_id=data.get('position_id'),
            order_id=data.get('order_id'),
            note=data.get('note'),
        )


@dataclass
class PositionRecord:
    """持仓记录（包含当前和历史）"""
    # 基本信息
    position_id: str
    symbol: str
    side: str  # "long" or "short"

    # 开仓信息
    entry_time: datetime
    entry_price: Decimal
    amount: Decimal
    leverage: int

    # 风控参数
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # 状态
    status: str = "open"  # "open", "closed", "stopped", "taken"

    # 平仓信息
    close_time: Optional[datetime] = None
    close_price: Optional[Decimal] = None
    close_reason: Optional[str] = None  # "manual", "stop_loss", "take_profit"

    # 盈亏
    realized_pnl: Optional[Decimal] = None
    realized_pnl_percent: Optional[float] = None

    # 关联的交易记录
    trade_ids: List[str] = field(default_factory=list)

    # 统计信息
    max_unrealized_pnl: Decimal = Decimal('0')  # 最大浮盈
    min_unrealized_pnl: Decimal = Decimal('0')  # 最大浮亏

    # 备注
    note: Optional[str] = None

    def is_open(self) -> bool:
        """是否为开仓状态"""
        return self.status == "open"

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """计算当前盈亏"""
        if self.side == "long":
            pnl = (current_price - self.entry_price) * self.amount
        else:  # short
            pnl = (self.entry_price - current_price) * self.amount
        return pnl

    def calculate_pnl_percent(self, current_price: Decimal) -> float:
        """计算盈亏百分比"""
        pnl = self.calculate_pnl(current_price)
        cost = self.entry_price * self.amount / self.leverage
        return float(pnl / cost * 100) if cost > 0 else 0.0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': float(self.entry_price),
            'amount': float(self.amount),
            'leverage': self.leverage,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'status': self.status,
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'close_price': float(self.close_price) if self.close_price else None,
            'close_reason': self.close_reason,
            'realized_pnl': float(self.realized_pnl) if self.realized_pnl else None,
            'realized_pnl_percent': self.realized_pnl_percent,
            'trade_ids': self.trade_ids,
            'max_unrealized_pnl': float(self.max_unrealized_pnl),
            'min_unrealized_pnl': float(self.min_unrealized_pnl),
            'note': self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PositionRecord':
        """从字典创建"""
        return cls(
            position_id=data['position_id'],
            symbol=data['symbol'],
            side=data['side'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            entry_price=Decimal(str(data['entry_price'])),
            amount=Decimal(str(data['amount'])),
            leverage=data['leverage'],
            stop_loss=Decimal(str(data['stop_loss'])) if data.get('stop_loss') else None,
            take_profit=Decimal(str(data['take_profit'])) if data.get('take_profit') else None,
            status=data.get('status', 'open'),
            close_time=datetime.fromisoformat(data['close_time']) if data.get('close_time') else None,
            close_price=Decimal(str(data['close_price'])) if data.get('close_price') else None,
            close_reason=data.get('close_reason'),
            realized_pnl=Decimal(str(data['realized_pnl'])) if data.get('realized_pnl') else None,
            realized_pnl_percent=data.get('realized_pnl_percent'),
            trade_ids=data.get('trade_ids', []),
            max_unrealized_pnl=Decimal(str(data.get('max_unrealized_pnl', 0))),
            min_unrealized_pnl=Decimal(str(data.get('min_unrealized_pnl', 0))),
            note=data.get('note'),
        )
