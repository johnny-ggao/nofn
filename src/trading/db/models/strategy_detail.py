"""策略交易详情模型。

存储每笔交易的详细信息。
"""

from typing import Any, Dict

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class StrategyDetail(Base):
    """策略交易详情记录。"""

    __tablename__ = "strategy_details"

    id = Column(Integer, primary_key=True, index=True)

    # 关联策略
    strategy_id = Column(
        String(100),
        ForeignKey("strategies.strategy_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="策略 ID",
    )

    # 标识符
    compose_id = Column(
        String(200), nullable=True, index=True, comment="决策周期 ID"
    )
    trade_id = Column(String(200), nullable=False, comment="交易 ID")
    instruction_id = Column(
        String(200), nullable=True, index=True, comment="指令 ID"
    )

    # 交易信息
    symbol = Column(String(50), nullable=False, index=True, comment="交易对")
    type = Column(String(20), nullable=False, comment="持仓类型: LONG/SHORT")
    side = Column(String(20), nullable=False, comment="交易方向: BUY/SELL")
    leverage = Column(Numeric(10, 4), nullable=True, comment="杠杆倍数")
    quantity = Column(Numeric(20, 8), nullable=False, comment="交易数量")

    # 价格和盈亏
    entry_price = Column(Numeric(20, 8), nullable=True, comment="入场价格")
    exit_price = Column(Numeric(20, 8), nullable=True, comment="出场价格")
    avg_exec_price = Column(Numeric(20, 8), nullable=True, comment="平均成交价")
    unrealized_pnl = Column(Numeric(20, 8), nullable=True, comment="未实现盈亏")
    realized_pnl = Column(Numeric(20, 8), nullable=True, comment="已实现盈亏")
    realized_pnl_pct = Column(Numeric(10, 6), nullable=True, comment="已实现盈亏百分比")
    notional_entry = Column(Numeric(20, 8), nullable=True, comment="入场名义价值")
    notional_exit = Column(Numeric(20, 8), nullable=True, comment="出场名义价值")
    fee_cost = Column(Numeric(20, 8), nullable=True, comment="手续费")

    # 时间
    holding_ms = Column(Integer, nullable=True, comment="持仓时长(毫秒)")
    entry_time = Column(DateTime(timezone=True), nullable=True, comment="入场时间")
    exit_time = Column(DateTime(timezone=True), nullable=True, comment="出场时间")

    # 备注
    note = Column(Text, nullable=True, comment="备注")

    # 时间戳
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("strategy_id", "trade_id", name="uq_strategy_trade_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<StrategyDetail(id={self.id}, strategy_id='{self.strategy_id}', "
            f"trade_id='{self.trade_id}', symbol='{self.symbol}')>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "trade_id": self.trade_id,
            "compose_id": self.compose_id,
            "instruction_id": self.instruction_id,
            "symbol": self.symbol,
            "type": self.type,
            "side": self.side,
            "leverage": float(self.leverage) if self.leverage else None,
            "quantity": float(self.quantity) if self.quantity else None,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "avg_exec_price": float(self.avg_exec_price) if self.avg_exec_price else None,
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
            "realized_pnl": float(self.realized_pnl) if self.realized_pnl else None,
            "realized_pnl_pct": float(self.realized_pnl_pct) if self.realized_pnl_pct else None,
            "notional_entry": float(self.notional_entry) if self.notional_entry else None,
            "notional_exit": float(self.notional_exit) if self.notional_exit else None,
            "fee_cost": float(self.fee_cost) if self.fee_cost else None,
            "holding_ms": self.holding_ms,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
