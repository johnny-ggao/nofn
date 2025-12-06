"""策略持仓快照模型。

存储策略在某一时刻的持仓状态。
"""

from typing import Any, Dict

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .base import Base


class StrategyHolding(Base):
    """策略持仓快照记录。"""

    __tablename__ = "strategy_holdings"

    id = Column(Integer, primary_key=True, index=True)

    # 关联策略
    strategy_id = Column(
        String(100),
        ForeignKey("strategies.strategy_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="策略 ID",
    )

    # 持仓信息
    symbol = Column(String(50), nullable=False, index=True, comment="交易对")
    type = Column(String(20), nullable=False, comment="持仓类型: LONG/SHORT")
    leverage = Column(Numeric(10, 4), nullable=True, comment="杠杆倍数")
    entry_price = Column(Numeric(20, 8), nullable=True, comment="平均入场价")
    quantity = Column(Numeric(20, 8), nullable=False, comment="持仓数量")
    unrealized_pnl = Column(Numeric(20, 8), nullable=True, comment="未实现盈亏")
    unrealized_pnl_pct = Column(Numeric(10, 4), nullable=True, comment="未实现盈亏百分比")

    # 快照时间
    snapshot_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="快照时间",
    )

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
        UniqueConstraint(
            "strategy_id", "symbol", "snapshot_ts", name="uq_strategy_holding_snapshot"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<StrategyHolding(id={self.id}, strategy_id='{self.strategy_id}', "
            f"symbol='{self.symbol}', quantity={self.quantity})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "type": self.type,
            "leverage": float(self.leverage) if self.leverage else None,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "quantity": float(self.quantity) if self.quantity else None,
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
            "unrealized_pnl_pct": float(self.unrealized_pnl_pct) if self.unrealized_pnl_pct else None,
            "snapshot_ts": self.snapshot_ts.isoformat() if self.snapshot_ts else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
