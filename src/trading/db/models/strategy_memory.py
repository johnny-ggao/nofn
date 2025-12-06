"""策略记忆模型。

存储策略的短期记忆状态，用于策略重启恢复。
"""

from typing import Any, Dict

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.sql import func

from .base import Base


class StrategyMemory(Base):
    """策略记忆状态。

    存储策略的短期记忆快照，支持策略重启后恢复状态。
    每个策略只保留最新的记忆状态。
    """

    __tablename__ = "strategy_memories"

    id = Column(Integer, primary_key=True, index=True)

    # 关联策略
    strategy_id = Column(
        String(100),
        ForeignKey("strategies.strategy_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # 每个策略只有一条记忆记录
        index=True,
        comment="策略 ID",
    )

    # 记忆数据（JSON 格式）
    decisions = Column(
        JSON,
        nullable=False,
        default=list,
        comment="最近决策记录列表",
    )

    pending_signals = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="待观察信号",
    )

    # 元数据
    cycle_index = Column(
        Integer,
        nullable=False,
        default=0,
        comment="最后一次决策周期索引",
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

    def __repr__(self) -> str:
        return (
            f"<StrategyMemory(strategy_id='{self.strategy_id}', "
            f"decisions={len(self.decisions or [])}, cycle={self.cycle_index})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "decisions": self.decisions or [],
            "pending_signals": self.pending_signals or {},
            "cycle_index": self.cycle_index,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
