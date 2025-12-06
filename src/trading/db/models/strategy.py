"""策略模型。

存储策略的基本信息和配置。
"""

from typing import Any, Dict

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from .base import Base


class Strategy(Base):
    """策略记录。"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)

    strategy_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="策略运行时 ID",
    )
    name = Column(String(200), nullable=True, comment="策略名称")
    description = Column(Text, nullable=True, comment="策略描述")

    status = Column(
        String(50), nullable=False, default="running", comment="策略状态"
    )

    # 配置和元数据
    config = Column(JSON, nullable=True, comment="UserRequest 配置")
    metadata_ = Column(
        "metadata", JSON, nullable=True, comment="额外元数据"
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
        return f"<Strategy(id={self.id}, strategy_id='{self.strategy_id}', status='{self.status}')>"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "config": self.config,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
