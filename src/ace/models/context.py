"""
Context Entry - ACE 知识库的基本单元
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import uuid


class EntryType(str, Enum):
    """知识条目类型"""
    STRATEGY = "strategy"          # 交易策略
    PATTERN = "pattern"            # 市场模式
    RISK_RULE = "risk_rule"        # 风控规则
    ERROR_PATTERN = "error_pattern"  # 错误模式（需要避免）
    MARKET_INSIGHT = "market_insight"  # 市场洞察


@dataclass
class ContextEntry:
    """
    ACE 框架的核心数据结构：知识条目

    每个条目是一个可复用的、结构化的知识片段，
    带有元数据来跟踪其有效性。
    """

    # 基本信息
    entry_id: str = field(default_factory=lambda: f"entry_{uuid.uuid4().hex[:12]}")
    entry_type: EntryType = EntryType.STRATEGY
    content: str = ""  # 条目的实际内容（策略描述、规则等）

    # ACE 元数据（核心）
    helpful_count: int = 0      # 被证明有用的次数
    harmful_count: int = 0      # 被证明有害的次数
    neutral_count: int = 0      # 无明显影响的次数

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    # 关联信息
    source_trace_ids: List[str] = field(default_factory=list)  # 来源的执行轨迹
    tags: List[str] = field(default_factory=list)  # 标签（如：trend_following, mean_reversion）

    # 向量表示（用于相似度计算）
    embedding: Optional[List[float]] = None

    def calculate_confidence(self) -> float:
        """
        计算条目的置信度

        基于 helpful 和 harmful 计数器，使用贝叶斯平滑
        """
        total = self.helpful_count + self.harmful_count + self.neutral_count

        if total == 0:
            return 0.5  # 初始中性置信度

        # 贝叶斯平滑：加入先验（假设有 2 个中性样本）
        prior_samples = 2
        prior_confidence = 0.5

        adjusted_helpful = self.helpful_count + (prior_samples * prior_confidence)
        adjusted_total = total + prior_samples

        return adjusted_helpful / adjusted_total

    @property
    def confidence(self) -> float:
        """置信度属性（只读）"""
        return self.calculate_confidence()

    def is_harmful(self) -> bool:
        """判断是否为有害模式"""
        return self.harmful_count > self.helpful_count and self.confidence < 0.4

    def is_helpful(self) -> bool:
        """判断是否为有用策略"""
        return self.helpful_count > self.harmful_count and self.confidence > 0.6

    def mark_helpful(self):
        """标记为有用"""
        self.helpful_count += 1
        self.last_used = datetime.now()
        self.last_updated = datetime.now()

    def mark_harmful(self):
        """标记为有害"""
        self.harmful_count += 1
        self.last_used = datetime.now()
        self.last_updated = datetime.now()

    def mark_neutral(self):
        """标记为中性"""
        self.neutral_count += 1
        self.last_used = datetime.now()
        self.last_updated = datetime.now()

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            'entry_id': self.entry_id,
            'entry_type': self.entry_type.value,
            'content': self.content,
            'helpful_count': self.helpful_count,
            'harmful_count': self.harmful_count,
            'neutral_count': self.neutral_count,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'last_updated': self.last_updated.isoformat(),
            'source_trace_ids': self.source_trace_ids,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ContextEntry':
        """从字典创建（用于反序列化）"""
        return cls(
            entry_id=data['entry_id'],
            entry_type=EntryType(data['entry_type']),
            content=data['content'],
            helpful_count=data.get('helpful_count', 0),
            harmful_count=data.get('harmful_count', 0),
            neutral_count=data.get('neutral_count', 0),
            created_at=datetime.fromisoformat(data['created_at']),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            last_updated=datetime.fromisoformat(data.get('last_updated', data['created_at'])),
            source_trace_ids=data.get('source_trace_ids', []),
            tags=data.get('tags', []),
        )

    def __repr__(self) -> str:
        return (
            f"ContextEntry(id={self.entry_id[:8]}..., "
            f"type={self.entry_type.value}, "
            f"confidence={self.confidence:.2f}, "
            f"h={self.helpful_count}, n={self.harmful_count})"
        )
