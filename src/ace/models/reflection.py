"""
Reflection - Reflector 的分析结果
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid


class FailureType(str, Enum):
    """失败类型分类"""
    CONCEPTUAL = "conceptual"      # 概念性错误（理解市场错误）
    COMPUTATIONAL = "computational"  # 计算错误（指标计算、价格估算等）
    STRATEGIC = "strategic"        # 策略性错误（时机选择、仓位管理等）
    EXECUTION = "execution"        # 执行错误（API 失败、网络问题等）
    NONE = "none"                  # 无明显错误


@dataclass
class StrategyEvaluation:
    """
    单个策略条目的评价

    Reflector 分析使用了哪些策略，并评估其有效性
    """
    entry_id: str
    is_helpful: bool  # True=有用, False=有害, None=中性
    reason: str = ""  # 为什么有用/有害

    def to_dict(self) -> dict:
        return {
            'entry_id': self.entry_id,
            'is_helpful': self.is_helpful,
            'reason': self.reason,
        }


@dataclass
class Reflection:
    """
    反思结果（Reflector 的输出）

    分析 ExecutionTrace，诊断问题，提取洞察
    """

    # 唯一标识
    reflection_id: str = field(default_factory=lambda: f"refl_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.now)

    # 关联的执行轨迹
    trace_id: str = ""

    # 结果分析
    is_successful: bool = False  # 整体是否成功
    failure_type: FailureType = FailureType.NONE

    # 策略评价
    strategy_evaluations: List[StrategyEvaluation] = field(default_factory=list)

    # 洞察提取
    key_insights: List[str] = field(default_factory=list)  # 新发现的规律
    error_patterns: List[str] = field(default_factory=list)  # 需要避免的错误
    improvement_suggestions: List[str] = field(default_factory=list)  # 改进建议

    # 完整的反思文本
    reflection_text: str = ""

    # 市场状态特征（用于后续检索相似情况）
    market_conditions: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化"""
        return {
            'reflection_id': self.reflection_id,
            'timestamp': self.timestamp.isoformat(),
            'trace_id': self.trace_id,
            'is_successful': self.is_successful,
            'failure_type': self.failure_type.value,
            'strategy_evaluations': [e.to_dict() for e in self.strategy_evaluations],
            'key_insights': self.key_insights,
            'error_patterns': self.error_patterns,
            'improvement_suggestions': self.improvement_suggestions,
            'reflection_text': self.reflection_text,
            'market_conditions': self.market_conditions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Reflection':
        """反序列化"""
        return cls(
            reflection_id=data['reflection_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            trace_id=data['trace_id'],
            is_successful=data['is_successful'],
            failure_type=FailureType(data['failure_type']),
            strategy_evaluations=[
                StrategyEvaluation(**e) for e in data.get('strategy_evaluations', [])
            ],
            key_insights=data.get('key_insights', []),
            error_patterns=data.get('error_patterns', []),
            improvement_suggestions=data.get('improvement_suggestions', []),
            reflection_text=data.get('reflection_text', ''),
            market_conditions=data.get('market_conditions', {}),
        )

    def get_helpful_entries(self) -> List[str]:
        """获取有用的策略ID列表"""
        return [
            eval.entry_id for eval in self.strategy_evaluations
            if eval.is_helpful
        ]

    def get_harmful_entries(self) -> List[str]:
        """获取有害的策略ID列表"""
        return [
            eval.entry_id for eval in self.strategy_evaluations
            if not eval.is_helpful
        ]

    def __repr__(self) -> str:
        status = "✅" if self.is_successful else "❌"
        return (
            f"Reflection({self.reflection_id[:8]}... {status}, "
            f"insights={len(self.key_insights)}, errors={len(self.error_patterns)})"
        )
