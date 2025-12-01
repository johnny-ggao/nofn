"""
LangGraph 状态定义

使用 TypedDict 定义工作流状态，这是 LangGraph 的推荐方式
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime
from operator import add

from ..engine.market_snapshot import MarketSnapshot


class TradingState(TypedDict, total=False):
    """
    交易工作流状态

    LangGraph 会在节点之间传递这个状态，每个节点可以读取和更新状态
    """
    # 输入
    symbols: List[str]
    iteration: int
    timestamp: datetime

    # 市场数据
    market_snapshot: Optional[MarketSnapshot]

    # 最近交易记录
    recent_trades: List[Dict[str, Any]]

    # 记忆上下文
    memory_context: Optional[str]
    similar_cases: List[Dict[str, Any]]

    # 决策
    decision: Optional[Dict[str, Any]]
    decision_raw_response: Optional[str]  # LLM 原始响应

    # 执行
    execution_results: List[Dict[str, Any]]

    # 评估
    evaluation: Optional[Dict[str, Any]]
    evaluation_raw_response: Optional[str]  # LLM 原始响应

    # 学习
    lessons_learned: Annotated[List[str], add]  # 使用 add 操作符累积经验
    quality_score: Optional[int]

    # 元数据
    errors: Annotated[List[str], add]  # 累积错误信息
    warnings: Annotated[List[str], add]  # 累积警告信息

    # 控制标志
    should_execute: bool  # 是否应该执行交易
    should_analyze: bool  # 是否应该运行深度分析
    human_approved: bool  # 人工是否批准


# 简化的状态（用于某些节点）
class DecisionState(TypedDict, total=False):
    """决策节点的输入状态"""
    market_snapshot: MarketSnapshot
    memory_context: str
    similar_cases: List[Dict]


class ExecutionState(TypedDict, total=False):
    """执行节点的输入状态"""
    decision: Dict[str, Any]
    market_snapshot: MarketSnapshot


class EvaluationState(TypedDict, total=False):
    """评估节点的输入状态"""
    decision: Dict[str, Any]
    execution_results: List[Dict]
    market_snapshot: MarketSnapshot