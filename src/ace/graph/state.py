"""
ACE State Model for LangGraph

定义 ACE Agent 在 LangGraph 中的状态结构
"""

from typing import TypedDict, List, Optional, Any
from ..models import ExecutionTrace, Reflection, ContextEntry


class ACEState(TypedDict):
    """
    ACE Agent 的状态模型

    在 LangGraph 工作流中传递的完整状态
    """
    # ========== 输入参数 ==========
    symbols: List[str]              # 交易对列表
    iteration: int                  # 当前迭代次数

    # ========== Generator Phase 输出 ==========
    trace: Optional[ExecutionTrace]  # 完整的执行轨迹

    # ========== Reflector Phase 输出 ==========
    reflection: Optional[Reflection]  # 反思结果

    # ========== Curator Phase 输出 ==========
    updated_entries: List[ContextEntry]  # 更新/创建的知识条目

    # ========== 控制流程 ==========
    should_continue: bool           # 是否继续下一轮迭代
    max_iterations: Optional[int]   # 最大迭代次数（None = 无限）

    # ========== 错误处理 ==========
    errors: List[str]               # 错误列表
