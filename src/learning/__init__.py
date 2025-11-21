"""
学习层 (Layer 3)

LangGraph驱动的学习和进化系统
- SqliteSaver Checkpointer（状态持久化、断点恢复）
- MemoryManager（SQLite 记忆管理 - 相似度匹配）
- 反思学习（Reflection）
- 策略优化（Optimization）

这是系统"智能"的核心
"""
from .learning_graph import LearningGraph, TradingState
from .memory_manager import MemoryManager, TradingCase

__all__ = [
    'LearningGraph',
    'TradingState',
    'MemoryManager',
    'TradingCase',
]
