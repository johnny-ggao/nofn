"""反思模式模块。

提供交易决策的事后分析和经验总结能力：
- ReflectionInsight: 反思洞察数据模型
- ReflectionAnalyzer: 分析器，从历史中提取教训
- ReflectiveComposer: 带反思能力的决策器
"""

from .models import (
    ReflectionInsight,
    ReflectionTrigger,
    PerformanceAlert,
    TradingLesson,
)
from .analyzer import ReflectionAnalyzer


def __getattr__(name):
    """延迟加载 ReflectiveComposer 以避免循环导入。"""
    if name == "ReflectiveComposer":
        from .composer import ReflectiveComposer
        return ReflectiveComposer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ReflectionInsight",
    "ReflectionTrigger",
    "PerformanceAlert",
    "TradingLesson",
    "ReflectionAnalyzer",
    "ReflectiveComposer",
]
