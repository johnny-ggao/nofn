"""特征计算器注册表。

提供特征计算器的注册和获取功能，支持按名称动态选择计算器。

使用方法：
    # 获取计算器实例
    computer = get_feature_computer("default")

    # 注册自定义计算器
    register_feature_computer("my_strategy", MyFeatureComputer)
"""

from typing import Callable, Dict, Type

from .base import CandleBasedFeatureComputer

# 计算器类型别名
FeatureComputerFactory = Callable[[], CandleBasedFeatureComputer]

# 注册表：名称 -> 计算器工厂函数
_REGISTRY: Dict[str, FeatureComputerFactory] = {}


def register_feature_computer(
    name: str,
    computer_cls: Type[CandleBasedFeatureComputer],
    **default_kwargs,
) -> None:
    """注册特征计算器。

    Args:
        name: 计算器名称，用于配置中引用
        computer_cls: 计算器类
        **default_kwargs: 实例化时的默认参数
    """
    def factory() -> CandleBasedFeatureComputer:
        return computer_cls(**default_kwargs)

    _REGISTRY[name] = factory


def get_feature_computer(name: str) -> CandleBasedFeatureComputer:
    """获取特征计算器实例。

    Args:
        name: 计算器名称

    Returns:
        计算器实例

    Raises:
        ValueError: 未知的计算器名称
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        available = ", ".join(_REGISTRY.keys()) or "(none)"
        raise ValueError(
            f"Unknown feature computer: '{name}'. Available: {available}"
        )
    return factory()


def list_feature_computers() -> list[str]:
    """列出所有已注册的计算器名称。"""
    return list(_REGISTRY.keys())


# =============================================================================
# 默认注册
# =============================================================================

def _register_defaults() -> None:
    """注册内置的特征计算器。"""
    from .default import DefaultFeatureComputer
    from .trend_following import TrendFollowingFeatureComputer

    register_feature_computer("default", DefaultFeatureComputer)
    register_feature_computer("trend_following", TrendFollowingFeatureComputer)


# 模块加载时自动注册默认计算器
_register_defaults()
