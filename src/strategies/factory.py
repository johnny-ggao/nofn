"""
策略工厂

负责策略的注册、加载和管理
"""
from typing import Dict, Type, Optional
from pathlib import Path

from .base import BaseStrategy, StrategyConfig


class StrategyFactory:
    """
    策略工厂

    支持：
    1. 通过名称加载预定义策略
    2. 通过配置文件加载自定义策略
    3. 策略注册和发现
    """

    # 已注册的策略类
    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        注册策略

        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        cls._registry[name.lower()] = strategy_class

    @classmethod
    def get(cls, name: str, config: Optional[StrategyConfig] = None) -> BaseStrategy:
        """
        获取策略实例

        Args:
            name: 策略名称
            config: 可选的策略配置，如果不提供则使用策略默认配置

        Returns:
            策略实例

        Raises:
            ValueError: 策略未注册
        """
        name_lower = name.lower()

        if name_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"策略 '{name}' 未注册。可用策略: {available}")

        strategy_class = cls._registry[name_lower]

        if config:
            return strategy_class(config)
        else:
            return strategy_class()

    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        """
        列出所有已注册的策略

        Returns:
            策略名称到描述的映射
        """
        result = {}
        for name, strategy_class in cls._registry.items():
            # 创建临时实例获取描述
            try:
                instance = strategy_class()
                result[name] = instance.config.description
            except Exception:
                result[name] = "（无法获取描述）"
        return result

    @classmethod
    def load_from_config(cls, config: StrategyConfig) -> BaseStrategy:
        """
        从配置加载策略

        根据配置中的 name 查找对应的策略类

        Args:
            config: 策略配置

        Returns:
            策略实例
        """
        return cls.get(config.name, config)

    @classmethod
    def create_default(cls) -> BaseStrategy:
        """
        创建默认策略

        Returns:
            MTF_Momentum 策略实例
        """
        return cls.get("mtf_momentum")


# 自动注册内置策略
def _register_builtin_strategies():
    """注册内置策略"""
    from .mtf_momentum import MTFMomentumStrategy
    from .turbo import TurboTraderStrategy

    # MTF_Momentum 策略（默认策略）
    StrategyFactory.register("mtf_momentum", MTFMomentumStrategy)
    StrategyFactory.register("mtf", MTFMomentumStrategy)  # 别名

    # TurboTrader 策略（激进策略）
    StrategyFactory.register("turbo", TurboTraderStrategy)
    StrategyFactory.register("turbotrader", TurboTraderStrategy)  # 别名


# 模块加载时自动注册
_register_builtin_strategies()