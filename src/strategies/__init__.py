"""
策略模块

每个策略包含：
1. 策略配置（指标参数、时间框架权重等）
2. Prompt 模板
3. 指标计算逻辑

可用策略：
- MTF_Momentum (默认): 多时间框架动量策略，1H/15M/5M 三级确认
- TurboTrader: 激进交易策略，三重涡轮确认系统
"""
from .base import BaseStrategy, StrategyConfig, TimeframeConfig, IndicatorConfig
from .factory import StrategyFactory
from .mtf_momentum import MTFMomentumStrategy
from .turbo import TurboTraderStrategy

__all__ = [
    # 基类和配置
    'BaseStrategy',
    'StrategyConfig',
    'TimeframeConfig',
    'IndicatorConfig',
    # 工厂
    'StrategyFactory',
    # 策略实现
    'MTFMomentumStrategy',
    'TurboTraderStrategy',
]