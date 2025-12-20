"""K 线特征计算器模块。

按策略/场景组织的特征计算器集合。每个计算器包含特定策略所需的指标组合。

可用的计算器：
- default: DefaultFeatureComputer - 通用技术指标组合（EMA 12/26/50, MACD, RSI, BB, ATR, ADX）
- trend_following: TrendFollowingFeatureComputer - 趋势跟踪指标（EMA 20/50/200, RSI, ATR）

扩展指南：
1. 创建新文件（如 scalping.py）
2. 继承 CandleBasedFeatureComputer 基类
3. 实现 compute_features() 方法
4. 在 registry.py 中注册
5. 在此处导出（可选）

使用注册表：
    from .registry import get_feature_computer, register_feature_computer

    # 获取计算器
    computer = get_feature_computer("trend_following")

    # 注册自定义计算器
    register_feature_computer("my_strategy", MyFeatureComputer)
"""

from .base import CandleBasedFeatureComputer
from .default import DefaultFeatureComputer, SimpleCandleFeatureComputer
from .trend_following import TrendFollowingFeatureComputer
from .registry import (
    get_feature_computer,
    list_feature_computers,
    register_feature_computer,
)

__all__ = [
    # 基类
    "CandleBasedFeatureComputer",
    # 实现
    "DefaultFeatureComputer",
    "SimpleCandleFeatureComputer",  # 向后兼容别名
    "TrendFollowingFeatureComputer",
    # 注册表
    "get_feature_computer",
    "list_feature_computers",
    "register_feature_computer",
]
