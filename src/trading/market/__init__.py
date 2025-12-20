"""市场数据与特征计算模块。

提供:
- 市场数据源: 获取K线和行情快照
- 特征计算器: 计算技术指标（位于 features/ 子目录）
- 特征管道: 组合数据获取和特征计算
"""

from .data_interfaces import BaseMarketDataSource, MarketSnapshotType
from .data_source import SimpleMarketDataSource
from .feature_interfaces import (
    BaseFeaturesPipeline,
    CandleBasedFeatureComputer,
    MarketSnapshotFeatureComputer,
)
from .features import DefaultFeatureComputer, SimpleCandleFeatureComputer
from .market_snapshot import SimpleMarketSnapshotFeatureComputer
from .pipeline import DefaultFeaturesPipeline

__all__ = [
    # 数据源
    "BaseMarketDataSource",
    "MarketSnapshotType",
    "SimpleMarketDataSource",
    # 特征计算接口
    "BaseFeaturesPipeline",
    "CandleBasedFeatureComputer",
    "MarketSnapshotFeatureComputer",
    # 特征计算器实现
    "DefaultFeatureComputer",
    "SimpleCandleFeatureComputer",  # 向后兼容别名
    "SimpleMarketSnapshotFeatureComputer",
    # 管道
    "DefaultFeaturesPipeline",
]
