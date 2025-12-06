"""市场数据与特征计算模块。

提供:
- 市场数据源: 获取K线和行情快照
- 特征计算器: 计算技术指标
- 特征管道: 组合数据获取和特征计算
"""

from .data_interfaces import BaseMarketDataSource, MarketSnapshotType
from .data_source import SimpleMarketDataSource
from .feature_interfaces import (
    BaseFeaturesPipeline,
    CandleBasedFeatureComputer,
    MarketSnapshotFeatureComputer,
)
from .candle import SimpleCandleFeatureComputer
from .market_snapshot import SimpleMarketSnapshotFeatureComputer
from .pipeline import DefaultFeaturesPipeline

__all__ = [
    # 数据源
    "BaseMarketDataSource",
    "MarketSnapshotType",
    "SimpleMarketDataSource",
    # 特征计算
    "BaseFeaturesPipeline",
    "CandleBasedFeatureComputer",
    "MarketSnapshotFeatureComputer",
    "SimpleCandleFeatureComputer",
    "SimpleMarketSnapshotFeatureComputer",
    # 管道
    "DefaultFeaturesPipeline",
]
