"""
执行层 (Layer 1)

确定性交易引擎 - 高效执行确定性任务
- 批量数据获取
- 指标批量计算
- 订单执行

不涉及LLM推理，专注于速度和可靠性
"""
from .trading_engine import TradingEngine
from .market_snapshot import MarketSnapshot, AssetData, IndicatorData

__all__ = [
    'TradingEngine',
    'MarketSnapshot',
    'AssetData',
    'IndicatorData',
]
