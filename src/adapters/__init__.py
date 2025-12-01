"""
交易所适配器模块

提供多交易所的统一接口：
- BaseExchangeAdapter: 适配器基类
- HyperliquidAdapter: Hyperliquid 永续合约
- BinanceAdapter: Binance 永续合约 (USDT-M / USDC-M)
- AdapterFactory: 适配器工厂
"""
from .base import BaseExchangeAdapter
from .hyperliquid import HyperliquidAdapter
from .binance import BinanceAdapter
from .factory import AdapterFactory, create_adapter

__all__ = [
    # 适配器
    "BaseExchangeAdapter",
    "HyperliquidAdapter",
    "BinanceAdapter",
    # 工厂
    "AdapterFactory",
    "create_adapter",
]