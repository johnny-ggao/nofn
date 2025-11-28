"""
交易所适配器模块

提供多交易所的统一接口和管理：
- BaseExchangeAdapter: 适配器基类
- HyperliquidAdapter: Hyperliquid 永续合约
- BinanceAdapter: Binance 永续合约 (USDT-M / USDC-M)
- AdapterManager: 适配器管理器
"""
from .base import BaseExchangeAdapter
from .hyperliquid import HyperliquidAdapter
from .binance import BinanceAdapter
from .manager import (
    AdapterManager,
    AdapterCapability,
    AdapterContext,
    get_adapter_manager,
)

__all__ = [
    # 适配器
    "BaseExchangeAdapter",
    "HyperliquidAdapter",
    "BinanceAdapter",
    # 管理器
    "AdapterManager",
    "AdapterCapability",
    "AdapterContext",
    "get_adapter_manager",
]