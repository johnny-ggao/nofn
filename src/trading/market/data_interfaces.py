"""Interfaces for market data sources."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..models import Candle


# Type alias for market snapshot data
MarketSnapshotType = Dict[str, Dict[str, Any]]


class BaseMarketDataSource(ABC):
    """Abstract base class for market data sources.

    Implementations should fetch real-time and historical market data
    from exchanges via CCXT or other APIs.
    """

    @abstractmethod
    async def get_recent_candles(
        self,
        symbols: List[str],
        interval: str,
        lookback: int,
    ) -> List[Candle]:
        """Fetch recent OHLCV candles for the given symbols.

        Args:
            symbols: List of trading symbols (e.g., ["BTC/USDT", "ETH/USDT"])
            interval: Candle interval (e.g., "1m", "5m", "1h")
            lookback: Number of candles to fetch

        Returns:
            List of Candle objects
        """
        ...

    @abstractmethod
    async def get_market_snapshot(
        self,
        symbols: List[str],
    ) -> MarketSnapshotType:
        """Fetch the current market snapshot for the given symbols.

        Returns ticker, open interest, and funding rate data.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to market data:
            {
                "BTC/USDT": {
                    "price": {...}, # ticker data
                    "open_interest": {...},
                    "funding_rate": {...},
                },
                ...
            }
        """
        ...
