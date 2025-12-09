"""Feature pipeline for computing technical indicators and market data.

This module provides the DefaultFeaturesPipeline which:
1. Fetches OHLCV candle data via CCXT
2. Computes technical indicators (EMA, MACD, RSI, Bollinger Bands)
3. Fetches market snapshots (ticker, funding rate, open interest)
4. Combines all features for LLM decision-making
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from termcolor import cprint

from ..models import (
    CandleConfig,
    FeaturesPipelineResult,
    FeatureVector,
    UserRequest,
)
from .data_interfaces import BaseMarketDataSource
from .data_source import SimpleMarketDataSource
from .candle import SimpleCandleFeatureComputer
from .feature_interfaces import BaseFeaturesPipeline, CandleBasedFeatureComputer
from .market_snapshot import SimpleMarketSnapshotFeatureComputer


class DefaultFeaturesPipeline(BaseFeaturesPipeline):
    """Default pipeline that fetches market data and computes technical indicators.

    Produces:
    - Candle-based features: EMA, MACD, RSI, Bollinger Bands
    - Market snapshot features: price, funding rate, open interest
    """

    def __init__(
        self,
        *,
        request: UserRequest,
        market_data_source: BaseMarketDataSource,
        candle_feature_computer: CandleBasedFeatureComputer,
        market_snapshot_computer: SimpleMarketSnapshotFeatureComputer,
        candle_configurations: Optional[List[CandleConfig]] = None,
    ) -> None:
        """Initialize features pipeline.

        Args:
            request: User request with configuration
            market_data_source: Market data source for fetching candles and snapshots
            candle_feature_computer: Computer for technical indicators
            market_snapshot_computer: Computer for market snapshot features
            candle_configurations: List of candle configs (interval, lookback pairs)
        """
        self._request = request
        self._market_data_source = market_data_source
        self._candle_feature_computer = candle_feature_computer
        self._market_snapshot_computer = market_snapshot_computer
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))

        # Default candle configurations
        # Binance 合约支持: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        # 注: 1s 仅现货支持，合约不支持
        self._candle_configurations = candle_configurations or [
            CandleConfig(interval="1m", lookback=60 * 3),      # 1 hour of 1m candles
            CandleConfig(interval="15m", lookback=60 * 4),     # 4 hours of 15m candles
        ]

    @classmethod
    def from_request(cls, request: UserRequest) -> DefaultFeaturesPipeline:
        """Factory creating the default pipeline from a user request."""
        exchange_id = request.exchange_config.exchange_id or "binance"

        market_data_source = SimpleMarketDataSource(exchange_id=exchange_id)
        candle_feature_computer = SimpleCandleFeatureComputer()
        market_snapshot_computer = SimpleMarketSnapshotFeatureComputer(
            exchange_id=exchange_id
        )

        return cls(
            request=request,
            market_data_source=market_data_source,
            candle_feature_computer=candle_feature_computer,
            market_snapshot_computer=market_snapshot_computer,
        )

    async def build(self) -> FeaturesPipelineResult:
        """Build feature vectors from market data.

        Fetches candles and market snapshot concurrently, computes features,
        and combines all results.
        """

        async def _fetch_candles(interval: str, lookback: int) -> List[FeatureVector]:
            """Fetch candles and compute features for a single config."""
            candles = await self._market_data_source.get_recent_candles(
                self._symbols, interval, lookback
            )
            return self._candle_feature_computer.compute_features(candles=candles)

        async def _fetch_market_features() -> List[FeatureVector]:
            """Fetch market snapshot and compute features."""
            market_snapshot = await self._market_data_source.get_market_snapshot(
                self._symbols
            )
            market_snapshot = market_snapshot or {}
            return self._market_snapshot_computer.compute_features(
                snapshot=market_snapshot
            )

        cprint(
            f"Starting concurrent data fetching for {len(self._candle_configurations)} "
            f"candle sets and market snapshot...",
            "white"
        )

        # Build concurrent tasks
        tasks = [
            _fetch_candles(config.interval, config.lookback)
            for config in self._candle_configurations
        ]
        tasks.append(_fetch_market_features())

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        cprint("Concurrent data fetching complete.", "white")

        # Process results, handling any exceptions
        all_features: List[FeatureVector] = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if i < len(self._candle_configurations):
                    config = self._candle_configurations[i]
                    cprint(
                        f"Failed to fetch candles for {config.interval}: {result}",
                        "yellow"
                    )
                else:
                    cprint(f"Failed to fetch market snapshot: {result}", "yellow")
                continue

            if isinstance(result, list):
                all_features.extend(result)

        return FeaturesPipelineResult(features=all_features)

    async def close(self) -> None:
        """Close any open connections."""
        pass
