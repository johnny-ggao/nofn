"""Feature pipeline for computing technical indicators and market data.

This module provides the DefaultFeaturesPipeline which:
1. Fetches OHLCV candle data via CCXT
2. Computes technical indicators (EMA, MACD, RSI, Bollinger Bands, ATR, ADX)
3. Fetches market snapshots (ticker, funding rate, open interest)
4. Combines all features for LLM decision-making
"""

from __future__ import annotations

import asyncio
import itertools
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
    - Candle-based features: EMA, MACD, RSI, Bollinger Bands, ATR, ADX
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
        self._candle_configurations = candle_configurations or [
            CandleConfig(interval="3m", lookback=60 * 3),
            CandleConfig(interval="4h", lookback=60 * 4),
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
            cprint(f"candles count: {len(candles)}", "white")
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

        # Build concurrent tasks: [candle_task_1, candle_task_2, ..., market_task]
        tasks = [
            _fetch_candles(config.interval, config.lookback)
            for config in self._candle_configurations
        ]
        tasks.append(_fetch_market_features())

        # Execute concurrently
        # results = [ [candle_features_1], [candle_features_2], ..., [market_features] ]
        results = await asyncio.gather(*tasks)
        cprint("Concurrent data fetching complete.", "white")

        # Extract market features (last item)
        market_features: List[FeatureVector] = results.pop()

        # Flatten the list of lists of candle features
        candle_features: List[FeatureVector] = list(
            itertools.chain.from_iterable(results)
        )

        # Combine all features
        candle_features.extend(market_features)

        cprint(f"candle_features count: {len(candle_features)}", "white")

        return FeaturesPipelineResult(features=candle_features)

    async def close(self) -> None:
        """Close any open connections."""
        pass
