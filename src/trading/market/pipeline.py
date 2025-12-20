"""Feature pipeline for computing technical indicators and market data.

This module provides the DefaultFeaturesPipeline which:
1. Fetches OHLCV candle data via CCXT
2. Computes technical indicators based on configured feature computers
3. Fetches market snapshots (ticker, funding rate, open interest)
4. Combines all features for LLM decision-making

Each candle configuration can specify its own feature computer, allowing
different indicators for different timeframes.
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
from .features import get_feature_computer
from .feature_interfaces import BaseFeaturesPipeline
from .market_snapshot import SimpleMarketSnapshotFeatureComputer


class DefaultFeaturesPipeline(BaseFeaturesPipeline):
    """Default pipeline that fetches market data and computes technical indicators.

    Supports per-timeframe feature computer configuration via CandleConfig.feature_computer.

    Produces:
    - Candle-based features: configurable per timeframe
    - Market snapshot features: price, funding rate, open interest
    """

    def __init__(
        self,
        *,
        request: UserRequest,
        market_data_source: BaseMarketDataSource,
        market_snapshot_computer: SimpleMarketSnapshotFeatureComputer,
        candle_configurations: Optional[List[CandleConfig]] = None,
    ) -> None:
        """Initialize features pipeline.

        Args:
            request: User request with configuration
            market_data_source: Market data source for fetching candles and snapshots
            market_snapshot_computer: Computer for market snapshot features
            candle_configurations: List of candle configs (interval, lookback, feature_computer)
        """
        self._request = request
        self._market_data_source = market_data_source
        self._market_snapshot_computer = market_snapshot_computer
        self._symbols = list(dict.fromkeys(request.trading_config.symbols))

        # Default candle configurations
        self._candle_configurations = candle_configurations or [
            CandleConfig(interval="15m", lookback=60 * 3, feature_computer="trend_following"),
            CandleConfig(interval="4h", lookback=60 * 3, feature_computer="trend_following"),
        ]

    @classmethod
    def from_request(cls, request: UserRequest) -> DefaultFeaturesPipeline:
        """Factory creating the default pipeline from a user request."""
        exchange_id = request.exchange_config.exchange_id or "binance"

        market_data_source = SimpleMarketDataSource(exchange_id=exchange_id)
        market_snapshot_computer = SimpleMarketSnapshotFeatureComputer(
            exchange_id=exchange_id
        )

        return cls(
            request=request,
            market_data_source=market_data_source,
            market_snapshot_computer=market_snapshot_computer,
        )

    async def build(self) -> FeaturesPipelineResult:
        """Build feature vectors from market data.

        Fetches candles and market snapshot concurrently, computes features,
        and combines all results. Each timeframe uses its configured feature computer.
        """

        async def _fetch_candles(config: CandleConfig) -> List[FeatureVector]:
            """获取 K 线并使用配置的计算器计算特征。"""
            candles = await self._market_data_source.get_recent_candles(
                self._symbols, config.interval, config.lookback
            )
            cprint(
                f"[{config.interval}] candles: {len(candles)}, "
                f"computer: {config.feature_computer}",
                "white",
            )
            # 根据配置获取对应的特征计算器
            computer = get_feature_computer(config.feature_computer)
            return computer.compute_features(candles=candles)

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
        tasks = [_fetch_candles(config) for config in self._candle_configurations]
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

        # 收集所有特征计算器的说明（去重）
        feature_instructions = self._collect_feature_instructions()

        return FeaturesPipelineResult(
            features=candle_features,
            feature_instructions=feature_instructions,
        )

    def _collect_feature_instructions(self) -> str:
        """收集所有配置的特征计算器的说明。"""
        seen_computers = set()
        instructions_parts = []

        for config in self._candle_configurations:
            if config.feature_computer in seen_computers:
                continue
            seen_computers.add(config.feature_computer)

            computer = get_feature_computer(config.feature_computer)
            instruction = computer.get_feature_instructions()
            if instruction:
                instructions_parts.append(instruction)

        return "\n\n".join(instructions_parts)

    async def close(self) -> None:
        """Close any open connections."""
        pass
