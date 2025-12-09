"""Market data source implementation using CCXT."""

import asyncio
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional

from termcolor import cprint

from ..models import Candle, InstrumentRef
from ..utils import get_exchange_cls, normalize_symbol
from .data_interfaces import BaseMarketDataSource, MarketSnapshotType


class SimpleMarketDataSource(BaseMarketDataSource):
    """Fetches market data via CCXT.pro.

    If `exchange_id` is provided, this class will fetch OHLCV data from the
    specified exchange. On any error (missing library, unknown exchange,
    network error), it logs a warning and returns empty data.
    """

    def __init__(self, exchange_id: Optional[str] = None) -> None:
        self._exchange_id = exchange_id or "binance"


    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize a symbol format for specific exchanges.

        Converts BTC-USDC to BTC/USDC:USDC (swap format)
        """
        base_symbol = symbol.replace("-", "/")

        if ":" not in base_symbol:
            parts = base_symbol.split("/")
            if len(parts) == 2:
                base_symbol = f"{parts[0]}/{parts[1]}:{parts[1]}"

        return base_symbol

    async def get_recent_candles(
        self, symbols: List[str], interval: str, lookback: int
    ) -> List[Candle]:
        """Fetch recent OHLCV candles for the given symbols."""

        async def _fetch_and_process(symbol: str) -> List[Candle]:
            exchange_cls = get_exchange_cls(self._exchange_id)
            exchange = exchange_cls({"newUpdates": False})

            symbol_candles: List[Candle] = []
            normalized_symbol = self._normalize_symbol(symbol)
            try:
                try:
                    raw = await exchange.fetch_ohlcv(
                        normalized_symbol,
                        timeframe=interval,
                        since=None,
                        limit=lookback,
                    )
                finally:
                    try:
                        await exchange.close()
                    except Exception:
                        pass

                for row in raw:
                    ts, open_v, high_v, low_v, close_v, vol = row
                    symbol_candles.append(
                        Candle(
                            ts=int(ts),
                            instrument=InstrumentRef(
                                symbol=symbol,
                                exchange_id=self._exchange_id,
                            ),
                            open=float(open_v),
                            high=float(high_v),
                            low=float(low_v),
                            close=float(close_v),
                            volume=float(vol),
                            interval=interval,
                        )
                    )
                return symbol_candles
            except Exception as exc:
                cprint(
                    f"Failed to fetch candles for {symbol} (normalized: {normalized_symbol}) from {self._exchange_id}, "
                    f"interval is {interval}, return empty candles. Error: {exc}",
                    "yellow"
                )
                return []

        tasks = [_fetch_and_process(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        candles: List[Candle] = list(itertools.chain.from_iterable(results))

        cprint(
            f"Fetched {len(candles)} candles ({lookback} × {len(symbols)} symbols), "
            f"interval: {interval}",
            "white"
        )
        return candles

    async def get_market_snapshot(self, symbols: List[str]) -> MarketSnapshotType:
        """Fetch latest prices for the given symbols.

        Tries to use fetch_ticker, fetch_open_interest, and fetch_funding_rate
        to build a comprehensive market snapshot.

        Returns:
            Dict mapping symbol to market data:
            {
                "BTC/USDT": {
                    "price": {...},       # ticker data
                    "open_interest": {...},
                    "funding_rate": {...},
                },
                ...
            }
        """
        snapshot: Dict[str, Dict[str, Any]] = defaultdict(dict)

        exchange_cls = get_exchange_cls(self._exchange_id)
        exchange = exchange_cls({"newUpdates": False})
        try:
            for symbol in symbols:
                sym = normalize_symbol(symbol)
                try:
                    ticker = await exchange.fetch_ticker(sym)
                    snapshot[symbol]["price"] = ticker

                    # Best-effort: fetch open interest
                    try:
                        oi = await exchange.fetch_open_interest(sym)
                        snapshot[symbol]["open_interest"] = oi
                    except Exception:
                        pass  # Silently ignore

                    # Best-effort: fetch funding rate
                    try:
                        fr = await exchange.fetch_funding_rate(sym)
                        snapshot[symbol]["funding_rate"] = fr
                    except Exception:
                        pass  # Silently ignore

                    cprint(f"Fetched market snapshot for {sym}", "magenta")
                except Exception as e:
                    cprint(
                        f"Failed to fetch market snapshot for {symbol} at {self._exchange_id}: {e}",
                        "red"
                    )
        finally:
            try:
                await exchange.close()
            except Exception as e:
                cprint(f"Failed to close exchange connection for {self._exchange_id}: {e}", "red")

        return dict(snapshot)
