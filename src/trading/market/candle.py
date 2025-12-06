"""Candle-based feature computation with technical indicators."""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..constants import (
    FEATURE_GROUP_BY_INTERVAL_PREFIX,
    FEATURE_GROUP_BY_KEY,
)
from ..models import Candle, FeatureVector
from .feature_interfaces import CandleBasedFeatureComputer


class SimpleCandleFeatureComputer(CandleBasedFeatureComputer):
    """Computes technical indicators from candle data.

    Supported indicators:
    - EMA (12, 26, 50)
    - MACD (with signal and histogram)
    - RSI (14-period)
    - Bollinger Bands (20, 2σ)
    """

    def compute_features(
        self,
        candles: Optional[List[Candle]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        """Compute technical indicators from candle data.

        Args:
            candles: List of OHLCV candles
            meta: Optional metadata to include in feature vectors

        Returns:
            List of FeatureVector objects with computed indicators
        """
        if not candles:
            return []

        # Group candles by symbol
        grouped: Dict[str, List[Candle]] = defaultdict(list)
        for candle in candles:
            grouped[candle.instrument.symbol].append(candle)

        features: List[FeatureVector] = []
        for symbol, series in grouped.items():
            # Build a DataFrame for indicator calculations
            series.sort(key=lambda item: item.ts)
            rows = [
                {
                    "ts": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "interval": c.interval,
                }
                for c in series
            ]
            df = pd.DataFrame(rows)

            # EMAs
            df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

            # RSI
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta).clip(lower=0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            df["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            change_pct = (
                (float(last["close"]) - float(prev["close"])) / float(prev["close"])
                if prev["close"]
                else 0.0
            )

            # Helper to safely extract float values
            def safe_float(val):
                if pd.isna(val):
                    return None
                return float(val)

            values = {
                "close": float(last["close"]),
                "volume": float(last["volume"]),
                "change_pct": float(change_pct),
                "ema_12": safe_float(last.get("ema_12")),
                "ema_26": safe_float(last.get("ema_26")),
                "ema_50": safe_float(last.get("ema_50")),
                "macd": safe_float(last.get("macd")),
                "macd_signal": safe_float(last.get("macd_signal")),
                "macd_histogram": safe_float(last.get("macd_histogram")),
                "rsi": safe_float(last.get("rsi")),
                "bb_upper": safe_float(last.get("bb_upper")),
                "bb_middle": safe_float(last.get("bb_middle")),
                "bb_lower": safe_float(last.get("bb_lower")),
            }

            # Build feature meta
            window_start_ts = int(rows[0]["ts"]) if rows else int(last["ts"])
            window_end_ts = int(last["ts"])
            interval = series[-1].interval
            fv_meta = {
                FEATURE_GROUP_BY_KEY: f"{FEATURE_GROUP_BY_INTERVAL_PREFIX}{interval}",
                "interval": interval,
                "count": len(series),
                "window_start_ts": window_start_ts,
                "window_end_ts": window_end_ts,
            }
            if meta:
                for k, v in meta.items():
                    fv_meta.setdefault(k, v)

            features.append(
                FeatureVector(
                    ts=int(last["ts"]),
                    instrument=series[-1].instrument,
                    values=values,
                    meta=fv_meta,
                )
            )
        return features
