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
    - ATR (14-period Average True Range)
    - ADX (14-period Average Directional Index with +DI and -DI)

    历史序列指标（最近 N 个值）:
    - ema_12_history, ema_26_history, ema_50_history
    - macd_history, macd_signal_history, macd_histogram_history
    - rsi_history
    """

    def __init__(self, history_length: int = 30) -> None:
        """初始化特征计算器。

        Args:
            history_length: 历史指标序列的长度，默认 30
        """
        self._history_length = history_length

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

            # ATR (Average True Range) - 14-period
            # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
            df["prev_close"] = df["close"].shift(1)
            df["prev_high"] = df["high"].shift(1)
            df["prev_low"] = df["low"].shift(1)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["prev_close"])
            df["tr3"] = abs(df["low"] - df["prev_close"])
            df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            df["atr"] = df["true_range"].rolling(window=14).mean()

            # ADX (Average Directional Index) - 14-period
            # Calculate directional movement
            df["high_diff"] = df["high"] - df["prev_high"]
            df["low_diff"] = df["prev_low"] - df["low"]

            # +DM and -DM (Directional Movement)
            df["plus_dm"] = np.where(
                (df["high_diff"] > df["low_diff"]) & (df["high_diff"] > 0),
                df["high_diff"],
                0
            )
            df["minus_dm"] = np.where(
                (df["low_diff"] > df["high_diff"]) & (df["low_diff"] > 0),
                df["low_diff"],
                0
            )

            # Smooth DM and TR using EMA
            period = 14
            df["plus_dm_smooth"] = df["plus_dm"].ewm(span=period, adjust=False).mean()
            df["minus_dm_smooth"] = df["minus_dm"].ewm(span=period, adjust=False).mean()
            df["tr_smooth"] = df["true_range"].ewm(span=period, adjust=False).mean()

            # +DI and -DI (Directional Indicators)
            df["plus_di"] = 100 * (df["plus_dm_smooth"] / df["tr_smooth"])
            df["minus_di"] = 100 * (df["minus_dm_smooth"] / df["tr_smooth"])

            # DX (Directional Index)
            df["di_diff"] = abs(df["plus_di"] - df["minus_di"])
            df["di_sum"] = df["plus_di"] + df["minus_di"]
            df["dx"] = 100 * (df["di_diff"] / df["di_sum"].replace(0, np.inf))

            # ADX (Average Directional Index)
            df["adx"] = df["dx"].ewm(span=period, adjust=False).mean()

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

            def extract_history(series: pd.Series, length: int) -> List[float]:
                """提取最近 N 个非 NaN 值，保留 2 位小数。"""
                valid = series.dropna().tail(length)
                return [round(float(v), 2) for v in valid.tolist()]

            # 提取历史序列（最近 N 个值）
            hist_len = self._history_length

            # K 线 OHLCV 历史数据
            ohlcv_history = []
            for _, row in df.tail(hist_len).iterrows():
                ohlcv_history.append({
                    "ts": int(row["ts"]),
                    "o": round(float(row["open"]), 2),
                    "h": round(float(row["high"]), 2),
                    "l": round(float(row["low"]), 2),
                    "c": round(float(row["close"]), 2),
                    "v": round(float(row["volume"]), 2),
                })

            # 指标历史序列
            ema_12_hist = extract_history(df["ema_12"], hist_len)
            ema_26_hist = extract_history(df["ema_26"], hist_len)
            ema_50_hist = extract_history(df["ema_50"], hist_len)
            macd_hist = extract_history(df["macd"], hist_len)
            macd_signal_hist = extract_history(df["macd_signal"], hist_len)
            macd_histogram_hist = extract_history(df["macd_histogram"], hist_len)
            rsi_hist = extract_history(df["rsi"], hist_len)

            values = {
                "close": float(last["close"]),
                "volume": float(last["volume"]),
                "change_pct": float(change_pct),
                # 当前值
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
                "atr": safe_float(last.get("atr")),
                "adx": safe_float(last.get("adx")),
                "plus_di": safe_float(last.get("plus_di")),
                "minus_di": safe_float(last.get("minus_di")),
                # K 线 OHLCV 历史（从旧到新）
                "ohlcv_history": ohlcv_history,
                # 指标历史序列（从旧到新）
                "ema_12_history": ema_12_hist,
                "ema_26_history": ema_26_hist,
                "ema_50_history": ema_50_hist,
                "macd_history": macd_hist,
                "macd_signal_history": macd_signal_hist,
                "macd_histogram_history": macd_histogram_hist,
                "rsi_history": rsi_hist,
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
