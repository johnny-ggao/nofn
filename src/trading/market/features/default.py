"""默认特征计算器 - 通用技术指标组合。

包含常用的技术指标，适用于大多数交易策略：
- 趋势指标：EMA (12, 26, 50), ADX
- 动量指标：MACD, RSI
- 波动率指标：ATR, Bollinger Bands

输出包含当前值和历史序列，便于 LLM 分析趋势变化。
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...constants import (
    FEATURE_GROUP_BY_INTERVAL_PREFIX,
    FEATURE_GROUP_BY_KEY,
)
from ...models import Candle, FeatureVector
from .base import CandleBasedFeatureComputer


class DefaultFeatureComputer(CandleBasedFeatureComputer):
    """默认特征计算器 - 通用技术指标组合。

    包含的指标：
    - EMA (12, 26, 50)
    - MACD (with signal and histogram)
    - RSI (14-period)
    - Bollinger Bands (20, 2σ)
    - ATR (14-period Average True Range)
    - ADX (14-period Average Directional Index with +DI and -DI)

    历史序列输出（最近 N 个值）:
    - ohlcv_history: K 线原始数据
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
        """计算技术指标特征。

        Args:
            candles: K 线数据列表
            meta: 可选的元数据

        Returns:
            每个交易对一个 FeatureVector，包含当前指标值和历史序列
        """
        if not candles:
            return []

        # 按交易对分组
        grouped: Dict[str, List[Candle]] = defaultdict(list)
        for candle in candles:
            grouped[candle.instrument.symbol].append(candle)

        features: List[FeatureVector] = []
        for symbol, series in grouped.items():
            # 按时间排序并构建 DataFrame
            series.sort(key=lambda item: item.ts)
            df = self._build_dataframe(series)

            # 计算指标
            df = self._compute_ema(df)
            df = self._compute_macd(df)
            df = self._compute_rsi(df)
            df = self._compute_bollinger_bands(df)
            df = self._compute_atr(df)
            df = self._compute_adx(df)

            # 构建输出
            values = self._extract_values(df)
            fv_meta = self._build_meta(df, series, meta)

            features.append(
                FeatureVector(
                    ts=int(df.iloc[-1]["ts"]),
                    instrument=series[-1].instrument,
                    values=values,
                    meta=fv_meta,
                )
            )

        return features

    def _build_dataframe(self, series: List[Candle]) -> pd.DataFrame:
        """将 K 线列表转换为 DataFrame。"""
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
        return pd.DataFrame(rows)

    def _compute_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 EMA 指标。"""
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        return df

    def _compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 MACD 指标。"""
        if "ema_12" not in df.columns:
            df = self._compute_ema(df)
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 RSI 指标。"""
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta).clip(lower=0).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _compute_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """计算布林带。"""
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        bb_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * std_dev)
        df["bb_lower"] = df["bb_middle"] - (bb_std * std_dev)
        return df

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 ATR (Average True Range)。"""
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["prev_close"])
        df["tr3"] = abs(df["low"] - df["prev_close"])
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=period).mean()
        return df

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 ADX (Average Directional Index)。"""
        # 确保 ATR 已计算
        if "true_range" not in df.columns:
            df = self._compute_atr(df, period)

        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)

        # 方向运动
        df["high_diff"] = df["high"] - df["prev_high"]
        df["low_diff"] = df["prev_low"] - df["low"]

        # +DM 和 -DM
        df["plus_dm"] = np.where(
            (df["high_diff"] > df["low_diff"]) & (df["high_diff"] > 0),
            df["high_diff"],
            0,
        )
        df["minus_dm"] = np.where(
            (df["low_diff"] > df["high_diff"]) & (df["low_diff"] > 0),
            df["low_diff"],
            0,
        )

        # 平滑 DM 和 TR
        df["plus_dm_smooth"] = df["plus_dm"].ewm(span=period, adjust=False).mean()
        df["minus_dm_smooth"] = df["minus_dm"].ewm(span=period, adjust=False).mean()
        df["tr_smooth"] = df["true_range"].ewm(span=period, adjust=False).mean()

        # +DI 和 -DI
        df["plus_di"] = 100 * (df["plus_dm_smooth"] / df["tr_smooth"])
        df["minus_di"] = 100 * (df["minus_dm_smooth"] / df["tr_smooth"])

        # DX 和 ADX
        df["di_diff"] = abs(df["plus_di"] - df["minus_di"])
        df["di_sum"] = df["plus_di"] + df["minus_di"]
        df["dx"] = 100 * (df["di_diff"] / df["di_sum"].replace(0, np.inf))
        df["adx"] = df["dx"].ewm(span=period, adjust=False).mean()

        return df

    def _extract_values(self, df: pd.DataFrame) -> Dict:
        """从 DataFrame 提取当前值和历史序列。"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        change_pct = (
            (float(last["close"]) - float(prev["close"])) / float(prev["close"])
            if prev["close"]
            else 0.0
        )

        def safe_float(val):
            if pd.isna(val):
                return None
            return float(val)

        def extract_history(series: pd.Series, length: int) -> List[float]:
            """提取最近 N 个非 NaN 值，保留 2 位小数。"""
            valid = series.dropna().tail(length)
            return [round(float(v), 2) for v in valid.tolist()]

        hist_len = self._history_length

        # K 线 OHLCV 历史
        ohlcv_history = []
        for _, row in df.tail(hist_len).iterrows():
            ohlcv_history.append(
                {
                    "ts": int(row["ts"]),
                    "o": round(float(row["open"]), 2),
                    "h": round(float(row["high"]), 2),
                    "l": round(float(row["low"]), 2),
                    "c": round(float(row["close"]), 2),
                    "v": round(float(row["volume"]), 2),
                }
            )

        return {
            # 当前值
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
            "atr": safe_float(last.get("atr")),
            "adx": safe_float(last.get("adx")),
            "plus_di": safe_float(last.get("plus_di")),
            "minus_di": safe_float(last.get("minus_di")),
            # 历史序列（从旧到新）
            "ohlcv_history": ohlcv_history,
            "ema_12_history": extract_history(df["ema_12"], hist_len),
            "ema_26_history": extract_history(df["ema_26"], hist_len),
            "ema_50_history": extract_history(df["ema_50"], hist_len),
            "macd_history": extract_history(df["macd"], hist_len),
            "macd_signal_history": extract_history(df["macd_signal"], hist_len),
            "macd_histogram_history": extract_history(df["macd_histogram"], hist_len),
            "rsi_history": extract_history(df["rsi"], hist_len),
        }

    def _build_meta(
        self,
        df: pd.DataFrame,
        series: List[Candle],
        meta: Optional[Dict[str, object]],
    ) -> Dict:
        """构建特征元数据。"""
        last = df.iloc[-1]
        interval = series[-1].interval

        fv_meta = {
            FEATURE_GROUP_BY_KEY: f"{FEATURE_GROUP_BY_INTERVAL_PREFIX}{interval}",
            "interval": interval,
            "count": len(series),
            "window_start_ts": int(df.iloc[0]["ts"]),
            "window_end_ts": int(last["ts"]),
        }

        if meta:
            for k, v in meta.items():
                fv_meta.setdefault(k, v)

        return fv_meta


# 保持向后兼容的别名
SimpleCandleFeatureComputer = DefaultFeatureComputer
