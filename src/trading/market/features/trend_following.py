"""趋势跟踪特征计算器
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

class TrendFollowingFeatureComputer(CandleBasedFeatureComputer):
    """趋势跟踪特征计算器。

    包含的指标：
    - EMA (20, 50, 200) - 多周期均线判断趋势
    - MACD (12, 26, 9) - 趋势动量指标
    - RSI (14) - 超买超卖判断
    - ATR (14) - 波动率评估
    - ADX (14) - 趋势强度指标
    - Bollinger Bands (20, 2) - 波动区间和超买超卖

    适用于中长期趋势跟踪策略。
    """

    def __init__(self, history_length: int = 10) -> None:
        """初始化趋势跟踪特征计算器。

        Args:
            history_length: 历史数据长度，默认 10（减少 token 消耗）
        """
        self.history_length = history_length

    def get_feature_instructions(self) -> str:
        """获取趋势跟踪策略的特征说明。"""
        return (
            "技术指标说明:\n"
            "- ohlcv: 历史K线，格式 [[ts,o,h,l,c,v],...] (时间戳,开,高,低,收,量)\n"
            "- ema_20/50/200: 指数移动平均线\n"
            "- macd/macd_signal/macd_histogram: MACD(12,26,9)\n"
            "- rsi: RSI(14)，>70超买，<30超卖\n"
            "- atr: 平均真实波幅\n"
            "- adx/plus_di/minus_di: ADX(14)，>25趋势明显\n"
            "- bb_upper/middle/lower/width/pct: 布林带(20,2)\n"
            "- *_hist: 历史序列(数组)，最后一个是最新值\n"
        )

    def compute_features(
        self,
        candles: Optional[List[Candle]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        
        if not candles:
            return []
        
        grouped: Dict[str, List[Candle]] = defaultdict(list)
        for candle in candles:
            grouped[candle.instrument.symbol].append(candle)
            
        features: List[FeatureVector] = []
        for symbol, series in grouped.items():
            series.sort(key=lambda item: item.ts)
            df = self._build_dataframe(series)
            
            # 计算指标
            df = self._compute_ema(df)
            df = self._compute_macd(df)
            df = self._compute_rsi(df)
            df = self._compute_atr(df)
            df = self._compute_adx(df)
            df = self._compute_bollinger_bands(df)
            
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

    def _compute_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 EMA 指标。"""
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        return df

    def _compute_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """计算 MACD 指标。"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta).clip(lower=0).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        df["rsi"] = 100 - (100 / (1 + rs))
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

    def _compute_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """计算布林带 (Bollinger Bands)。

        Args:
            df: 包含 close 列的 DataFrame
            period: 移动平均周期，默认 20
            std_dev: 标准差倍数，默认 2.0

        Returns:
            添加了 bb_upper, bb_middle, bb_lower, bb_width, bb_pct 列的 DataFrame
        """
        # 中轨 - 简单移动平均
        df["bb_middle"] = df["close"].rolling(window=period).mean()

        # 标准差
        rolling_std = df["close"].rolling(window=period).std()

        # 上下轨
        df["bb_upper"] = df["bb_middle"] + (rolling_std * std_dev)
        df["bb_lower"] = df["bb_middle"] - (rolling_std * std_dev)

        # 布林带宽度 (bandwidth) - 反映波动率
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # %B - 价格在布林带中的位置 (0 = 下轨, 1 = 上轨)
        band_range = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / band_range.replace(0, np.inf)

        return df
    
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
    
    def _extract_values(self, df: pd.DataFrame) -> Dict:
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
            """提取最近 N 个非 NaN 值，保留 1 位小数（减少 token）。"""
            valid = series.dropna().tail(length)
            return [round(float(v), 1) for v in valid.tolist()]

        hist_len = self.history_length

        # 压缩格式: [[ts, o, h, l, c, v], ...] 代替 [{"ts":..., "o":...}, ...]
        # 时间戳使用秒（而非毫秒）进一步节省空间
        ohlcv_history = []
        for _, row in df.tail(hist_len).iterrows():
            ohlcv_history.append([
                int(row["ts"] // 1000),  # 秒级时间戳
                round(float(row["open"]), 1),
                round(float(row["high"]), 1),
                round(float(row["low"]), 1),
                round(float(row["close"]), 1),
                round(float(row["volume"]), 1),
            ])
        
        return {
            # 当前值（保留完整字段名，便于理解）
            "close": round(float(last["close"]), 1),
            "volume": round(float(last["volume"]), 1),
            "change_pct": round(float(change_pct), 4),
            # EMA
            "ema_20": safe_float(last.get("ema_20")),
            "ema_50": safe_float(last.get("ema_50")),
            "ema_200": safe_float(last.get("ema_200")),
            # MACD
            "macd": safe_float(last.get("macd")),
            "macd_signal": safe_float(last.get("macd_signal")),
            "macd_histogram": safe_float(last.get("macd_histogram")),
            # RSI
            "rsi": safe_float(last.get("rsi")),
            # ATR
            "atr": safe_float(last.get("atr")),
            # ADX
            "adx": safe_float(last.get("adx")),
            "plus_di": safe_float(last.get("plus_di")),
            "minus_di": safe_float(last.get("minus_di")),
            # Bollinger Bands
            "bb_upper": safe_float(last.get("bb_upper")),
            "bb_middle": safe_float(last.get("bb_middle")),
            "bb_lower": safe_float(last.get("bb_lower")),
            "bb_width": safe_float(last.get("bb_width")),
            "bb_pct": safe_float(last.get("bb_pct")),
            # 历史序列（压缩字段名 *_hist，从旧到新）
            "ohlcv": ohlcv_history,  # [[ts,o,h,l,c,v], ...]
            "ema20_hist": extract_history(df["ema_20"], hist_len),
            "ema50_hist": extract_history(df["ema_50"], hist_len),
            "ema200_hist": extract_history(df["ema_200"], hist_len),
            "macd_hist": extract_history(df["macd"], hist_len),
            "macd_sig_hist": extract_history(df["macd_signal"], hist_len),
            "macd_bar_hist": extract_history(df["macd_histogram"], hist_len),
            "rsi_hist": extract_history(df["rsi"], hist_len),
            "bb_up_hist": extract_history(df["bb_upper"], hist_len),
            "bb_mid_hist": extract_history(df["bb_middle"], hist_len),
            "bb_lo_hist": extract_history(df["bb_lower"], hist_len),
            "bb_w_hist": extract_history(df["bb_width"], hist_len),
            "bb_pct_hist": extract_history(df["bb_pct"], hist_len),
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
    