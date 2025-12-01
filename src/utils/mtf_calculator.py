"""
多时间框架指标计算器 (Multi-Timeframe Calculator)

根据不同时间框架计算相应的技术指标:
- 1小时: 趋势确认
- 15分钟: 入场时机
- 5分钟: 精确入场
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .indicators import (
    ema, macd, rsi, atr, bollinger_bands,
    stochastic_slow, adx, volume_roc, volume_sma, to_float_array
)
from ..engine.market_snapshot import TimeframeIndicators, Timeframe


@dataclass
class OHLCVData:
    """OHLCV K线数据"""
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


class MTFCalculator:
    """
    多时间框架指标计算器

    用法:
        calculator = MTFCalculator()

        # 计算1小时指标 (趋势确认)
        tf_1h = calculator.calculate_1h(ohlcv_1h, current_price)

        # 计算15分钟指标 (入场时机)
        tf_15m = calculator.calculate_15m(ohlcv_15m, current_price)

        # 计算5分钟指标 (精确入场)
        tf_5m = calculator.calculate_5m(ohlcv_5m, current_price)
    """

    @staticmethod
    def calculate_1h(
        data: OHLCVData,
        current_price: float,
        series_length: int = 10
    ) -> TimeframeIndicators:
        """
        计算1小时级别指标 (趋势确认)

        指标配置:
        - EMA(8, 21, 50): 判断趋势方向和多空排列
        - MACD(6, 13, 5): 识别趋势转折点和动量变化
        - RSI(14): 判断超买(>70)、超卖(<30)状态
        - ADX(14) + DI: 判断趋势强度(ADX>25表示强趋势)
        - ATR(14): 衡量市场波动性，计算止损距离
        - BB(20, 2): 判断价格波动区间和超买超卖

        Args:
            data: OHLCV数据
            current_price: 当前价格
            series_length: 序列数据长度

        Returns:
            TimeframeIndicators: 1小时级别指标
        """
        closes = to_float_array(data.close)
        highs = to_float_array(data.high)
        lows = to_float_array(data.low)
        n = len(closes)

        indicators = TimeframeIndicators(timeframe="1h")

        # EMA (8, 21, 50)
        if n >= 8:
            ema8_arr = ema(closes, 8)
            indicators.ema8 = float(ema8_arr[-1]) if not np.isnan(ema8_arr[-1]) else None
            indicators.ema8_series = _get_series(ema8_arr, series_length)

        if n >= 21:
            ema21_arr = ema(closes, 21)
            indicators.ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else None
            indicators.ema21_series = _get_series(ema21_arr, series_length)

        if n >= 50:
            ema50_arr = ema(closes, 50)
            indicators.ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else None
            indicators.ema50_series = _get_series(ema50_arr, series_length)

        # MACD (6, 13, 5) - 快速参数
        if n >= 18:  # 13 + 5
            macd_line, signal_line, histogram = macd(closes, fast_period=6, slow_period=13, signal_period=5)
            indicators.macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
            indicators.macd_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None
            indicators.macd_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
            indicators.macd_series = _get_series(macd_line, series_length)

        # RSI (14)
        if n >= 15:
            rsi_arr = rsi(closes, 14)
            indicators.rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
            indicators.rsi_series = _get_series(rsi_arr, series_length)

        # ADX (14)
        if n >= 28:  # 需要 2 * period
            adx_arr, plus_di_arr, minus_di_arr = adx(highs, lows, closes, 14)
            indicators.adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else None
            indicators.plus_di = float(plus_di_arr[-1]) if not np.isnan(plus_di_arr[-1]) else None
            indicators.minus_di = float(minus_di_arr[-1]) if not np.isnan(minus_di_arr[-1]) else None

        # ATR (14)
        if n >= 15:
            atr_arr = atr(highs, lows, closes, 14)
            atr_value = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else None
            indicators.atr = atr_value
            if atr_value and current_price > 0:
                indicators.atr_percent = (atr_value / current_price) * 100

        # Bollinger Bands (20, 2)
        if n >= 20:
            bb_upper, bb_middle, bb_lower = bollinger_bands(closes, 20, 2.0)
            indicators.bb_upper = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
            indicators.bb_middle = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None
            indicators.bb_lower = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None

        # 价格序列
        indicators.prices_series = _get_series(closes, series_length)

        return indicators

    @staticmethod
    def calculate_15m(
        data: OHLCVData,
        current_price: float,
        series_length: int = 10
    ) -> TimeframeIndicators:
        """
        计算15分钟级别指标 (入场时机)

        指标配置:
        - EMA(8, 21, 50)
        - RSI(14)
        - MACD(6, 13, 5)
        - Stochastic(14, 3, 3): 识别超买超卖和背离信号
        - Volume ROC(5): 确认价格变动是否有成交量支撑
        - ATR(14)

        Args:
            data: OHLCV数据
            current_price: 当前价格
            series_length: 序列数据长度

        Returns:
            TimeframeIndicators: 15分钟级别指标
        """
        closes = to_float_array(data.close)
        highs = to_float_array(data.high)
        lows = to_float_array(data.low)
        volumes = to_float_array(data.volume)
        n = len(closes)

        indicators = TimeframeIndicators(timeframe="15m")

        # EMA (8, 21, 50)
        if n >= 8:
            ema8_arr = ema(closes, 8)
            indicators.ema8 = float(ema8_arr[-1]) if not np.isnan(ema8_arr[-1]) else None
            indicators.ema8_series = _get_series(ema8_arr, series_length)

        if n >= 21:
            ema21_arr = ema(closes, 21)
            indicators.ema21 = float(ema21_arr[-1]) if not np.isnan(ema21_arr[-1]) else None
            indicators.ema21_series = _get_series(ema21_arr, series_length)

        if n >= 50:
            ema50_arr = ema(closes, 50)
            indicators.ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else None
            indicators.ema50_series = _get_series(ema50_arr, series_length)

        # MACD (6, 13, 5)
        if n >= 18:
            macd_line, signal_line, histogram = macd(closes, fast_period=6, slow_period=13, signal_period=5)
            indicators.macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
            indicators.macd_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None
            indicators.macd_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
            indicators.macd_series = _get_series(macd_line, series_length)

        # RSI (14)
        if n >= 15:
            rsi_arr = rsi(closes, 14)
            indicators.rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
            indicators.rsi_series = _get_series(rsi_arr, series_length)

        # Stochastic (14, 3, 3)
        if n >= 20:
            stoch_k, stoch_d = stochastic_slow(highs, lows, closes, k_period=14, k_smooth=3, d_period=3)
            indicators.stoch_k = float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None
            indicators.stoch_d = float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else None

        # Volume ROC (5)
        if n >= 6:
            vol_roc_arr = volume_roc(volumes, 5)
            indicators.volume_roc = float(vol_roc_arr[-1]) if not np.isnan(vol_roc_arr[-1]) else None

        # ATR (14)
        if n >= 15:
            atr_arr = atr(highs, lows, closes, 14)
            atr_value = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else None
            indicators.atr = atr_value
            if atr_value and current_price > 0:
                indicators.atr_percent = (atr_value / current_price) * 100

        # 价格序列
        indicators.prices_series = _get_series(closes, series_length)

        return indicators

    @staticmethod
    def calculate_5m(
        data: OHLCVData,
        current_price: float,
        series_length: int = 10
    ) -> TimeframeIndicators:
        """
        计算5分钟级别指标 (精确入场)

        指标配置:
        - EMA(8): 快速均线
        - RSI(9): 更敏感的参数
        - MACD(5, 10, 3): 超快速参数
        - Volume MA(5): 成交量均线

        Args:
            data: OHLCV数据
            current_price: 当前价格
            series_length: 序列数据长度

        Returns:
            TimeframeIndicators: 5分钟级别指标
        """
        closes = to_float_array(data.close)
        highs = to_float_array(data.high)
        lows = to_float_array(data.low)
        volumes = to_float_array(data.volume)
        n = len(closes)

        indicators = TimeframeIndicators(timeframe="5m")

        # EMA (8) - 快速均线
        if n >= 8:
            ema8_arr = ema(closes, 8)
            indicators.ema8 = float(ema8_arr[-1]) if not np.isnan(ema8_arr[-1]) else None
            indicators.ema8_series = _get_series(ema8_arr, series_length)

        # RSI (9) - 更敏感
        if n >= 10:
            rsi_arr = rsi(closes, 9)
            indicators.rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
            indicators.rsi_series = _get_series(rsi_arr, series_length)

        # MACD (5, 10, 3) - 超快速参数
        if n >= 13:
            macd_line, signal_line, histogram = macd(closes, fast_period=5, slow_period=10, signal_period=3)
            indicators.macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
            indicators.macd_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None
            indicators.macd_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
            indicators.macd_series = _get_series(macd_line, series_length)

        # Volume MA (5)
        if n >= 5:
            vol_ma_arr = volume_sma(volumes, 5)
            vol_ma = float(vol_ma_arr[-1]) if not np.isnan(vol_ma_arr[-1]) else None
            indicators.volume_ma = vol_ma

            # 计算成交量比率
            if vol_ma and vol_ma > 0:
                current_vol = float(volumes[-1])
                indicators.volume_ratio = current_vol / vol_ma

        # 价格序列
        indicators.prices_series = _get_series(closes, series_length)

        return indicators


def _get_series(arr: np.ndarray, length: int) -> Optional[List[float]]:
    """获取序列数据（最后N个非NaN值）"""
    valid = [float(x) for x in arr if not np.isnan(x)]
    if not valid:
        return None
    return valid[-length:] if len(valid) >= length else valid


# 便捷函数
def calculate_mtf_indicators(
    ohlcv_1h: Optional[OHLCVData] = None,
    ohlcv_15m: Optional[OHLCVData] = None,
    ohlcv_5m: Optional[OHLCVData] = None,
    current_price: float = 0.0
) -> Dict[str, Optional[TimeframeIndicators]]:
    """
    一次性计算所有时间框架的指标

    Args:
        ohlcv_1h: 1小时K线数据
        ohlcv_15m: 15分钟K线数据
        ohlcv_5m: 5分钟K线数据
        current_price: 当前价格

    Returns:
        Dict: {"1h": ..., "15m": ..., "5m": ...}
    """
    calculator = MTFCalculator()
    result = {
        "1h": None,
        "15m": None,
        "5m": None,
    }

    if ohlcv_1h:
        result["1h"] = calculator.calculate_1h(ohlcv_1h, current_price)

    if ohlcv_15m:
        result["15m"] = calculator.calculate_15m(ohlcv_15m, current_price)

    if ohlcv_5m:
        result["5m"] = calculator.calculate_5m(ohlcv_5m, current_price)

    return result