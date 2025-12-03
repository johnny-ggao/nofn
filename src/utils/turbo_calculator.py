"""
TurboTrader 指标计算器

根据 TurboTrader 策略的三重涡轮确认系统计算指标:
- 1小时: 趋势确认 (EMA 5/20/50, ADX 7, MACD)
- 15分钟: 信号触发 (ATR突破, MACD, RSI 7, BB, 成交量)
- 5分钟: 精确入场 (K线形态, VWAP, 瞬时动量)
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from .indicators import (
    ema, macd, rsi, atr, bollinger_bands,
    adx, vwap, volume_sma, to_float_array
)
from ..engine.market_snapshot import TimeframeIndicators


@dataclass
class OHLCVData:
    """OHLCV K线数据"""
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


class TurboCalculator:
    """
    TurboTrader 指标计算器

    核心特点：
    1. 1H 使用 EMA(5,20,50) + ADX(7) 快速判断趋势
    2. 15M 使用 RSI(7) + ATR通道 + BB 捕捉信号
    3. 5M 使用 VWAP + 成交量比率精确入场

    用法:
        calculator = TurboCalculator()

        # 计算1小时指标 (趋势确认)
        tf_1h = calculator.calculate_1h(ohlcv_1h, current_price)

        # 计算15分钟指标 (信号触发)
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

        指标配置 (三重涡轮第一重):
        - EMA(5, 20, 50): 判断趋势方向和多空排列
          * 多头: EMA(5) > EMA(20) > EMA(50)
          * 空头: EMA(5) < EMA(20) < EMA(50)
        - ADX(7): 快速判断趋势强度 (ADX > 30 为强趋势)
        - +DI / -DI: 判断趋势方向
        - MACD: 判断柱状线是否扩大
        - ATR(14): 衡量市场波动性

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

        # EMA (5, 20, 50) - TurboTrader 特有参数
        if n >= 5:
            ema5_arr = ema(closes, 5)
            indicators.ema5 = float(ema5_arr[-1]) if not np.isnan(ema5_arr[-1]) else None
            indicators.ema5_series = _get_series(ema5_arr, series_length)

            # 计算 EMA 角度 (用于动态仓位计算)
            if len(ema5_arr) >= 5:
                ema_slope = (ema5_arr[-1] - ema5_arr[-5]) / ema5_arr[-5] * 100 if ema5_arr[-5] != 0 else 0
                # 简化角度计算: 斜率 * 放大系数
                indicators.ema_angle = float(np.arctan(ema_slope * 10) * 180 / np.pi)

        if n >= 20:
            ema20_arr = ema(closes, 20)
            indicators.ema20 = float(ema20_arr[-1]) if not np.isnan(ema20_arr[-1]) else None
            indicators.ema20_series = _get_series(ema20_arr, series_length)

        if n >= 50:
            ema50_arr = ema(closes, 50)
            indicators.ema50 = float(ema50_arr[-1]) if not np.isnan(ema50_arr[-1]) else None
            indicators.ema50_series = _get_series(ema50_arr, series_length)

        # ADX (7) - TurboTrader 使用更短周期快速响应
        if n >= 14:  # 需要 2 * period
            adx_arr, plus_di_arr, minus_di_arr = adx(highs, lows, closes, 7)
            indicators.adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else None
            indicators.plus_di = float(plus_di_arr[-1]) if not np.isnan(plus_di_arr[-1]) else None
            indicators.minus_di = float(minus_di_arr[-1]) if not np.isnan(minus_di_arr[-1]) else None

        # MACD (12, 26, 9) - 标准参数
        if n >= 35:
            macd_line, signal_line, histogram = macd(closes, fast_period=12, slow_period=26, signal_period=9)
            indicators.macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
            indicators.macd_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None
            indicators.macd_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
            indicators.macd_series = _get_series(macd_line, series_length)

            # 判断柱状线是否扩大
            if len(histogram) >= 3:
                hist_values = [h for h in histogram[-3:] if not np.isnan(h)]
                if len(hist_values) >= 2:
                    indicators.macd_expanding = abs(hist_values[-1]) > abs(hist_values[-2])

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
    def calculate_15m(
        data: OHLCVData,
        current_price: float,
        series_length: int = 10
    ) -> TimeframeIndicators:
        """
        计算15分钟级别指标 (信号触发)

        指标配置 (三重涡轮第二重，必须满足3/4):
        - ATR突破: 价格 > 上轨(2.0×ATR) 或 < 下轨(2.0×ATR)
        - MACD确认: 快线金叉 + MACD>0 (多头) / 死叉 + MACD<0 (空头)
        - RSI(7): RSI > 68且<82 (多头) / RSI < 32且>18 (空头)
        - BB突破: 突破上轨+带宽扩张>15% (多头) / 跌破下轨+带宽扩张>15% (空头)
        - 成交量: 成交量 > 2倍均量 (多头) / > 1.8倍均量 (空头)

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

        # RSI (7) - TurboTrader 使用更短周期
        if n >= 8:
            rsi_arr = rsi(closes, 7)
            indicators.rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
            indicators.rsi_series = _get_series(rsi_arr, series_length)

        # MACD (12, 26, 9)
        if n >= 35:
            macd_line, signal_line, histogram = macd(closes, fast_period=12, slow_period=26, signal_period=9)
            indicators.macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
            indicators.macd_signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None
            indicators.macd_histogram = float(histogram[-1]) if not np.isnan(histogram[-1]) else None
            indicators.macd_series = _get_series(macd_line, series_length)

            # 检测金叉/死叉
            if len(macd_line) >= 2 and len(signal_line) >= 2:
                prev_macd = macd_line[-2] if not np.isnan(macd_line[-2]) else 0
                prev_signal = signal_line[-2] if not np.isnan(signal_line[-2]) else 0
                curr_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
                curr_signal = signal_line[-1] if not np.isnan(signal_line[-1]) else 0

                indicators.macd_golden_cross = prev_macd <= prev_signal and curr_macd > curr_signal
                indicators.macd_death_cross = prev_macd >= prev_signal and curr_macd < curr_signal

        # ATR (14) - 用于计算ATR通道
        if n >= 15:
            atr_arr = atr(highs, lows, closes, 14)
            atr_value = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else None
            indicators.atr = atr_value
            if atr_value and current_price > 0:
                indicators.atr_percent = (atr_value / current_price) * 100
                # ATR通道 (2.0倍)
                indicators.atr_upper = current_price + atr_value * 2.0
                indicators.atr_lower = current_price - atr_value * 2.0

        # Bollinger Bands (20, 2)
        if n >= 20:
            bb_upper, bb_middle, bb_lower = bollinger_bands(closes, 20, 2.0)
            indicators.bb_upper = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
            indicators.bb_middle = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None
            indicators.bb_lower = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None

            # 计算带宽变化率
            if len(bb_upper) >= 2 and len(bb_lower) >= 2:
                prev_width = bb_upper[-2] - bb_lower[-2] if not np.isnan(bb_upper[-2]) and not np.isnan(bb_lower[-2]) else 0
                curr_width = bb_upper[-1] - bb_lower[-1] if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]) else 0
                if prev_width > 0:
                    indicators.bb_width_change = (curr_width - prev_width) / prev_width * 100

        # Volume MA (20) - 用于计算成交量倍数
        if n >= 20:
            vol_ma_arr = volume_sma(volumes, 20)
            vol_ma = float(vol_ma_arr[-1]) if not np.isnan(vol_ma_arr[-1]) else None
            indicators.volume_ma = vol_ma

            if vol_ma and vol_ma > 0:
                current_vol = float(volumes[-1])
                indicators.volume_ratio = current_vol / vol_ma

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

        指标配置 (三重涡轮第三重，必须满足2/3):
        - K线形态: 涡轮阳线(实体>前5根平均2.5倍), 三阳开泰, 跳空高开
        - VWAP系统: 价格>VWAP且VWAP斜率>0 (多头) / 价格<VWAP且VWAP斜率<0 (空头)
        - 瞬时动量: 连续3根同向K线 + 动量递增 + 成交量配合

        Args:
            data: OHLCV数据
            current_price: 当前价格
            series_length: 序列数据长度

        Returns:
            TimeframeIndicators: 5分钟级别指标
        """
        opens = to_float_array(data.open)
        closes = to_float_array(data.close)
        highs = to_float_array(data.high)
        lows = to_float_array(data.low)
        volumes = to_float_array(data.volume)
        n = len(closes)

        indicators = TimeframeIndicators(timeframe="5m")

        # EMA (8, 13) - 快速均线
        if n >= 8:
            ema8_arr = ema(closes, 8)
            indicators.ema8 = float(ema8_arr[-1]) if not np.isnan(ema8_arr[-1]) else None
            indicators.ema8_series = _get_series(ema8_arr, series_length)

        if n >= 13:
            ema13_arr = ema(closes, 13)
            indicators.ema13 = float(ema13_arr[-1]) if not np.isnan(ema13_arr[-1]) else None

        # RSI (9) - 快速RSI
        if n >= 10:
            rsi_arr = rsi(closes, 9)
            indicators.rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
            indicators.rsi_series = _get_series(rsi_arr, series_length)

        # VWAP
        if n >= 1:
            vwap_arr = vwap(highs, lows, closes, volumes)
            indicators.vwap = float(vwap_arr[-1]) if not np.isnan(vwap_arr[-1]) else None

            # VWAP 斜率
            if len(vwap_arr) >= 3:
                vwap_slope = (vwap_arr[-1] - vwap_arr[-3]) / vwap_arr[-3] * 100 if vwap_arr[-3] != 0 else 0
                indicators.vwap_slope = float(vwap_slope)

        # Volume MA (5) - 短期成交量均线
        if n >= 5:
            vol_ma_arr = volume_sma(volumes, 5)
            vol_ma = float(vol_ma_arr[-1]) if not np.isnan(vol_ma_arr[-1]) else None
            indicators.volume_ma = vol_ma

            if vol_ma and vol_ma > 0:
                current_vol = float(volumes[-1])
                indicators.volume_ratio = current_vol / vol_ma

        # K线形态分析
        if n >= 6:
            # 计算实体大小
            bodies = [abs(closes[i] - opens[i]) for i in range(n)]
            avg_body_5 = np.mean(bodies[-6:-1])  # 前5根平均

            current_body = bodies[-1]

            # 涡轮阳线: 实体 > 前5根平均2.5倍
            indicators.turbo_candle = current_body > avg_body_5 * 2.5 if avg_body_5 > 0 else False

            # 三阳开泰: 连续3根阳线，涨幅递增
            if n >= 3:
                is_bullish = [closes[i] > opens[i] for i in range(-3, 0)]
                gains = [closes[i] - opens[i] for i in range(-3, 0)]
                indicators.three_soldiers = all(is_bullish) and gains[-1] > gains[-2] > gains[-3] > 0

            # 跳空高开: 缺口 > 0.8%且不补
            if n >= 2:
                gap = (opens[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] != 0 else 0
                gap_not_filled = lows[-1] > closes[-2]
                indicators.gap_up = gap > 0.8 and gap_not_filled

        # 瞬时动量分析
        if n >= 3:
            # 连续3根同向K线
            changes = [closes[i] - closes[i-1] for i in range(-2, 1)]
            all_up = all(c > 0 for c in changes)
            all_down = all(c < 0 for c in changes)

            # 动量递增
            momentum_increasing = False
            if all_up:
                momentum_increasing = abs(changes[-1]) > abs(changes[-2]) > abs(changes[-3])
            elif all_down:
                momentum_increasing = abs(changes[-1]) > abs(changes[-2]) > abs(changes[-3])

            # 成交量配合
            vol_increasing = volumes[-1] > volumes[-2] > volumes[-3] if n >= 3 else False

            indicators.momentum_signal = (all_up or all_down) and momentum_increasing and vol_increasing

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
def calculate_turbo_indicators(
    ohlcv_1h: Optional[OHLCVData] = None,
    ohlcv_15m: Optional[OHLCVData] = None,
    ohlcv_5m: Optional[OHLCVData] = None,
    current_price: float = 0.0
) -> Dict[str, Optional[TimeframeIndicators]]:
    """
    一次性计算所有时间框架的 TurboTrader 指标

    Args:
        ohlcv_1h: 1小时K线数据
        ohlcv_15m: 15分钟K线数据
        ohlcv_5m: 5分钟K线数据
        current_price: 当前价格

    Returns:
        Dict: {"1h": ..., "15m": ..., "5m": ...}
    """
    calculator = TurboCalculator()
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