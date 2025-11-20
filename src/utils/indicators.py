"""
量化指标工具模块

提供常用的技术指标计算方法，包括：
- SMA (简单移动平均)
- EMA (指数移动平均)
- RSI (相对强弱指标)
- MACD (指数平滑异同移动平均线)
- ATR (平均真实波幅)
- Bollinger Bands (布林带)
- 等其他常用指标
"""
from typing import List, Tuple, Optional, Union
from decimal import Decimal
import numpy as np
from numpy.typing import NDArray


def to_float_array(data: Union[List[float], List[Decimal], NDArray]) -> NDArray:
    """
    将数据转换为 numpy 浮点数组

    Args:
        data: 输入数据（可以是列表或 numpy 数组）

    Returns:
        NDArray: numpy 浮点数组
    """
    if isinstance(data, np.ndarray):
        return data.astype(float)
    return np.array([float(x) for x in data], dtype=float)


def sma(data: Union[List[float], List[Decimal], NDArray], period: int) -> NDArray:
    """
    简单移动平均 (Simple Moving Average)

    Args:
        data: 价格数据
        period: 周期

    Returns:
        NDArray: SMA 值数组

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106]
        >>> sma_values = sma(prices, period=5)
    """
    arr = to_float_array(data)

    if len(arr) < period:
        raise ValueError(f"数据长度 {len(arr)} 小于周期 {period}")

    # 使用卷积计算 SMA
    weights = np.ones(period) / period
    sma_values = np.convolve(arr, weights, mode='valid')

    # 前面的值用 NaN 填充
    result = np.full(len(arr), np.nan)
    result[period-1:] = sma_values

    return result


def ema(data: Union[List[float], List[Decimal], NDArray], period: int) -> NDArray:
    """
    指数移动平均 (Exponential Moving Average)

    Args:
        data: 价格数据
        period: 周期

    Returns:
        NDArray: EMA 值数组

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106]
        >>> ema_values = ema(prices, period=5)
    """
    arr = to_float_array(data)

    if len(arr) < period:
        raise ValueError(f"数据长度 {len(arr)} 小于周期 {period}")

    # EMA 计算公式: EMA = (Close - EMA_prev) * multiplier + EMA_prev
    # multiplier = 2 / (period + 1)
    multiplier = 2.0 / (period + 1)

    ema_values = np.full(len(arr), np.nan)

    # 第一个 EMA 值使用 SMA
    ema_values[period-1] = np.mean(arr[:period])

    # 计算后续的 EMA
    for i in range(period, len(arr)):
        ema_values[i] = (arr[i] - ema_values[i-1]) * multiplier + ema_values[i-1]

    return ema_values


def rsi(data: Union[List[float], List[Decimal], NDArray], period: int = 14) -> NDArray:
    """
    相对强弱指标 (Relative Strength Index)

    RSI = 100 - (100 / (1 + RS))
    RS = 平均涨幅 / 平均跌幅

    Args:
        data: 价格数据
        period: 周期，默认 14

    Returns:
        NDArray: RSI 值数组 (0-100)

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        >>> rsi_values = rsi(prices, period=14)
    """
    arr = to_float_array(data)

    if len(arr) < period + 1:
        raise ValueError(f"数据长度 {len(arr)} 不足以计算 RSI (需要至少 {period + 1} 个数据点)")

    # 计算价格变化
    deltas = np.diff(arr)

    # 分离涨跌
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # 初始化结果数组
    rsi_values = np.full(len(arr), np.nan)

    # 计算第一个 RS (使用简单平均)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

    # 使用平滑方法计算后续的 RSI
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values


def macd(
    data: Union[List[float], List[Decimal], NDArray],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    指数平滑异同移动平均线 (Moving Average Convergence Divergence)

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        data: 价格数据
        fast_period: 快线周期，默认 12
        slow_period: 慢线周期，默认 26
        signal_period: 信号线周期，默认 9

    Returns:
        Tuple[NDArray, NDArray, NDArray]: (MACD线, 信号线, 柱状图)

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106, ...]
        >>> macd_line, signal_line, histogram = macd(prices)
    """
    arr = to_float_array(data)

    min_length = slow_period + signal_period
    if len(arr) < min_length:
        raise ValueError(f"数据长度 {len(arr)} 不足以计算 MACD (需要至少 {min_length} 个数据点)")

    # 计算快线和慢线的 EMA
    ema_fast = ema(arr, fast_period)
    ema_slow = ema(arr, slow_period)

    # 计算 MACD 线
    macd_line = ema_fast - ema_slow

    # 计算信号线 (MACD 的 EMA)
    # 需要跳过 NaN 值
    valid_start = slow_period - 1
    signal_line = np.full(len(arr), np.nan)

    if valid_start + signal_period <= len(arr):
        macd_valid = macd_line[valid_start:]
        signal_values = ema(macd_valid, signal_period)
        signal_line[valid_start:] = signal_values

    # 计算柱状图
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def atr(
    high: Union[List[float], List[Decimal], NDArray],
    low: Union[List[float], List[Decimal], NDArray],
    close: Union[List[float], List[Decimal], NDArray],
    period: int = 14
) -> NDArray:
    """
    平均真实波幅 (Average True Range)

    TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
    ATR = EMA(TR, period) 或 SMA(TR, period)

    Args:
        high: 最高价数据
        low: 最低价数据
        close: 收盘价数据
        period: 周期，默认 14

    Returns:
        NDArray: ATR 值数组

    Example:
        >>> highs = [105, 107, 106, 108, ...]
        >>> lows = [100, 102, 101, 103, ...]
        >>> closes = [102, 104, 103, 105, ...]
        >>> atr_values = atr(highs, lows, closes, period=14)
    """
    high_arr = to_float_array(high)
    low_arr = to_float_array(low)
    close_arr = to_float_array(close)

    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("high, low, close 数组长度必须相同")

    if len(high_arr) < period + 1:
        raise ValueError(f"数据长度 {len(high_arr)} 不足以计算 ATR (需要至少 {period + 1} 个数据点)")

    # 计算真实波幅 (True Range)
    tr = np.full(len(high_arr), np.nan)

    # 第一个 TR 只考虑 high - low
    tr[0] = high_arr[0] - low_arr[0]

    # 后续的 TR 考虑三个值的最大值
    for i in range(1, len(high_arr)):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close_arr[i-1])
        lc = abs(low_arr[i] - close_arr[i-1])
        tr[i] = max(hl, hc, lc)

    # 计算 ATR (使用 EMA 平滑)
    atr_values = np.full(len(high_arr), np.nan)

    # 第一个 ATR 使用 SMA
    atr_values[period] = np.mean(tr[1:period+1])

    # 后续使用 EMA 平滑
    for i in range(period + 1, len(high_arr)):
        atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period

    return atr_values


def bollinger_bands(
    data: Union[List[float], List[Decimal], NDArray],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    布林带 (Bollinger Bands)

    中轨 = SMA(period)
    上轨 = 中轨 + (标准差 * std_dev)
    下轨 = 中轨 - (标准差 * std_dev)

    Args:
        data: 价格数据
        period: 周期，默认 20
        std_dev: 标准差倍数，默认 2.0

    Returns:
        Tuple[NDArray, NDArray, NDArray]: (上轨, 中轨, 下轨)

    Example:
        >>> prices = [100, 102, 101, 103, 105, 104, 106, ...]
        >>> upper, middle, lower = bollinger_bands(prices, period=20, std_dev=2.0)
    """
    arr = to_float_array(data)

    if len(arr) < period:
        raise ValueError(f"数据长度 {len(arr)} 小于周期 {period}")

    # 计算中轨 (SMA)
    middle_band = sma(arr, period)

    # 计算标准差
    std = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        std[i] = np.std(arr[i - period + 1:i + 1], ddof=0)

    # 计算上下轨
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


def stochastic_oscillator(
    high: Union[List[float], List[Decimal], NDArray],
    low: Union[List[float], List[Decimal], NDArray],
    close: Union[List[float], List[Decimal], NDArray],
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[NDArray, NDArray]:
    """
    随机震荡指标 (Stochastic Oscillator)

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K, d_period)

    Args:
        high: 最高价数据
        low: 最低价数据
        close: 收盘价数据
        k_period: %K 周期，默认 14
        d_period: %D 周期，默认 3

    Returns:
        Tuple[NDArray, NDArray]: (%K, %D)

    Example:
        >>> highs = [105, 107, 106, 108, ...]
        >>> lows = [100, 102, 101, 103, ...]
        >>> closes = [102, 104, 103, 105, ...]
        >>> k, d = stochastic_oscillator(highs, lows, closes)
    """
    high_arr = to_float_array(high)
    low_arr = to_float_array(low)
    close_arr = to_float_array(close)

    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("high, low, close 数组长度必须相同")

    if len(high_arr) < k_period:
        raise ValueError(f"数据长度 {len(high_arr)} 小于 K 周期 {k_period}")

    # 计算 %K
    k_values = np.full(len(close_arr), np.nan)

    for i in range(k_period - 1, len(close_arr)):
        lowest_low = np.min(low_arr[i - k_period + 1:i + 1])
        highest_high = np.max(high_arr[i - k_period + 1:i + 1])

        if highest_high - lowest_low == 0:
            k_values[i] = 50.0  # 避免除以零
        else:
            k_values[i] = (close_arr[i] - lowest_low) / (highest_high - lowest_low) * 100.0

    # 计算 %D (对 %K 进行 SMA 平滑)
    d_values = np.full(len(close_arr), np.nan)
    valid_start = k_period - 1

    if valid_start + d_period <= len(close_arr):
        k_valid = k_values[valid_start:]
        d_result = sma(k_valid, d_period)
        d_values[valid_start:] = d_result

    return k_values, d_values


def obv(
    close: Union[List[float], List[Decimal], NDArray],
    volume: Union[List[float], List[Decimal], NDArray]
) -> NDArray:
    """
    能量潮指标 (On Balance Volume)

    如果今日收盘价 > 昨日收盘价，则 OBV = 昨日OBV + 今日成交量
    如果今日收盘价 < 昨日收盘价，则 OBV = 昨日OBV - 今日成交量
    如果今日收盘价 = 昨日收盘价，则 OBV = 昨日OBV

    Args:
        close: 收盘价数据
        volume: 成交量数据

    Returns:
        NDArray: OBV 值数组

    Example:
        >>> closes = [100, 102, 101, 103, 105, ...]
        >>> volumes = [1000, 1200, 900, 1500, 1300, ...]
        >>> obv_values = obv(closes, volumes)
    """
    close_arr = to_float_array(close)
    volume_arr = to_float_array(volume)

    if len(close_arr) != len(volume_arr):
        raise ValueError("close 和 volume 数组长度必须相同")

    if len(close_arr) < 2:
        raise ValueError("数据长度不足")

    obv_values = np.zeros(len(close_arr))
    obv_values[0] = volume_arr[0]  # 第一个值设为第一天的成交量

    for i in range(1, len(close_arr)):
        if close_arr[i] > close_arr[i-1]:
            obv_values[i] = obv_values[i-1] + volume_arr[i]
        elif close_arr[i] < close_arr[i-1]:
            obv_values[i] = obv_values[i-1] - volume_arr[i]
        else:
            obv_values[i] = obv_values[i-1]

    return obv_values


def vwap(
    high: Union[List[float], List[Decimal], NDArray],
    low: Union[List[float], List[Decimal], NDArray],
    close: Union[List[float], List[Decimal], NDArray],
    volume: Union[List[float], List[Decimal], NDArray]
) -> NDArray:
    """
    成交量加权平均价 (Volume Weighted Average Price)

    VWAP = Σ(典型价格 × 成交量) / Σ成交量
    典型价格 = (High + Low + Close) / 3

    Args:
        high: 最高价数据
        low: 最低价数据
        close: 收盘价数据
        volume: 成交量数据

    Returns:
        NDArray: VWAP 值数组

    Example:
        >>> highs = [105, 107, 106, 108, ...]
        >>> lows = [100, 102, 101, 103, ...]
        >>> closes = [102, 104, 103, 105, ...]
        >>> volumes = [1000, 1200, 900, 1500, ...]
        >>> vwap_values = vwap(highs, lows, closes, volumes)
    """
    high_arr = to_float_array(high)
    low_arr = to_float_array(low)
    close_arr = to_float_array(close)
    volume_arr = to_float_array(volume)

    if not (len(high_arr) == len(low_arr) == len(close_arr) == len(volume_arr)):
        raise ValueError("所有数组长度必须相同")

    # 计算典型价格
    typical_price = (high_arr + low_arr + close_arr) / 3.0

    # 计算 VWAP
    vwap_values = np.zeros(len(close_arr))
    cumulative_tp_volume = 0.0
    cumulative_volume = 0.0

    for i in range(len(close_arr)):
        cumulative_tp_volume += typical_price[i] * volume_arr[i]
        cumulative_volume += volume_arr[i]

        if cumulative_volume > 0:
            vwap_values[i] = cumulative_tp_volume / cumulative_volume
        else:
            vwap_values[i] = typical_price[i]

    return vwap_values


# ========== 辅助函数 ==========

def crossover(series1: NDArray, series2: NDArray) -> NDArray:
    """
    检测序列1是否上穿序列2

    Args:
        series1: 序列1
        series2: 序列2

    Returns:
        NDArray: 布尔数组，True 表示发生上穿
    """
    above = series1 > series2
    return np.append([False], above[1:] & ~above[:-1])


def crossunder(series1: NDArray, series2: NDArray) -> NDArray:
    """
    检测序列1是否下穿序列2

    Args:
        series1: 序列1
        series2: 序列2

    Returns:
        NDArray: 布尔数组，True 表示发生下穿
    """
    below = series1 < series2
    return np.append([False], below[1:] & ~below[:-1])
