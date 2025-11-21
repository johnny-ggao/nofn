"""
Market Analyzer - 市场分析器

整合现有的 MarketIndicatorsNode，计算技术指标并提供市场分析
"""
from typing import Dict, Optional, List
from decimal import Decimal
import numpy as np

from ...adapters.base import BaseExchangeAdapter
from ...models.strategy import TimeFrame, MarketIndicators
from ...models.market import Candle
from ...utils.indicators import ema, rsi, macd, atr


class MarketAnalyzer:
    """
    市场分析器

    基于现有的指标计算逻辑，为 Agent 提供技术分析
    """

    def __init__(self, adapter: BaseExchangeAdapter):
        self.adapter = adapter

    async def calculate_indicators(
        self,
        symbol: str,
        candle_data: Dict[str, List[Candle]]
    ) -> Dict[TimeFrame, Dict]:
        """
        计算多时间框架技术指标

        Args:
            symbol: 交易对
            candle_data: K线数据字典 {timeframe: [Kline]}

        Returns:
            Dict[TimeFrame, Dict]: 各时间框架的指标字典
        """
        indicators_map = {}

        # 获取资金费率和持仓量（只需获取一次，所有时间框架共享）
        funding_rate_val = None
        open_interest_val = None

        try:
            # 获取资金费率
            funding_rate_data = await self.adapter.get_funding_rate(symbol)
            if funding_rate_data:
                funding_rate_val = funding_rate_data.funding_rate
        except Exception:
            # 获取失败不影响其他指标
            pass

        try:
            # 获取持仓量
            open_interest_val = await self.adapter.get_open_interest(symbol)
        except Exception:
            # 获取失败不影响其他指标
            pass

        for tf_str, candle in candle_data.items():
            if not candle:
                continue

            try:
                # 转换时间框架
                tf = TimeFrame(tf_str)

                # 计算指标（传入资金费率和持仓量）
                indicators = self._calculate_single_timeframe(
                    candle, tf, funding_rate_val, open_interest_val
                )
                indicators_map[tf] = indicators.model_dump() if indicators else {}

            except Exception as e:
                continue

        return indicators_map

    def _calculate_single_timeframe(
        self,
        candles: List[Candle],
        timeframe: TimeFrame,
        funding_rate: Optional[Decimal] = None,
        open_interest: Optional[Decimal] = None
    ) -> Optional[MarketIndicators]:
        """
        计算单个时间框架的技术指标

        这个方法复用了 market_indicators.py 中的逻辑
        """
        if not candles or len(candles) < 50:
            return None

        closes = np.array([float(k.close) for k in candles])
        highs = np.array([float(k.high) for k in candles])
        lows = np.array([float(k.low) for k in candles])
        volumes = np.array([float(k.volume) for k in candles])

        current_price = Decimal(str(closes[-1]))

        # 1. 趋势指标
        ema_20_arr = ema(closes, 20)
        ema_50_arr = ema(closes, 50)
        ema_200_arr = None
        # EMA 200 需要更多数据点，如果数据足够才计算
        if len(closes) >= 200:
            ema_200_arr = ema(closes, 200)

        # 2. 动量指标
        rsi_7_arr = rsi(closes, 7)
        rsi_14_arr = rsi(closes, 14)
        macd_line, macd_signal_arr, macd_hist = macd(closes, 12, 26, 9)

        # 3. 波动率指标
        atr_arr = atr(highs, lows, closes, 14)

        # 辅助函数：安全转换为 Decimal
        def safe_decimal(arr: np.ndarray) -> Optional[Decimal]:
            if arr is None:
                return None
            val = arr[-1]
            if np.isnan(val):
                return None
            return Decimal(str(round(val, 8)))

        def safe_decimal_array(arr: np.ndarray, n: int = 10) -> Optional[List[Decimal]]:
            if arr is None:
                return None
            # 获取最后 n 个值，反转为新到旧
            last_n = arr[-n:] if len(arr) >= n else arr
            result = []
            for val in reversed(last_n):
                if not np.isnan(val):
                    result.append(Decimal(str(round(val, 8))))
                else:
                    result.append(None)
            return result if result else None

        # 计算 ATR 百分比
        atr_val = safe_decimal(atr_arr)
        atr_percent = None
        if atr_val and current_price > 0:
            atr_percent = (atr_val / current_price) * Decimal("100")

        # 构建 MarketIndicators 对象
        return MarketIndicators(
            timeframe=timeframe,
            current_price=current_price,
            high_24h=Decimal(str(max(highs[-1440:] if len(highs) > 1440 else highs))),
            low_24h=Decimal(str(min(lows[-1440:] if len(lows) > 1440 else lows))),
            # 趋势指标（最近10个点数组）
            ema_20=safe_decimal_array(ema_20_arr, 10),
            ema_50=safe_decimal_array(ema_50_arr, 10),
            ema_200=safe_decimal_array(ema_200_arr, 10) if ema_200_arr is not None else None,
            # 动量指标（最近10个点数组）
            macd_line=safe_decimal_array(macd_line, 10),
            macd_signal=safe_decimal_array(macd_signal_arr, 10),
            macd_histogram=safe_decimal_array(macd_hist, 10),
            rsi_7=safe_decimal_array(rsi_7_arr, 10),
            rsi_14=safe_decimal_array(rsi_14_arr, 10),
            # 波动率指标
            atr=atr_val,
            atr_percent=atr_percent,
            # 成交量指标
            volume_24h=Decimal(str(sum(volumes[-1440:] if len(volumes) > 1440 else volumes))),
            # 市场情绪指标（永续合约特有）
            funding_rate=funding_rate,
            open_interest=open_interest,
        )

    def analyze_trend(self, indicators: MarketIndicators) -> str:
        """
        分析市场趋势

        Args:
            indicators: 技术指标

        Returns:
            str: 趋势描述 (bullish/bearish/neutral)
        """
        if not indicators.ema_20 or not indicators.ema_50:
            return "neutral"

        ema_20_latest = indicators.ema_20[0] if indicators.ema_20[0] is not None else None
        ema_50_latest = indicators.ema_50[0] if indicators.ema_50[0] is not None else None

        if ema_20_latest and ema_50_latest:
            if ema_20_latest > ema_50_latest:
                return "bullish"
            elif ema_20_latest < ema_50_latest:
                return "bearish"

        return "neutral"
