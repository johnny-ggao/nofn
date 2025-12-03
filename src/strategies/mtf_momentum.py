"""
多时间框架动量策略 (MTF Momentum Strategy)

策略特点：
- 1H: 趋势确认 (EMA 7/21/55, MACD, RSI, ADX, ATR, BB)
- 15M: 入场时机 (EMA 8/21/50, MACD, RSI, Stochastic, Volume ROC, ATR)
- 5M: 精确入场 (EMA 8, RSI, MACD, Volume MA)
"""
from typing import Dict, Any, Optional

from .base import BaseStrategy, StrategyConfig, TimeframeConfig, IndicatorConfig
from ..utils.mtf_calculator import MTFCalculator, OHLCVData
from ..engine.market_snapshot import TimeframeIndicators


def create_mtf_momentum_config(
    prompt_path: str = "src/prompts/nofn_v2.txt",
    version: str = "2.0.0"
) -> StrategyConfig:
    """
    创建多时间框架动量策略配置

    这是系统默认策略，使用 1H/15M/5M 三个时间框架
    """
    return StrategyConfig(
        name="MTF_Momentum",
        version=version,
        description="多时间框架动量策略：1H趋势确认 + 15M入场时机 + 5M精确入场",
        prompt_path=prompt_path,

        # 时间框架配置
        timeframes=[
            # 1小时级别 - 趋势确认 (权重 40%)
            TimeframeConfig(
                timeframe="1h",
                weight=0.40,
                purpose="趋势确认",
                candle_limit=200,
                indicators=[
                    IndicatorConfig(name="ema", params={"periods": [7, 21, 55]}),
                    IndicatorConfig(name="macd", params={"fast": 6, "slow": 13, "signal": 5}),
                    IndicatorConfig(name="rsi", params={"period": 14}),
                    IndicatorConfig(name="adx", params={"period": 14}),
                    IndicatorConfig(name="atr", params={"period": 14}),
                    IndicatorConfig(name="bollinger", params={"period": 20, "std_dev": 2.0}),
                ]
            ),
            # 15分钟级别 - 入场时机 (权重 35%)
            TimeframeConfig(
                timeframe="15m",
                weight=0.35,
                purpose="入场时机",
                candle_limit=100,
                indicators=[
                    IndicatorConfig(name="ema", params={"periods": [8, 21, 50]}),
                    IndicatorConfig(name="macd", params={"fast": 6, "slow": 13, "signal": 5}),
                    IndicatorConfig(name="rsi", params={"period": 14}),
                    IndicatorConfig(name="stochastic", params={"k_period": 14, "k_smooth": 3, "d_period": 3}),
                    IndicatorConfig(name="volume_roc", params={"period": 5}),
                    IndicatorConfig(name="atr", params={"period": 14}),
                ]
            ),
            # 5分钟级别 - 精确入场 (权重 25%)
            TimeframeConfig(
                timeframe="5m",
                weight=0.25,
                purpose="精确入场",
                candle_limit=100,
                indicators=[
                    IndicatorConfig(name="ema", params={"periods": [8]}),
                    IndicatorConfig(name="rsi", params={"period": 9}),
                    IndicatorConfig(name="macd", params={"fast": 5, "slow": 10, "signal": 3}),
                    IndicatorConfig(name="volume_ma", params={"period": 5}),
                ]
            ),
        ],

        # 风控参数
        min_confidence=60,
        min_risk_reward_ratio=2.0,
        max_position_percent=40.0,
        max_leverage=20,
        default_risk_percent=2.0,
    )


class MTFMomentumStrategy(BaseStrategy):
    """
    多时间框架动量策略

    核心逻辑：
    1. 1H级别确认趋势方向（EMA排列 + ADX趋势强度）
    2. 15M级别等待入场时机（MACD + RSI + Stochastic）
    3. 5M级别确定精确入场点（EMA穿越 + 成交量确认）
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        初始化策略

        Args:
            config: 策略配置，如果不提供则使用默认配置
        """
        if config is None:
            config = create_mtf_momentum_config()
        super().__init__(config)
        self._calculator = MTFCalculator()

    def get_indicator_calculator(self) -> MTFCalculator:
        """获取指标计算器"""
        return self._calculator

    def calculate_indicators(
        self,
        ohlcv_data: Dict[str, OHLCVData],
        current_price: float
    ) -> Dict[str, TimeframeIndicators]:
        """
        根据策略配置计算所有时间框架的指标

        Args:
            ohlcv_data: K线数据，格式 {"1h": OHLCVData, "15m": OHLCVData, "5m": OHLCVData}
            current_price: 当前价格

        Returns:
            各时间框架的指标数据
        """
        result = {}

        for tf_config in self.config.timeframes:
            tf = tf_config.timeframe
            if tf not in ohlcv_data:
                continue

            data = ohlcv_data[tf]

            # 根据时间框架调用对应的计算方法
            if tf == "1h":
                result[tf] = self._calculator.calculate_1h(data, current_price)
            elif tf == "15m":
                result[tf] = self._calculator.calculate_15m(data, current_price)
            elif tf == "5m":
                result[tf] = self._calculator.calculate_5m(data, current_price)

        return result

    def get_ema_params(self, timeframe: str) -> list:
        """获取指定时间框架的 EMA 参数"""
        tf_config = self.config.get_timeframe(timeframe)
        if tf_config:
            ema_config = tf_config.get_indicator("ema")
            if ema_config:
                return ema_config.params.get("periods", [])
        return []

    def get_macd_params(self, timeframe: str) -> Dict[str, int]:
        """获取指定时间框架的 MACD 参数"""
        tf_config = self.config.get_timeframe(timeframe)
        if tf_config:
            macd_config = tf_config.get_indicator("macd")
            if macd_config:
                return {
                    "fast": macd_config.params.get("fast", 12),
                    "slow": macd_config.params.get("slow", 26),
                    "signal": macd_config.params.get("signal", 9),
                }
        return {"fast": 12, "slow": 26, "signal": 9}