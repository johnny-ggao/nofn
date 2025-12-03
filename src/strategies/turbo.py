"""
TurboTrader 激进交易策略

策略特点：
- 1H: 趋势确认 (EMA 5/20/50, ADX 7, MACD)
- 15M: 信号触发 (ATR突破, MACD, RSI 7, BB, 成交量)
- 5M: 精确入场 (K线形态, VWAP, 瞬时动量)

风控特点：
- 动态仓位管理（基础40%，最高70%）
- 多层止损系统（硬止损 + 移动止损）
- 四重止盈策略
"""
from typing import Dict, Any, Optional

from .base import BaseStrategy, StrategyConfig, TimeframeConfig, IndicatorConfig
from ..utils.turbo_calculator import TurboCalculator, OHLCVData
from ..engine.market_snapshot import TimeframeIndicators


def create_turbo_config(
    prompt_path: str = "src/prompts/turbo.txt",
    version: str = "1.0.0"
) -> StrategyConfig:
    """
    创建 TurboTrader 策略配置

    激进交易策略，使用 1H/15M/5M 三个时间框架
    核心理念：大周期定方向，中周期找时机，小周期精确入场
    """
    return StrategyConfig(
        name="TurboTrader",
        version=version,
        description="激进交易策略：三重涡轮确认系统，动态仓位管理，多层止损止盈",
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
                    # EMA(5, 20, 50) - 判断趋势方向和多空排列
                    IndicatorConfig(name="ema", params={"periods": [5, 20, 50]}),
                    # ADX(7) - 快速ADX判断趋势强度
                    IndicatorConfig(name="adx", params={"period": 7}),
                    # MACD - 识别趋势转折点
                    IndicatorConfig(name="macd", params={"fast": 12, "slow": 26, "signal": 9}),
                    # ATR(14) - 波动性判断
                    IndicatorConfig(name="atr", params={"period": 14}),
                ]
            ),
            # 15分钟级别 - 信号触发 (权重 35%)
            TimeframeConfig(
                timeframe="15m",
                weight=0.35,
                purpose="信号触发",
                candle_limit=100,
                indicators=[
                    # EMA(8, 21, 50)
                    IndicatorConfig(name="ema", params={"periods": [8, 21, 50]}),
                    # RSI(7) - 快速RSI
                    IndicatorConfig(name="rsi", params={"period": 7}),
                    # MACD
                    IndicatorConfig(name="macd", params={"fast": 12, "slow": 26, "signal": 9}),
                    # Bollinger Bands - 突破判断
                    IndicatorConfig(name="bollinger", params={"period": 20, "std_dev": 2.0}),
                    # ATR(14) - 计算ATR通道
                    IndicatorConfig(name="atr", params={"period": 14}),
                    # Volume - 成交量确认
                    IndicatorConfig(name="volume_ma", params={"period": 20}),
                ]
            ),
            # 5分钟级别 - 精确入场 (权重 25%)
            TimeframeConfig(
                timeframe="5m",
                weight=0.25,
                purpose="精确入场",
                candle_limit=100,
                indicators=[
                    # EMA(8) - 快速均线
                    IndicatorConfig(name="ema", params={"periods": [8, 13]}),
                    # VWAP - 成交量加权价格
                    IndicatorConfig(name="vwap", params={}),
                    # Volume MA - 成交量均线
                    IndicatorConfig(name="volume_ma", params={"period": 5}),
                    # RSI(9) - 快速RSI
                    IndicatorConfig(name="rsi", params={"period": 9}),
                ]
            ),
        ],

        # 风控参数（激进配置）
        min_confidence=60,              # 最小信心度阈值
        min_risk_reward_ratio=1.0,      # 最小盈亏比（激进策略可接受1:1）
        max_position_percent=70.0,      # 单仓最大资金占比（动态可达70%）
        max_leverage=20,                # 最大杠杆
        default_risk_percent=2.0,       # 默认每笔风险
    )


class TurboTraderStrategy(BaseStrategy):
    """
    TurboTrader 激进交易策略

    核心逻辑（三重涡轮确认）：
    1. 1H级别确认趋势方向（EMA 5>20>50 排列 + ADX(7)>30 + MACD柱状线扩大）
    2. 15M级别等待信号触发（ATR突破 + MACD确认 + RSI位置 + BB突破 + 成交量）
    3. 5M级别精确入场（K线形态 + VWAP系统 + 瞬时动量）

    仓位管理：
    - 基础仓位 40%
    - 动态加成（趋势强度、EMA角度、成交量、突破强度、指标共振）
    - 最大单次仓位 70%
    - 金字塔加仓（盈利1%/2.5%/4.5%时加仓）

    风控管理：
    - 动态止损（根据ATR调整：高波动1.8%，中波动1.5%，低波动1.2%）
    - 移动止损（盈利0.5%保本，盈利1.5%开始ATR跟踪）
    - 四重止盈（1:1/1:2/1:3/1:5分批止盈）
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        初始化策略

        Args:
            config: 策略配置，如果不提供则使用默认配置
        """
        if config is None:
            config = create_turbo_config()
        super().__init__(config)
        self._calculator = TurboCalculator()

    def get_indicator_calculator(self) -> TurboCalculator:
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

    @staticmethod
    def calculate_position_size(
        adx: float,
        ema_angle: float,
        volume_ratio: float,
        atr_breakout_ratio: float,
        confirmed_indicators: int,
        pattern_score: float = 0,
        orderflow_score: float = 0,
    ) -> Dict[str, float]:
        """
        计算动态仓位

        Args:
            adx: ADX值
            ema_angle: EMA角度（度）
            volume_ratio: 成交量倍数
            atr_breakout_ratio: 突破幅度/ATR 比率
            confirmed_indicators: 确认的指标数量
            pattern_score: 形态得分 (0-1)
            orderflow_score: 订单流得分 (0-1)

        Returns:
            Dict: {
                "base": 基础仓位,
                "bonus": 总加成,
                "max_position": 最大单次仓位,
                "initial_position": 初始入场仓位
            }
        """
        base_position = 0.40  # 基础仓位 40%

        # 动态加成计算
        bonuses = {
            # 趋势强度：ADX每超过32加3%，上限12%
            "trend_strength": min((adx - 32) / 32 * 0.12, 0.12) if adx > 32 else 0,
            # EMA角度：每5度加2%，上限8%
            "ema_angle": min(abs(ema_angle) / 10 * 0.08, 0.08),
            # 成交量：超过3倍后每0.5倍加2%，上限8%
            "volume": min((volume_ratio - 3) / 0.5 * 0.02, 0.08) if volume_ratio > 3 else 0,
            # 突破强度：每突破0.3倍ATR加2%，上限10%
            "breakout": min(atr_breakout_ratio / 0.3 * 0.02, 0.10),
            # 指标共振：每多一个指标加2%
            "indicator_confluence": confirmed_indicators * 0.02,
            # 执行强度：强形态最高加6%
            "execution": (pattern_score + orderflow_score) * 0.05,
        }

        total_bonus = sum(max(0, v) for v in bonuses.values())
        max_position = min(base_position + total_bonus, 0.70)  # 最大70%
        initial_position = max_position * 0.75  # 初始入场仓位

        return {
            "base": base_position,
            "bonus": total_bonus,
            "bonuses_detail": bonuses,
            "max_position": max_position,
            "initial_position": initial_position,
        }

    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        side: str,
        atr_percent: float,
    ) -> Dict[str, float]:
        """
        计算动态止损

        Args:
            entry_price: 入场价格
            side: 方向 ("long" | "short")
            atr_percent: ATR百分比

        Returns:
            Dict: {
                "stop_loss": 止损价格,
                "stop_percent": 止损百分比,
                "volatility": 波动环境
            }
        """
        # 根据ATR判断波动环境
        if atr_percent > 2.0:
            stop_percent = 0.018  # 高波动：1.8%
            volatility = "high"
        elif atr_percent >= 1.0:
            stop_percent = 0.015  # 中波动：1.5%
            volatility = "medium"
        else:
            stop_percent = 0.012  # 低波动：1.2%
            volatility = "low"

        if side == "long":
            stop_loss = entry_price * (1 - stop_percent)
        else:
            stop_loss = entry_price * (1 + stop_percent)

        return {
            "stop_loss": stop_loss,
            "stop_percent": stop_percent * 100,
            "volatility": volatility,
        }

    @staticmethod
    def calculate_take_profits(
        entry_price: float,
        stop_loss: float,
        side: str,
    ) -> list:
        """
        计算四重止盈目标

        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            side: 方向 ("long" | "short")

        Returns:
            List[Dict]: 止盈目标列表
        """
        risk = abs(entry_price - stop_loss)

        targets = [
            {"ratio": 1.0, "percent": 0.25, "action": "部分了结，锁定利润"},
            {"ratio": 2.0, "percent": 0.35, "action": "继续减仓，降低风险"},
            {"ratio": 3.0, "percent": 0.25, "action": "让利润奔跑"},
            {"ratio": 5.0, "percent": 0.15, "action": "追求超额收益"},
        ]

        result = []
        for target in targets:
            if side == "long":
                tp_price = entry_price + risk * target["ratio"]
            else:
                tp_price = entry_price - risk * target["ratio"]

            result.append({
                "risk_reward": f"1:{target['ratio']:.0f}",
                "price": tp_price,
                "close_percent": target["percent"],
                "action": target["action"],
            })

        return result

    def format_indicators(self, asset_data) -> str:
        """
        TurboTrader 策略的指标格式化

        强调三重涡轮确认系统的指标展示：
        - 1H: EMA(5,20,50), ADX(7), MACD
        - 15M: ATR通道, RSI(7), BB, 成交量
        - 5M: VWAP, K线形态, 瞬时动量
        """
        lines = [f"## {asset_data.symbol}", ""]

        # ========== 市场概况 ==========
        lines.append("**市场概况:**")
        lines.append(f"- 当前价格: ${float(asset_data.current_price):.2f}")
        if asset_data.change_24h_percent is not None:
            change_emoji = "🟢" if asset_data.change_24h_percent >= 0 else "🔴"
            lines.append(f"- 24H涨跌: {change_emoji} {asset_data.change_24h_percent:+.2f}%")
        lines.append("")

        # ========== 永续合约指标 ==========
        if asset_data.open_interest is not None or asset_data.funding_rate is not None:
            lines.append("**永续合约指标:**")
            if asset_data.open_interest is not None:
                oi_m = asset_data.open_interest / 1_000_000
                lines.append(f"- 持仓量: ${oi_m:.2f}M")
            if asset_data.funding_rate is not None:
                fr_percent = asset_data.funding_rate * 100
                lines.append(f"- 资金费率: {fr_percent:+.4f}%")
            lines.append("")

        # ========== 1H 趋势确认（三重涡轮第一重）==========
        if asset_data.tf_1h:
            lines.append("**【1H 趋势确认】**")
            tf = asset_data.tf_1h

            # EMA(5, 20, 50) 排列 - TurboTrader核心
            ema_parts = []
            ema_values = []
            if tf.ema5:
                ema_parts.append(f"EMA5=${tf.ema5:.2f}")
                ema_values.append(tf.ema5)
            if tf.ema20:
                ema_parts.append(f"EMA20=${tf.ema20:.2f}")
                ema_values.append(tf.ema20)
            if tf.ema50:
                ema_parts.append(f"EMA50=${tf.ema50:.2f}")
                ema_values.append(tf.ema50)
            if ema_parts:
                # 判断排列
                if len(ema_values) >= 3:
                    if ema_values[0] > ema_values[1] > ema_values[2]:
                        arrangement = "🟢 多头排列"
                    elif ema_values[0] < ema_values[1] < ema_values[2]:
                        arrangement = "🔴 空头排列"
                    else:
                        arrangement = "⚪ 缠绕"
                else:
                    arrangement = ""
                lines.append(f"- EMA: {' > '.join(ema_parts)} {arrangement}")

            # EMA角度（动态仓位关键指标）
            if tf.ema_angle is not None:
                angle_status = "强势" if abs(tf.ema_angle) > 5 else "温和"
                lines.append(f"- EMA角度: {tf.ema_angle:+.1f}° [{angle_status}]")

            # ADX(7) - 快速趋势强度
            if tf.adx is not None:
                if tf.adx > 30:
                    adx_status = "🟢 强趋势"
                elif tf.adx > 20:
                    adx_status = "⚪ 弱趋势"
                else:
                    adx_status = "🔴 震荡"
                lines.append(f"- ADX(7): {tf.adx:.1f} {adx_status}")
                if tf.plus_di is not None and tf.minus_di is not None:
                    di_bias = "+DI领先" if tf.plus_di > tf.minus_di else "-DI领先"
                    lines.append(f"  (+DI={tf.plus_di:.1f}, -DI={tf.minus_di:.1f}) [{di_bias}]")

            # MACD 转折点识别
            if tf.macd_value is not None:
                macd_bias = "多头" if tf.macd_histogram and tf.macd_histogram > 0 else "空头"
                expanding = ""
                if tf.macd_expanding is not None:
                    expanding = " | 柱状线扩大" if tf.macd_expanding else " | 柱状线收缩"
                lines.append(f"- MACD: {tf.macd_value:.2f} [{macd_bias}{expanding}]")

            # ATR波动率
            if tf.atr_percent is not None:
                if tf.atr_percent > 2.0:
                    vol_env = "高波动"
                elif tf.atr_percent >= 1.0:
                    vol_env = "中波动"
                else:
                    vol_env = "低波动"
                lines.append(f"- ATR: {tf.atr_percent:.2f}% [{vol_env}]")

            lines.append("")

        # ========== 15M 信号触发（三重涡轮第二重）==========
        if asset_data.tf_15m:
            lines.append("**【15M 信号触发】**")
            tf = asset_data.tf_15m

            # ATR通道
            if tf.atr_upper is not None and tf.atr_lower is not None:
                price = float(asset_data.current_price)
                if price > tf.atr_upper:
                    channel_pos = "🟢 突破上轨"
                elif price < tf.atr_lower:
                    channel_pos = "🔴 突破下轨"
                else:
                    channel_pos = "通道内"
                lines.append(f"- ATR通道: 上${tf.atr_upper:.2f} / 下${tf.atr_lower:.2f} [{channel_pos}]")

            # RSI(7) - 快速RSI
            if tf.rsi is not None:
                if tf.rsi > 70:
                    rsi_status = "超买"
                elif tf.rsi < 30:
                    rsi_status = "超卖"
                elif tf.rsi > 50:
                    rsi_status = "多头区"
                else:
                    rsi_status = "空头区"
                lines.append(f"- RSI(7): {tf.rsi:.1f} [{rsi_status}]")

            # MACD
            if tf.macd_value is not None:
                cross = ""
                if tf.macd_golden_cross:
                    cross = " | 🟢 金叉"
                elif tf.macd_death_cross:
                    cross = " | 🔴 死叉"
                macd_bias = "多头" if tf.macd_histogram and tf.macd_histogram > 0 else "空头"
                lines.append(f"- MACD: {macd_bias}{cross}")

            # 布林带
            if tf.bb_upper and tf.bb_lower:
                price = float(asset_data.current_price)
                if price > tf.bb_upper:
                    bb_pos = "🟢 突破上轨"
                elif price < tf.bb_lower:
                    bb_pos = "🔴 突破下轨"
                else:
                    bb_width = (tf.bb_upper - tf.bb_lower) / tf.bb_middle * 100 if tf.bb_middle else 0
                    bb_pos = f"通道内 (宽度{bb_width:.1f}%)"
                lines.append(f"- 布林带: {bb_pos}")
                if tf.bb_width_change is not None:
                    squeeze = "收窄" if tf.bb_width_change < 0 else "扩张"
                    lines.append(f"  带宽变化: {tf.bb_width_change:+.1f}% [{squeeze}]")

            # 成交量确认
            if tf.volume_ratio is not None:
                if tf.volume_ratio > 3:
                    vol_status = "🟢 巨量"
                elif tf.volume_ratio > 1.5:
                    vol_status = "🟢 放量"
                elif tf.volume_ratio < 0.5:
                    vol_status = "🔴 缩量"
                else:
                    vol_status = "正常"
                lines.append(f"- 成交量: {tf.volume_ratio:.2f}x 均量 [{vol_status}]")

            lines.append("")

        # ========== 5M 精确入场（三重涡轮第三重）==========
        if asset_data.tf_5m:
            lines.append("**【5M 精确入场】**")
            tf = asset_data.tf_5m

            # VWAP系统
            if tf.vwap is not None:
                price = float(asset_data.current_price)
                vwap_pos = "🟢 上方" if price > tf.vwap else "🔴 下方"
                lines.append(f"- VWAP: ${tf.vwap:.2f} [价格在{vwap_pos}]")
                if tf.vwap_slope is not None:
                    slope_dir = "向上" if tf.vwap_slope > 0 else "向下"
                    lines.append(f"  VWAP斜率: {tf.vwap_slope:+.4f} [{slope_dir}]")

            # EMA(8, 13)
            ema_parts = []
            if tf.ema8:
                ema_parts.append(f"EMA8=${tf.ema8:.2f}")
            if tf.ema13:
                ema_parts.append(f"EMA13=${tf.ema13:.2f}")
            if ema_parts:
                price = float(asset_data.current_price)
                pos = "价格在上方" if price > (tf.ema8 or tf.ema13) else "价格在下方"
                lines.append(f"- EMA: {', '.join(ema_parts)} [{pos}]")

            # K线形态信号
            patterns = []
            if tf.turbo_candle:
                patterns.append("🟢 涡轮阳线")
            if tf.three_soldiers:
                patterns.append("🟢 三阳开泰")
            if tf.gap_up:
                patterns.append("🟢 跳空高开")
            if tf.momentum_signal:
                patterns.append("🟢 动量信号")
            if patterns:
                lines.append(f"- K线形态: {', '.join(patterns)}")

            # 成交量确认
            if tf.volume_ratio is not None:
                vol_status = "放量" if tf.volume_ratio > 1.5 else ("缩量" if tf.volume_ratio < 0.5 else "正常")
                lines.append(f"- 成交量: {tf.volume_ratio:.2f}x 均量 [{vol_status}]")

            lines.append("")

        # ========== 当前持仓 ==========
        if asset_data.has_position():
            lines.append("**当前持仓:**")
            lines.append(f"- 方向: {asset_data.position_side.upper()}")
            lines.append(f"- 数量: {float(asset_data.position_size)}")
            lines.append(f"- 入场价: ${float(asset_data.entry_price):.2f}")
            if asset_data.unrealized_pnl:
                pnl_emoji = "🟢" if asset_data.unrealized_pnl > 0 else "🔴"
                pnl_percent = float(asset_data.unrealized_pnl) / float(asset_data.entry_price * asset_data.position_size) * 100
                lines.append(f"- 浮动盈亏: {pnl_emoji} ${float(asset_data.unrealized_pnl):.2f} ({pnl_percent:+.2f}%)")
            if asset_data.stop_loss:
                lines.append(f"- 止损: ${float(asset_data.stop_loss):.2f}")
            if asset_data.take_profit:
                lines.append(f"- 止盈: ${float(asset_data.take_profit):.2f}")
            lines.append("")

        return "\n".join(lines)