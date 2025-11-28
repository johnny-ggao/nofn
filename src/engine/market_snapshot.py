"""
市场快照数据模型

用于在执行层和决策层之间传递预处理的市场数据

多时间框架分析:
- 4小时: 确认趋势方向
- 1小时: 确认入场时机
- 15分钟: 确认精确入场点
"""
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class Timeframe(str, Enum):
    """时间框架"""
    M15 = "15m"   # 15分钟 - 精确入场
    H1 = "1h"     # 1小时 - 入场时机
    H4 = "4h"     # 4小时 - 趋势确认


@dataclass
class TimeframeIndicators:
    """
    单一时间框架的指标数据

    4小时级别指标:
    - EMA(8, 21, 50, 200): 判断趋势方向和多空排列
    - MACD(6, 13, 5): 识别趋势转折点和动量变化
    - RSI(14): 判断超买(>70)、超卖(<30)状态
    - ADX(14) + DI: 判断趋势强度(ADX>25表示强趋势)
    - ATR(14): 衡量市场波动性，计算止损距离
    - BB(20, 2): 判断价格波动区间和超买超卖

    1小时级别指标:
    - EMA(8, 21, 50)
    - RSI(14)
    - MACD(6, 13, 5)
    - Stochastic(14, 3, 3): 识别超买超卖和背离信号
    - Volume ROC(5): 确认价格变动是否有成交量支撑
    - ATR(14)

    15分钟级别指标:
    - EMA(8): 快速均线
    - RSI(9): 更敏感的参数
    - MACD(5, 10, 3): 超快速参数
    - Volume MA(5): 成交量均线
    """
    timeframe: str = "1h"

    # ========== EMA 移动平均 ==========
    ema8: Optional[float] = None
    ema21: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None

    # ========== MACD ==========
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # ========== RSI ==========
    rsi: Optional[float] = None  # RSI值

    # ========== ADX 趋势强度 ==========
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None

    # ========== ATR 波动率 ==========
    atr: Optional[float] = None
    atr_percent: Optional[float] = None  # ATR百分比 (ATR/Price * 100)

    # ========== 布林带 ==========
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None

    # ========== 随机指标 ==========
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None

    # ========== 成交量指标 ==========
    volume_roc: Optional[float] = None  # 成交量变化率
    volume_ma: Optional[float] = None   # 成交量均线
    volume_ratio: Optional[float] = None  # 当前成交量/均线

    # ========== 持仓量指标 (OI - Open Interest) ==========
    # 仅在4小时级别使用，作为市场情绪指标
    oi_current: Optional[float] = None  # 当前持仓量 (USD)
    oi_change_4h: Optional[float] = None  # 4小时持仓量变化率 (%)
    oi_change_24h: Optional[float] = None  # 24小时持仓量变化率 (%)
    oi_series: Optional[List[float]] = None  # 持仓量序列 (最近N个4小时数据)

    # ========== 序列数据（最近N个点）==========
    prices_series: Optional[List[float]] = None
    ema8_series: Optional[List[float]] = None
    ema21_series: Optional[List[float]] = None
    ema50_series: Optional[List[float]] = None
    macd_series: Optional[List[float]] = None
    rsi_series: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            'timeframe': self.timeframe,
            'ema': {
                'ema8': self.ema8,
                'ema21': self.ema21,
                'ema50': self.ema50,
                'ema200': self.ema200,
            },
            'macd': {
                'value': self.macd_value,
                'signal': self.macd_signal,
                'histogram': self.macd_histogram,
            },
            'rsi': self.rsi,
            'adx': {
                'value': self.adx,
                'plus_di': self.plus_di,
                'minus_di': self.minus_di,
            },
            'atr': {
                'value': self.atr,
                'percent': self.atr_percent,
            },
            'bollinger': {
                'upper': self.bb_upper,
                'middle': self.bb_middle,
                'lower': self.bb_lower,
            },
            'stochastic': {
                'k': self.stoch_k,
                'd': self.stoch_d,
            },
            'volume': {
                'roc': self.volume_roc,
                'ma': self.volume_ma,
                'ratio': self.volume_ratio,
            },
            'open_interest': {
                'current': self.oi_current,
                'change_4h': self.oi_change_4h,
                'change_24h': self.oi_change_24h,
            },
        }

    def get_trend_direction(self) -> str:
        """
        根据EMA排列判断趋势方向

        Returns:
            "bullish" | "bearish" | "neutral"
        """
        if self.ema8 and self.ema21 and self.ema50:
            if self.ema8 > self.ema21 > self.ema50:
                return "bullish"
            elif self.ema8 < self.ema21 < self.ema50:
                return "bearish"
        return "neutral"

    def get_trend_strength(self) -> str:
        """
        根据ADX判断趋势强度

        Returns:
            "strong" | "weak" | "ranging"
        """
        if self.adx:
            if self.adx > 25:
                return "strong"
            elif self.adx < 20:
                return "ranging"
        return "weak"


# 保持向后兼容的别名
@dataclass
class IndicatorData:
    """技术指标数据 (向后兼容)"""
    # 移动平均
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None

    # 动量指标
    rsi14: Optional[float] = None

    # 趋势指标
    macd_value: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # 波动率
    atr14: Optional[float] = None

    # 序列数据（最近N个点，用于展示趋势）
    prices_series: Optional[List[float]] = None  # 价格序列
    ema20_series: Optional[List[float]] = None  # EMA20序列
    macd_series: Optional[List[float]] = None  # MACD序列
    rsi14_series: Optional[List[float]] = None  # RSI14序列

    def to_dict(self) -> dict:
        return {
            'ema20': self.ema20,
            'ema50': self.ema50,
            'ema200': self.ema200,
            'rsi14': self.rsi14,
            'macd': {
                'value': self.macd_value,
                'signal': self.macd_signal,
                'histogram': self.macd_histogram,
            },
            'atr14': self.atr14,
        }


@dataclass
class AssetData:
    """单个资产的完整数据"""
    symbol: str

    # 价格数据
    current_price: Decimal
    mark_price: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None

    # 24小时统计
    volume_24h: Optional[Decimal] = None
    change_24h_percent: Optional[float] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None

    # ========== 多时间框架指标 ==========
    # 4小时级别 - 确认趋势方向
    tf_4h: Optional[TimeframeIndicators] = None
    # 1小时级别 - 确认入场时机
    tf_1h: Optional[TimeframeIndicators] = None
    # 15分钟级别 - 精确入场点
    tf_15m: Optional[TimeframeIndicators] = None

    # 旧版指标 (向后兼容)
    indicators: IndicatorData = field(default_factory=IndicatorData)
    indicators_4h: Optional[IndicatorData] = None

    # 市场情绪指标（永续合约，独立于时间框架）
    funding_rate: Optional[float] = None  # 资金费率
    open_interest: Optional[float] = None  # 持仓量（USD，当前值）

    # 成交量指标
    volume_current: Optional[float] = None  # 当前成交量
    volume_avg: Optional[float] = None  # 平均成交量（最近20根K线）

    # 持仓信息
    position_size: Decimal = Decimal('0')
    position_side: Optional[str] = None  # "long", "short", None
    entry_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def has_position(self) -> bool:
        """是否有持仓"""
        return self.position_size > 0

    def get_market_bias(self) -> str:
        """
        综合多时间框架判断市场倾向

        Returns:
            "strong_bullish" | "bullish" | "neutral" | "bearish" | "strong_bearish"
        """
        scores = []

        # 4小时趋势权重最高
        if self.tf_4h:
            direction = self.tf_4h.get_trend_direction()
            strength = self.tf_4h.get_trend_strength()
            if direction == "bullish":
                scores.append(2 if strength == "strong" else 1)
            elif direction == "bearish":
                scores.append(-2 if strength == "strong" else -1)

        # 1小时趋势
        if self.tf_1h:
            direction = self.tf_1h.get_trend_direction()
            if direction == "bullish":
                scores.append(1)
            elif direction == "bearish":
                scores.append(-1)

        # 15分钟趋势
        if self.tf_15m:
            direction = self.tf_15m.get_trend_direction()
            if direction == "bullish":
                scores.append(0.5)
            elif direction == "bearish":
                scores.append(-0.5)

        if not scores:
            return "neutral"

        total = sum(scores)
        if total >= 3:
            return "strong_bullish"
        elif total >= 1:
            return "bullish"
        elif total <= -3:
            return "strong_bearish"
        elif total <= -1:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'price': {
                'current': float(self.current_price),
                'mark': float(self.mark_price) if self.mark_price else None,
                'bid': float(self.bid) if self.bid else None,
                'ask': float(self.ask) if self.ask else None,
            },
            'stats_24h': {
                'volume': float(self.volume_24h) if self.volume_24h else None,
                'change_percent': self.change_24h_percent,
                'high': float(self.high_24h) if self.high_24h else None,
                'low': float(self.low_24h) if self.low_24h else None,
            },
            # 多时间框架指标
            'timeframes': {
                '4h': self.tf_4h.to_dict() if self.tf_4h else None,
                '1h': self.tf_1h.to_dict() if self.tf_1h else None,
                '15m': self.tf_15m.to_dict() if self.tf_15m else None,
            },
            'market_bias': self.get_market_bias(),
            # 旧版指标 (向后兼容)
            'indicators': self.indicators.to_dict(),
            'indicators_4h': self.indicators_4h.to_dict() if self.indicators_4h else None,
            'market_sentiment': {
                'funding_rate': self.funding_rate,
                'open_interest': self.open_interest,
            },
            'position': {
                'size': float(self.position_size),
                'side': self.position_side,
                'entry_price': float(self.entry_price) if self.entry_price else None,
                'unrealized_pnl': float(self.unrealized_pnl) if self.unrealized_pnl else None,
                'stop_loss': float(self.stop_loss) if self.stop_loss else None,
                'take_profit': float(self.take_profit) if self.take_profit else None,
            },
            'timestamp': self.timestamp.isoformat(),
        }

    def to_text(self) -> str:
        """转换为文本（供LLM阅读）- 多时间框架格式"""
        lines = [f"## {self.symbol}", ""]

        # ========== 市场概况 ==========
        lines.append("**市场概况:**")
        lines.append(f"- 当前价格: ${float(self.current_price):.2f}")
        bias = self.get_market_bias()
        bias_emoji = {"strong_bullish": "🟢🟢", "bullish": "🟢", "neutral": "⚪",
                      "bearish": "🔴", "strong_bearish": "🔴🔴"}.get(bias, "⚪")
        lines.append(f"- 市场倾向: {bias_emoji} {bias.replace('_', ' ').title()}")
        lines.append("")

        # ========== 永续合约指标 ==========
        if self.open_interest is not None or self.funding_rate is not None:
            lines.append("**永续合约指标:**")
            if self.open_interest is not None:
                oi_m = self.open_interest / 1_000_000
                lines.append(f"- 持仓量: ${oi_m:.2f}M")
            if self.funding_rate is not None:
                fr_percent = self.funding_rate * 100
                lines.append(f"- 资金费率: {fr_percent:+.4f}%")
            lines.append("")

        # ========== 4小时级别 - 趋势确认 ==========
        if self.tf_4h:
            lines.append("**4小时级别 (趋势确认):**")
            tf = self.tf_4h
            trend = tf.get_trend_direction()
            strength = tf.get_trend_strength()
            lines.append(f"- 趋势: {trend} ({strength})")

            # EMA 排列
            ema_parts = []
            if tf.ema8: ema_parts.append(f"EMA8=${tf.ema8:.2f}")
            if tf.ema21: ema_parts.append(f"EMA21=${tf.ema21:.2f}")
            if tf.ema50: ema_parts.append(f"EMA50=${tf.ema50:.2f}")
            if tf.ema200: ema_parts.append(f"EMA200=${tf.ema200:.2f}")
            if ema_parts:
                lines.append(f"- EMA: {' > '.join(ema_parts)}")

            # MACD
            if tf.macd_value is not None:
                macd_signal = "多头" if tf.macd_histogram and tf.macd_histogram > 0 else "空头"
                lines.append(f"- MACD: {tf.macd_value:.2f} (Signal: {tf.macd_signal:.2f}) [{macd_signal}]")

            # RSI
            if tf.rsi is not None:
                rsi_status = "超买" if tf.rsi > 70 else ("超卖" if tf.rsi < 30 else "中性")
                lines.append(f"- RSI(14): {tf.rsi:.1f} [{rsi_status}]")

            # ADX
            if tf.adx is not None:
                lines.append(f"- ADX: {tf.adx:.1f} (+DI={tf.plus_di:.1f}, -DI={tf.minus_di:.1f})")

            # ATR
            if tf.atr is not None:
                lines.append(f"- ATR(14): ${tf.atr:.2f} ({tf.atr_percent:.2f}%)")

            # 布林带
            if tf.bb_upper and tf.bb_lower:
                price = float(self.current_price)
                bb_pos = "上轨" if price > tf.bb_upper else ("下轨" if price < tf.bb_lower else "中轨")
                lines.append(f"- 布林带: 上${tf.bb_upper:.2f} / 中${tf.bb_middle:.2f} / 下${tf.bb_lower:.2f} [价格在{bb_pos}]")

            lines.append("")

        # ========== 1小时级别 - 入场时机 ==========
        if self.tf_1h:
            lines.append("**1小时级别 (入场时机):**")
            tf = self.tf_1h

            # EMA
            ema_parts = []
            if tf.ema8: ema_parts.append(f"EMA8=${tf.ema8:.2f}")
            if tf.ema21: ema_parts.append(f"EMA21=${tf.ema21:.2f}")
            if tf.ema50: ema_parts.append(f"EMA50=${tf.ema50:.2f}")
            if ema_parts:
                lines.append(f"- EMA: {' > '.join(ema_parts)}")

            # MACD
            if tf.macd_value is not None:
                macd_signal = "多头" if tf.macd_histogram and tf.macd_histogram > 0 else "空头"
                lines.append(f"- MACD(6,13,5): {tf.macd_value:.2f} [{macd_signal}]")

            # RSI
            if tf.rsi is not None:
                lines.append(f"- RSI(14): {tf.rsi:.1f}")

            # Stochastic
            if tf.stoch_k is not None:
                stoch_status = "超买" if tf.stoch_k > 80 else ("超卖" if tf.stoch_k < 20 else "中性")
                lines.append(f"- Stochastic: %K={tf.stoch_k:.1f}, %D={tf.stoch_d:.1f} [{stoch_status}]")

            # Volume ROC
            if tf.volume_roc is not None:
                vol_trend = "放量" if tf.volume_roc > 20 else ("缩量" if tf.volume_roc < -20 else "正常")
                lines.append(f"- Volume ROC: {tf.volume_roc:+.1f}% [{vol_trend}]")

            # ATR
            if tf.atr is not None:
                lines.append(f"- ATR(14): ${tf.atr:.2f}")

            lines.append("")

        # ========== 15分钟级别 - 精确入场 ==========
        if self.tf_15m:
            lines.append("**15分钟级别 (精确入场):**")
            tf = self.tf_15m

            # EMA8 快速均线
            if tf.ema8:
                price = float(self.current_price)
                pos = "上方" if price > tf.ema8 else "下方"
                lines.append(f"- EMA(8): ${tf.ema8:.2f} [价格在{pos}]")

            # RSI (更敏感)
            if tf.rsi is not None:
                lines.append(f"- RSI(9): {tf.rsi:.1f}")

            # 快速 MACD
            if tf.macd_value is not None:
                macd_signal = "多头" if tf.macd_histogram and tf.macd_histogram > 0 else "空头"
                lines.append(f"- MACD(5,10,3): {tf.macd_value:.4f} [{macd_signal}]")

            # Volume MA
            if tf.volume_ma is not None and tf.volume_ratio is not None:
                vol_status = "放量" if tf.volume_ratio > 1.5 else ("缩量" if tf.volume_ratio < 0.5 else "正常")
                lines.append(f"- 成交量: {tf.volume_ratio:.2f}x 均量 [{vol_status}]")

            lines.append("")

        # ========== 当前持仓 ==========
        if self.has_position():
            lines.append("**当前持仓:**")
            lines.append(f"- 方向: {self.position_side.upper()}")
            lines.append(f"- 数量: {float(self.position_size)}")
            lines.append(f"- 入场价: ${float(self.entry_price):.2f}")
            if self.unrealized_pnl:
                pnl_emoji = "🟢" if self.unrealized_pnl > 0 else "🔴"
                lines.append(f"- 浮动盈亏: {pnl_emoji} ${float(self.unrealized_pnl):.2f}")
            if self.stop_loss:
                lines.append(f"- 止损: ${float(self.stop_loss):.2f}")
            if self.take_profit:
                lines.append(f"- 止盈: ${float(self.take_profit):.2f}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class MarketSnapshot:
    """市场快照 - 包含所有监控资产的完整数据"""
    assets: Dict[str, AssetData]

    # 账户信息
    account_balance: Decimal = Decimal('0')
    account_available: Decimal = Decimal('0')
    total_position_value: Decimal = Decimal('0')
    total_unrealized_pnl: Decimal = Decimal('0')

    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    def get_asset(self, symbol: str) -> Optional[AssetData]:
        """获取指定资产数据"""
        return self.assets.get(symbol)

    def get_positions(self) -> List[AssetData]:
        """获取所有有持仓的资产"""
        return [asset for asset in self.assets.values() if asset.has_position()]

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'assets': {symbol: asset.to_dict() for symbol, asset in self.assets.items()},
            'account': {
                'balance': float(self.account_balance),
                'available': float(self.account_available),
                'position_value': float(self.total_position_value),
                'unrealized_pnl': float(self.total_unrealized_pnl),
            },
            'timestamp': self.timestamp.isoformat(),
        }

    def to_text(self) -> str:
        """转换为文本（供LLM阅读）"""
        lines = [
            "# 市场快照",
            f"时间: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 账户状态",
            f"- 总资产: ${float(self.account_balance):.2f}",
            f"- 可用资金: ${float(self.account_available):.2f}",
            f"- 持仓市值: ${float(self.total_position_value):.2f}",
        ]

        if self.total_unrealized_pnl != 0:
            pnl_emoji = "🟢" if self.total_unrealized_pnl > 0 else "🔴"
            lines.append(f"- 浮动盈亏: {pnl_emoji} ${float(self.total_unrealized_pnl):.2f}")

        lines.append("")
        lines.append("## 资产数据")
        lines.append("")

        for symbol in sorted(self.assets.keys()):
            asset = self.assets[symbol]
            lines.append(asset.to_text())
            lines.append("")

        return "\n".join(lines)