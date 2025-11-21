"""
市场快照数据模型

用于在执行层和决策层之间传递预处理的市场数据
"""
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class IndicatorData:
    """技术指标数据"""
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

    # 技术指标
    indicators: IndicatorData = field(default_factory=IndicatorData)  # 5分钟级别（入场信号）
    indicators_4h: Optional[IndicatorData] = None  # 4小时级别（大趋势判断）

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
        """转换为文本（供LLM阅读）- 优化格式"""
        lines = [f"## {self.symbol}", ""]

        # ========== 当前快照 ==========
        lines.append("**当前快照:**")
        lines.append(f"- 当前价格 = ${float(self.current_price):.2f}")

        ind = self.indicators
        if ind.ema20:
            lines.append(f"- 当前EMA20 = ${ind.ema20:.2f}")
        if ind.macd_value is not None:
            lines.append(f"- 当前MACD = {ind.macd_value:.2f}")
        if ind.rsi14:
            lines.append(f"- 当前RSI(14周期) = {ind.rsi14:.1f}")

        lines.append("")

        # ========== 永续合约指标 ==========
        lines.append("**永续合约指标:**")
        if self.open_interest is not None:
            oi_m = self.open_interest / 1_000_000
            lines.append(f"- 持仓量: ${oi_m:.2f}M")

        if self.funding_rate is not None:
            fr_percent = self.funding_rate * 100
            lines.append(f"- 资金费率: {fr_percent:+.4f}%")

        lines.append("")

        # ========== 日内序列（5分钟级别）==========
        lines.append("**日内序列（5分钟间隔，从旧到新）:**")
        lines.append("")

        if ind.prices_series:
            prices_str = ", ".join([f"{p:.2f}" for p in ind.prices_series])
            lines.append(f"中间价格: [{prices_str}]")
            lines.append("")

        if ind.ema20_series:
            ema20_str = ", ".join([f"{e:.2f}" for e in ind.ema20_series])
            lines.append(f"EMA指标（20周期）: [{ema20_str}]")
            lines.append("")

        if ind.macd_series:
            macd_str = ", ".join([f"{m:.2f}" for m in ind.macd_series])
            lines.append(f"MACD指标: [{macd_str}]")
            lines.append("")

        if ind.rsi14_series:
            rsi_str = ", ".join([f"{r:.1f}" for r in ind.rsi14_series])
            lines.append(f"RSI指标（14周期）: [{rsi_str}]")
            lines.append("")

        # ========== 长期背景（4小时级别）==========
        if self.indicators_4h:
            lines.append("**长期背景（4小时时间框架）:**")
            lines.append("")

            ind_4h = self.indicators_4h
            if ind_4h.ema20 and ind_4h.ema50:
                lines.append(f"20周期EMA: ${ind_4h.ema20:.2f} vs. 50周期EMA: ${ind_4h.ema50:.2f}")
                lines.append("")

            if ind_4h.atr14:
                lines.append(f"14周期ATR: ${ind_4h.atr14:.2f}")
                lines.append("")

            if self.volume_current and self.volume_avg:
                lines.append(f"当前成交量: {self.volume_current:.2f} vs. 平均成交量: {self.volume_avg:.2f}")
                lines.append("")

            if ind_4h.macd_series:
                macd_4h_str = ", ".join([f"{m:.2f}" for m in ind_4h.macd_series])
                lines.append(f"MACD指标（4小时）: [{macd_4h_str}]")
                lines.append("")

            if ind_4h.rsi14_series:
                rsi_4h_str = ", ".join([f"{r:.1f}" for r in ind_4h.rsi14_series])
                lines.append(f"RSI指标（14周期，4小时）: [{rsi_4h_str}]")
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