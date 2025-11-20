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

    # 成交量
    obv: Optional[float] = None
    volume_trend: Optional[str] = None

    # 震荡指标
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None

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
            'obv': self.obv,
            'volume_trend': self.volume_trend,
            'stochastic': {
                'k': self.stoch_k,
                'd': self.stoch_d,
            }
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
    indicators: IndicatorData = field(default_factory=IndicatorData)

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
        """转换为文本（供LLM阅读）"""
        lines = [
            f"## {self.symbol}",
            f"",
            f"### 价格信息",
            f"- 当前价: ${float(self.current_price):.2f}",
        ]

        if self.change_24h_percent is not None:
            change_emoji = "📈" if self.change_24h_percent > 0 else "📉"
            lines.append(f"- 24小时涨跌: {change_emoji} {self.change_24h_percent:+.2f}%")

        lines.append("")
        lines.append("### 技术指标")

        ind = self.indicators
        if ind.ema20 and ind.ema50:
            trend = "多头排列" if ind.ema20 > ind.ema50 else "空头排列"
            lines.append(f"- EMA趋势: {trend} (EMA20: ${ind.ema20:.2f}, EMA50: ${ind.ema50:.2f})")

        if self.has_position():
            lines.append("")
            lines.append("### 当前持仓")
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
