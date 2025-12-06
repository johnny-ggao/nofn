"""反思模式数据模型。

定义反思分析的输入输出结构。
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ReflectionTrigger(str, Enum):
    """反思触发原因。"""

    SHARPE_CRITICAL = "sharpe_critical"      # 夏普比严重恶化 (< -0.5)
    SHARPE_WARNING = "sharpe_warning"        # 夏普比警告 (-0.5 ~ 0)
    HIGH_DRAWDOWN = "high_drawdown"          # 高回撤 (> 15%)
    LOW_WIN_RATE = "low_win_rate"            # 低胜率 (< 35%)
    OVERTRADING = "overtrading"              # 过度交易
    CONSECUTIVE_LOSSES = "consecutive_losses" # 连续亏损
    POSITION_STUCK = "position_stuck"        # 持仓时间过长
    SCHEDULED = "scheduled"                  # 定期反思


class PerformanceAlert(BaseModel):
    """绩效警报。"""

    trigger: ReflectionTrigger = Field(..., description="触发类型")
    severity: str = Field(default="warning", description="严重程度: info/warning/critical")
    metric_name: str = Field(..., description="相关指标名称")
    current_value: float = Field(..., description="当前值")
    threshold: float = Field(..., description="阈值")
    message: str = Field(..., description="警报消息")


class TradingLesson(BaseModel):
    """交易教训/经验。"""

    category: str = Field(..., description="类别: entry/exit/sizing/timing/selection")
    observation: str = Field(..., description="观察到的现象")
    lesson: str = Field(..., description="总结的教训")
    recommendation: str = Field(..., description="改进建议")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="置信度")


class ReflectionInsight(BaseModel):
    """反思洞察结果。

    包含对近期交易表现的分析、警报和改进建议。
    """

    ts: int = Field(..., description="反思时间戳")

    # 绩效快照
    sharpe_ratio: Optional[float] = Field(default=None, description="当前夏普比")
    sharpe_trend: Optional[str] = Field(default=None, description="夏普比趋势: improving/stable/declining")
    total_trades: int = Field(default=0, description="分析周期内总交易数")
    win_rate: Optional[float] = Field(default=None, description="胜率")
    avg_holding_minutes: Optional[float] = Field(default=None, description="平均持仓时间(分钟)")
    max_drawdown_pct: Optional[float] = Field(default=None, description="最大回撤百分比")

    # 警报
    alerts: List[PerformanceAlert] = Field(default_factory=list, description="绩效警报列表")

    # 教训与建议
    lessons: List[TradingLesson] = Field(default_factory=list, description="交易教训")

    # 行为调整建议
    suggested_min_confidence: Optional[float] = Field(
        default=None, description="建议的最低置信度阈值"
    )
    suggested_max_trades_per_hour: Optional[int] = Field(
        default=None, description="建议的每小时最大交易数"
    )
    suggested_min_holding_minutes: Optional[int] = Field(
        default=None, description="建议的最小持仓时间(分钟)"
    )
    cooldown_cycles: int = Field(
        default=0, description="建议的冷静周期数(选择noop)"
    )

    # 标的级别建议
    symbols_to_avoid: List[str] = Field(
        default_factory=list, description="建议回避的交易对"
    )
    symbols_performing_well: List[str] = Field(
        default_factory=list, description="表现良好的交易对"
    )

    # 整体建议摘要
    summary: str = Field(default="", description="反思总结")

    def has_critical_alerts(self) -> bool:
        """是否有严重警报。"""
        return any(a.severity == "critical" for a in self.alerts)

    def should_pause_trading(self) -> bool:
        """是否应该暂停交易。"""
        return self.cooldown_cycles > 0 or self.has_critical_alerts()

    def to_prompt_context(self) -> str:
        """转换为可嵌入提示词的上下文文本。"""
        lines = ["## 反思分析结果"]

        # 绩效快照
        if self.sharpe_ratio is not None:
            trend_cn = {"improving": "改善中", "stable": "稳定", "declining": "恶化中"}.get(
                self.sharpe_trend, ""
            )
            lines.append(f"- 夏普比: {self.sharpe_ratio:.3f} ({trend_cn})")
        if self.win_rate is not None:
            lines.append(f"- 胜率: {self.win_rate:.1%}")
        if self.avg_holding_minutes is not None:
            lines.append(f"- 平均持仓: {self.avg_holding_minutes:.1f} 分钟")
        if self.max_drawdown_pct is not None:
            lines.append(f"- 最大回撤: {self.max_drawdown_pct:.1%}")

        # 警报
        if self.alerts:
            lines.append("\n### 警报")
            for alert in self.alerts:
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    alert.severity, "⚪"
                )
                lines.append(f"{severity_icon} {alert.message}")

        # 教训
        if self.lessons:
            lines.append("\n### 近期教训")
            for lesson in self.lessons[:3]:  # 最多展示3条
                lines.append(f"- {lesson.lesson}")
                lines.append(f"  建议: {lesson.recommendation}")

        # 行为调整
        adjustments = []
        if self.suggested_min_confidence:
            adjustments.append(f"最低置信度调整为 {self.suggested_min_confidence:.0%}")
        if self.suggested_max_trades_per_hour:
            adjustments.append(f"每小时最多 {self.suggested_max_trades_per_hour} 笔交易")
        if self.suggested_min_holding_minutes:
            adjustments.append(f"最短持仓 {self.suggested_min_holding_minutes} 分钟")
        if self.cooldown_cycles > 0:
            adjustments.append(f"冷静期: 接下来 {self.cooldown_cycles} 个周期建议 noop")

        if adjustments:
            lines.append("\n### 行为调整")
            for adj in adjustments:
                lines.append(f"- {adj}")

        # 标的建议
        if self.symbols_to_avoid:
            lines.append(f"\n### 建议回避: {', '.join(self.symbols_to_avoid)}")
        if self.symbols_performing_well:
            lines.append(f"### 表现良好: {', '.join(self.symbols_performing_well)}")

        # 总结
        if self.summary:
            lines.append(f"\n### 总结\n{self.summary}")

        return "\n".join(lines)
