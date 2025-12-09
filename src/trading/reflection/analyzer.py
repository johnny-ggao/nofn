"""反思分析器。

从历史交易记录中提取教训和改进建议。
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from termcolor import cprint

from ..models import (
    HistoryRecord,
    TradeDigest,
    TradeHistoryEntry,
    get_current_timestamp_ms,
)
from .models import (
    PerformanceAlert,
    ReflectionInsight,
    ReflectionTrigger,
    TradingLesson,
)


class ReflectionAnalyzer:
    """反思分析器。

    分析交易历史，识别问题模式，生成改进建议。
    """

    def __init__(
        self,
        *,
        sharpe_critical_threshold: float = -0.5,
        sharpe_warning_threshold: float = 0.0,
        drawdown_warning_pct: float = 0.10,
        drawdown_critical_pct: float = 0.20,
        win_rate_warning: float = 0.35,
        overtrading_threshold: int = 5,  # 每小时交易数
        consecutive_loss_threshold: int = 3,
        stuck_position_hours: float = 24.0,
    ) -> None:
        """初始化分析器。

        Args:
            sharpe_critical_threshold: 夏普比严重警报阈值
            sharpe_warning_threshold: 夏普比警告阈值
            drawdown_warning_pct: 回撤警告阈值
            drawdown_critical_pct: 回撤严重警报阈值
            win_rate_warning: 胜率警告阈值
            overtrading_threshold: 过度交易阈值(每小时)
            consecutive_loss_threshold: 连续亏损警报阈值
            stuck_position_hours: 持仓过久警报阈值(小时)
        """
        self._sharpe_critical = sharpe_critical_threshold
        self._sharpe_warning = sharpe_warning_threshold
        self._drawdown_warning = drawdown_warning_pct
        self._drawdown_critical = drawdown_critical_pct
        self._win_rate_warning = win_rate_warning
        self._overtrading_threshold = overtrading_threshold
        self._consecutive_loss_threshold = consecutive_loss_threshold
        self._stuck_position_hours = stuck_position_hours

        # 历史夏普比用于趋势分析
        self._sharpe_history: List[Tuple[int, float]] = []

    def analyze(
        self,
        digest: TradeDigest,
        recent_records: List[HistoryRecord],
        previous_insight: Optional[ReflectionInsight] = None,
    ) -> ReflectionInsight:
        """执行反思分析。

        Args:
            digest: 当前交易摘要
            recent_records: 近期历史记录
            previous_insight: 上次反思结果(用于趋势对比)

        Returns:
            ReflectionInsight: 反思洞察
        """
        ts = get_current_timestamp_ms()
        alerts: List[PerformanceAlert] = []
        lessons: List[TradingLesson] = []

        # 提取交易记录
        trades = self._extract_trades(recent_records)

        # 计算基础指标
        sharpe = digest.sharpe_ratio
        win_rate = self._calculate_win_rate(trades)
        avg_holding = self._calculate_avg_holding_minutes(trades)
        max_drawdown = self._calculate_max_drawdown(digest)
        trades_per_hour = self._calculate_trades_per_hour(trades)

        # 更新夏普比历史
        if sharpe is not None:
            self._sharpe_history.append((ts, sharpe))
            # 只保留最近20个数据点
            self._sharpe_history = self._sharpe_history[-20:]

        sharpe_trend = self._analyze_sharpe_trend()

        # === 生成警报 ===

        # 夏普比警报
        if sharpe is not None:
            if sharpe < self._sharpe_critical:
                alerts.append(
                    PerformanceAlert(
                        trigger=ReflectionTrigger.SHARPE_CRITICAL,
                        severity="critical",
                        metric_name="sharpe_ratio",
                        current_value=sharpe,
                        threshold=self._sharpe_critical,
                        message=f"夏普比严重恶化: {sharpe:.3f}，建议暂停交易并反思策略",
                    )
                )
            elif sharpe < self._sharpe_warning:
                alerts.append(
                    PerformanceAlert(
                        trigger=ReflectionTrigger.SHARPE_WARNING,
                        severity="warning",
                        metric_name="sharpe_ratio",
                        current_value=sharpe,
                        threshold=self._sharpe_warning,
                        message=f"夏普比偏低: {sharpe:.3f}，建议收紧入场标准",
                    )
                )

        # 回撤警报
        if max_drawdown is not None:
            if max_drawdown > self._drawdown_critical:
                alerts.append(
                    PerformanceAlert(
                        trigger=ReflectionTrigger.HIGH_DRAWDOWN,
                        severity="critical",
                        metric_name="max_drawdown",
                        current_value=max_drawdown,
                        threshold=self._drawdown_critical,
                        message=f"回撤过大: {max_drawdown:.1%}，风险失控警告",
                    )
                )
            elif max_drawdown > self._drawdown_warning:
                alerts.append(
                    PerformanceAlert(
                        trigger=ReflectionTrigger.HIGH_DRAWDOWN,
                        severity="warning",
                        metric_name="max_drawdown",
                        current_value=max_drawdown,
                        threshold=self._drawdown_warning,
                        message=f"回撤警告: {max_drawdown:.1%}，建议减少仓位",
                    )
                )

        # 胜率警报
        if win_rate is not None and win_rate < self._win_rate_warning:
            alerts.append(
                PerformanceAlert(
                    trigger=ReflectionTrigger.LOW_WIN_RATE,
                    severity="warning",
                    metric_name="win_rate",
                    current_value=win_rate,
                    threshold=self._win_rate_warning,
                    message=f"胜率偏低: {win_rate:.1%}，建议提高信号质量",
                )
            )

        # 过度交易警报
        if trades_per_hour > self._overtrading_threshold:
            alerts.append(
                PerformanceAlert(
                    trigger=ReflectionTrigger.OVERTRADING,
                    severity="warning",
                    metric_name="trades_per_hour",
                    current_value=trades_per_hour,
                    threshold=float(self._overtrading_threshold),
                    message=f"交易过于频繁: {trades_per_hour:.1f}笔/小时，建议降低频率",
                )
            )

        # 连续亏损警报
        consecutive_losses = self._count_consecutive_losses(trades)
        if consecutive_losses >= self._consecutive_loss_threshold:
            alerts.append(
                PerformanceAlert(
                    trigger=ReflectionTrigger.CONSECUTIVE_LOSSES,
                    severity="warning",
                    metric_name="consecutive_losses",
                    current_value=float(consecutive_losses),
                    threshold=float(self._consecutive_loss_threshold),
                    message=f"连续亏损 {consecutive_losses} 笔，建议冷静观望",
                )
            )

        # === 生成教训 ===
        lessons = self._generate_lessons(
            trades=trades,
            sharpe=sharpe,
            win_rate=win_rate,
            avg_holding=avg_holding,
            trades_per_hour=trades_per_hour,
            digest=digest,
        )

        # === 生成行为调整建议 ===
        suggested_confidence = None
        suggested_max_trades = None
        suggested_min_holding = None
        cooldown = 0

        # 基于夏普比调整
        if sharpe is not None:
            if sharpe < self._sharpe_critical:
                cooldown = 6  # 暂停6个周期
                suggested_confidence = 0.90
                suggested_max_trades = 1
                suggested_min_holding = 60
            elif sharpe < self._sharpe_warning:
                suggested_confidence = 0.80
                suggested_max_trades = 2
                suggested_min_holding = 30

        # 基于过度交易调整
        if trades_per_hour > self._overtrading_threshold:
            suggested_max_trades = max(1, self._overtrading_threshold - 1)

        # 基于持仓时间调整
        if avg_holding is not None and avg_holding < 15:
            suggested_min_holding = 20

        # === 标的级别分析 ===
        symbols_to_avoid, symbols_good = self._analyze_symbol_performance(digest)

        # === 生成总结 ===
        summary = self._generate_summary(
            sharpe=sharpe,
            sharpe_trend=sharpe_trend,
            alerts=alerts,
            lessons=lessons,
            cooldown=cooldown,
        )

        return ReflectionInsight(
            ts=ts,
            sharpe_ratio=sharpe,
            sharpe_trend=sharpe_trend,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_holding_minutes=avg_holding,
            max_drawdown_pct=max_drawdown,
            alerts=alerts,
            lessons=lessons,
            suggested_min_confidence=suggested_confidence,
            suggested_max_trades_per_hour=suggested_max_trades,
            suggested_min_holding_minutes=suggested_min_holding,
            cooldown_cycles=cooldown,
            symbols_to_avoid=symbols_to_avoid,
            symbols_performing_well=symbols_good,
            summary=summary,
        )

    def _extract_trades(self, records: List[HistoryRecord]) -> List[TradeHistoryEntry]:
        """从历史记录中提取交易。"""
        trades: List[TradeHistoryEntry] = []
        for record in records:
            if record.kind != "execution":
                continue
            payload = record.payload or {}
            trade_list = payload.get("trades", [])
            for t in trade_list:
                if isinstance(t, TradeHistoryEntry):
                    trades.append(t)
                elif isinstance(t, dict):
                    try:
                        trades.append(TradeHistoryEntry(**t))
                    except Exception:
                        pass
        return trades

    def _calculate_win_rate(self, trades: List[TradeHistoryEntry]) -> Optional[float]:
        """计算胜率。"""
        if not trades:
            return None
        wins = sum(1 for t in trades if (t.realized_pnl or 0) > 0)
        return wins / len(trades) if trades else None

    def _calculate_avg_holding_minutes(
        self, trades: List[TradeHistoryEntry]
    ) -> Optional[float]:
        """计算平均持仓时间(分钟)。"""
        holding_times = []
        for t in trades:
            if t.entry_ts and t.exit_ts and t.exit_ts > t.entry_ts:
                holding_ms = t.exit_ts - t.entry_ts
                holding_times.append(holding_ms / 60000.0)  # 转分钟
        if not holding_times:
            return None
        return sum(holding_times) / len(holding_times)

    def _calculate_max_drawdown(self, digest: TradeDigest) -> Optional[float]:
        """从摘要中获取最大回撤。"""
        max_dd = 0.0
        for entry in digest.by_instrument.values():
            if entry.max_drawdown and entry.max_drawdown > max_dd:
                max_dd = entry.max_drawdown
        return max_dd if max_dd > 0 else None

    def _calculate_trades_per_hour(
        self, trades: List[TradeHistoryEntry]
    ) -> float:
        """计算每小时交易数。"""
        if len(trades) < 2:
            return 0.0
        timestamps = sorted(t.ts for t in trades if t.ts)
        if len(timestamps) < 2:
            return 0.0
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600000.0
        if duration_hours < 0.1:
            return 0.0
        return len(trades) / duration_hours

    def _count_consecutive_losses(self, trades: List[TradeHistoryEntry]) -> int:
        """计算最近连续亏损次数。"""
        if not trades:
            return 0
        sorted_trades = sorted(trades, key=lambda t: t.ts or 0, reverse=True)
        count = 0
        for t in sorted_trades:
            if (t.realized_pnl or 0) < 0:
                count += 1
            else:
                break
        return count

    def _analyze_sharpe_trend(self) -> Optional[str]:
        """分析夏普比趋势。"""
        if len(self._sharpe_history) < 3:
            return None

        recent = [s for _, s in self._sharpe_history[-5:]]
        if len(recent) < 3:
            return None

        # 简单线性趋势
        avg_first_half = sum(recent[: len(recent) // 2]) / (len(recent) // 2)
        avg_second_half = sum(recent[len(recent) // 2 :]) / (
            len(recent) - len(recent) // 2
        )

        diff = avg_second_half - avg_first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    def _generate_lessons(
        self,
        trades: List[TradeHistoryEntry],
        sharpe: Optional[float],
        win_rate: Optional[float],
        avg_holding: Optional[float],
        trades_per_hour: float,
        digest: TradeDigest,
    ) -> List[TradingLesson]:
        """生成交易教训。"""
        lessons: List[TradingLesson] = []

        # 过度交易教训
        if trades_per_hour > self._overtrading_threshold:
            lessons.append(
                TradingLesson(
                    category="timing",
                    observation=f"每小时交易 {trades_per_hour:.1f} 笔，频率过高",
                    lesson="高频交易增加滑点和手续费成本，且容易受噪音影响",
                    recommendation="提高入场门槛，等待更明确的信号",
                    confidence=0.85,
                )
            )

        # 持仓时间过短
        if avg_holding is not None and avg_holding < 15:
            lessons.append(
                TradingLesson(
                    category="exit",
                    observation=f"平均持仓仅 {avg_holding:.1f} 分钟",
                    lesson="过早平仓错失趋势利润，且增加交易成本",
                    recommendation="延长持仓时间，让盈利充分发展",
                    confidence=0.80,
                )
            )

        # 胜率低
        if win_rate is not None and win_rate < self._win_rate_warning:
            lessons.append(
                TradingLesson(
                    category="entry",
                    observation=f"胜率仅 {win_rate:.1%}",
                    lesson="入场信号质量不足，需要更多确认条件",
                    recommendation="增加信号共振要求，提高置信度门槛",
                    confidence=0.75,
                )
            )

        # 基于标的表现的教训
        for symbol, entry in digest.by_instrument.items():
            if entry.win_rate is not None and entry.win_rate < 0.3:
                if entry.trade_count >= 3:
                    lessons.append(
                        TradingLesson(
                            category="selection",
                            observation=f"{symbol} 胜率仅 {entry.win_rate:.1%}，共 {entry.trade_count} 笔",
                            lesson=f"该标的近期表现不佳，可能不适合当前策略",
                            recommendation=f"考虑暂时回避 {symbol}",
                            confidence=0.70,
                        )
                    )

        return lessons[:5]  # 最多返回5条教训

    def _analyze_symbol_performance(
        self, digest: TradeDigest
    ) -> Tuple[List[str], List[str]]:
        """分析各标的表现，识别应回避和表现好的标的。"""
        to_avoid: List[str] = []
        performing_well: List[str] = []

        for symbol, entry in digest.by_instrument.items():
            # 至少3笔交易才有统计意义
            if entry.trade_count < 3:
                continue

            # 胜率过低
            if entry.win_rate is not None and entry.win_rate < 0.3:
                to_avoid.append(symbol)
            # 表现良好
            elif entry.win_rate is not None and entry.win_rate > 0.6:
                if entry.realized_pnl > 0:
                    performing_well.append(symbol)

        return to_avoid, performing_well

    def _generate_summary(
        self,
        sharpe: Optional[float],
        sharpe_trend: Optional[str],
        alerts: List[PerformanceAlert],
        lessons: List[TradingLesson],
        cooldown: int,
    ) -> str:
        """生成反思总结。"""
        parts = []

        # 整体状态
        if sharpe is not None:
            if sharpe < self._sharpe_critical:
                parts.append("策略表现严重不佳，建议立即暂停交易并深入反思。")
            elif sharpe < self._sharpe_warning:
                parts.append("策略表现欠佳，需要调整交易行为。")
            elif sharpe > 0.7:
                parts.append("策略运行良好，继续保持当前纪律。")
            else:
                parts.append("策略表现一般，有改进空间。")

        # 趋势
        if sharpe_trend == "declining":
            parts.append("夏普比呈下降趋势，需警惕。")
        elif sharpe_trend == "improving":
            parts.append("夏普比正在改善，策略调整初见成效。")

        # 主要问题
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        if critical_alerts:
            problems = [a.message.split("，")[0] for a in critical_alerts]
            parts.append(f"严重问题: {'; '.join(problems)}。")

        # 主要教训
        if lessons:
            main_lessons = [l.lesson for l in lessons[:2]]
            parts.append(f"主要教训: {'; '.join(main_lessons)}。")

        # 冷静期
        if cooldown > 0:
            parts.append(f"建议接下来 {cooldown} 个周期选择观望(noop)。")

        return " ".join(parts) if parts else "暂无特别发现。"
