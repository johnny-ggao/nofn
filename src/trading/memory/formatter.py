"""记忆格式化器。

将短期记忆格式化为 LLM 可读的文本，注入到 prompt 中。
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .short_term import DecisionRecord, ShortTermMemory


class MemoryFormatter:
    """记忆格式化器。

    将 ShortTermMemory 中的数据格式化为结构化文本，
    便于 LLM 理解历史决策上下文。
    """

    def __init__(
        self,
        max_recent_decisions: int = 5,
        include_stats: bool = True,
        include_pending_signals: bool = True,
        language: str = "zh",
    ) -> None:
        """初始化格式化器。

        Args:
            max_recent_decisions: 最多展示多少条最近决策
            include_stats: 是否包含统计摘要
            include_pending_signals: 是否包含待观察信号
            language: 输出语言 ("zh" | "en")
        """
        self._max_recent = max_recent_decisions
        self._include_stats = include_stats
        self._include_pending = include_pending_signals
        self._lang = language

    def format(self, memory: ShortTermMemory) -> str:
        """格式化记忆为文本。

        Args:
            memory: 短期记忆实例

        Returns:
            格式化后的文本
        """
        if len(memory) == 0:
            return self._empty_memory_text()

        sections = []

        # 最近决策历史
        recent = memory.get_recent(self._max_recent)
        if recent:
            sections.append(self._format_recent_decisions(recent))

        # 待观察信号
        if self._include_pending:
            pending = memory.get_all_pending_signals()
            if pending:
                sections.append(self._format_pending_signals(pending))

        # 统计摘要
        if self._include_stats:
            stats = memory.get_stats()
            sections.append(self._format_stats(stats))

        return "\n\n".join(sections)

    def format_for_symbol(
        self,
        memory: ShortTermMemory,
        symbol: str,
        n: int = 3,
    ) -> str:
        """格式化特定 symbol 的记忆。

        Args:
            memory: 短期记忆实例
            symbol: 交易对
            n: 最多展示多少条

        Returns:
            格式化后的文本
        """
        records = memory.get_by_symbol(symbol, n)
        if not records:
            if self._lang == "zh":
                return f"[{symbol}] 无历史决策记录"
            return f"[{symbol}] No historical decisions"

        return self._format_recent_decisions(records, symbol_filter=symbol)

    def _empty_memory_text(self) -> str:
        """空记忆时的文本。"""
        if self._lang == "zh":
            return "【历史记忆】这是首次决策，无历史记录。"
        return "[History] This is the first decision, no historical records."

    def _format_recent_decisions(
        self,
        records: List[DecisionRecord],
        symbol_filter: Optional[str] = None,
    ) -> str:
        """格式化最近决策列表。"""
        if self._lang == "zh":
            title = f"【最近 {len(records)} 次决策】" if not symbol_filter else f"【{symbol_filter} 最近决策】"
        else:
            title = f"[Recent {len(records)} Decisions]" if not symbol_filter else f"[{symbol_filter} Recent]"

        lines = [title]
        for r in records:
            lines.append(self._format_single_decision(r))

        return "\n".join(lines)

    def _format_single_decision(self, r: DecisionRecord) -> str:
        """格式化单条决策记录。"""
        # 时间格式化
        dt = datetime.fromtimestamp(r.timestamp_ms / 1000, tz=timezone.utc)
        time_str = dt.strftime("%H:%M")

        # 操作描述
        if r.action is None or r.action.value == "noop":
            if self._lang == "zh":
                action_str = "观望"
            else:
                action_str = "HOLD"
        else:
            action_str = r.action.value.upper()

        # 构建行
        parts = [f"#{r.cycle_index}", time_str]

        if r.symbol:
            parts.append(r.symbol)

        parts.append(action_str)

        if r.quantity:
            parts.append(f"qty={r.quantity:.4f}")

        if r.executed:
            if self._lang == "zh":
                parts.append("✓已执行")
            else:
                parts.append("✓executed")

            if r.exec_price:
                parts.append(f"@{r.exec_price:.2f}")

            if r.realized_pnl is not None:
                pnl_str = f"+{r.realized_pnl:.2f}" if r.realized_pnl >= 0 else f"{r.realized_pnl:.2f}"
                parts.append(f"PnL={pnl_str}")
        else:
            if r.action and r.action.value != "noop":
                if self._lang == "zh":
                    parts.append("✗未执行")
                else:
                    parts.append("✗not executed")

        # 简短理由
        if r.rationale:
            reason = r.rationale[:50] + "..." if len(r.rationale) > 50 else r.rationale
            parts.append(f"({reason})")

        return "  " + " | ".join(parts)

    def _format_pending_signals(self, signals: Dict[str, str]) -> str:
        """格式化待观察信号。"""
        if self._lang == "zh":
            title = "【待观察信号】"
        else:
            title = "[Pending Signals]"

        lines = [title]
        for symbol, signal in signals.items():
            lines.append(f"  {symbol}: {signal}")

        return "\n".join(lines)

    def _format_stats(self, stats: Dict) -> str:
        """格式化统计摘要。"""
        if self._lang == "zh":
            title = "【决策统计】"
            items = [
                f"总决策: {stats.get('total_decisions', 0)}",
                f"已执行: {stats.get('executed', 0)}",
                f"累计盈亏: {stats.get('total_realized_pnl', 0):.2f}",
            ]
        else:
            title = "[Decision Stats]"
            items = [
                f"Total: {stats.get('total_decisions', 0)}",
                f"Executed: {stats.get('executed', 0)}",
                f"Total PnL: {stats.get('total_realized_pnl', 0):.2f}",
            ]

        # 操作分布
        actions = stats.get("actions", {})
        if actions:
            action_parts = [f"{k}={v}" for k, v in actions.items()]
            if self._lang == "zh":
                items.append(f"操作分布: {', '.join(action_parts)}")
            else:
                items.append(f"Actions: {', '.join(action_parts)}")

        return title + "\n  " + " | ".join(items)

    def to_dict(self, memory: ShortTermMemory) -> Dict:
        """将记忆转换为字典格式（用于 JSON 注入）。

        Args:
            memory: 短期记忆实例

        Returns:
            结构化的字典
        """
        recent = memory.get_recent(self._max_recent)

        return {
            "recent_decisions": [r.to_dict() for r in recent],
            "pending_signals": memory.get_all_pending_signals(),
            "stats": memory.get_stats() if self._include_stats else None,
        }
