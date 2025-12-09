"""短期记忆数据结构。

存储策略运行期间的决策历史，用于注入到 LLM 上下文中。
支持序列化/反序列化以实现持久化。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import deque

from ..models import (
    TradeDecisionAction,
    TradeHistoryEntry,
    TradeInstruction,
)


@dataclass
class DecisionRecord:
    """单次决策记录。

    记录一个决策周期的关键信息，用于后续决策参考。
    """

    # 标识
    compose_id: str
    cycle_index: int
    timestamp_ms: int

    # 决策内容
    action: Optional[TradeDecisionAction] = None
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None

    # 执行结果
    executed: bool = False
    exec_price: Optional[float] = None
    exec_qty: Optional[float] = None
    fee_cost: Optional[float] = None

    # 结果
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None

    def to_dict(self) -> Dict:
        """转换为字典，过滤 None 值。"""
        result = {
            "cycle": self.cycle_index,
            "ts": self.timestamp_ms,
        }

        if self.action and self.action != TradeDecisionAction.NOOP:
            result["action"] = self.action.value
        if self.symbol:
            result["symbol"] = self.symbol
        if self.quantity:
            result["qty"] = round(self.quantity, 6)
        if self.rationale:
            # 截断过长的 rationale
            result["reason"] = self.rationale[:100] if len(self.rationale) > 100 else self.rationale
        if self.confidence:
            result["confidence"] = round(self.confidence, 2)

        if self.executed:
            result["executed"] = True
            if self.exec_price:
                result["exec_price"] = round(self.exec_price, 2)
            if self.exec_qty:
                result["exec_qty"] = round(self.exec_qty, 6)
            if self.fee_cost:
                result["fee"] = round(self.fee_cost, 4)

        if self.realized_pnl is not None:
            result["realized_pnl"] = round(self.realized_pnl, 4)
        if self.unrealized_pnl is not None:
            result["unrealized_pnl"] = round(self.unrealized_pnl, 4)

        # 添加 compose_id 用于持久化
        result["compose_id"] = self.compose_id

        return result

    def to_full_dict(self) -> Dict[str, Any]:
        """转换为完整字典（用于持久化）。"""
        return {
            "compose_id": self.compose_id,
            "cycle_index": self.cycle_index,
            "timestamp_ms": self.timestamp_ms,
            "action": self.action.value if self.action else None,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "executed": self.executed,
            "exec_price": self.exec_price,
            "exec_qty": self.exec_qty,
            "fee_cost": self.fee_cost,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        """从字典恢复 DecisionRecord。"""
        action = None
        if data.get("action"):
            try:
                action = TradeDecisionAction(data["action"])
            except ValueError:
                pass

        return cls(
            compose_id=data.get("compose_id", ""),
            cycle_index=data.get("cycle_index", 0),
            timestamp_ms=data.get("timestamp_ms", 0),
            action=action,
            symbol=data.get("symbol"),
            quantity=data.get("quantity"),
            rationale=data.get("rationale"),
            confidence=data.get("confidence"),
            executed=data.get("executed", False),
            exec_price=data.get("exec_price"),
            exec_qty=data.get("exec_qty"),
            fee_cost=data.get("fee_cost"),
            realized_pnl=data.get("realized_pnl"),
            unrealized_pnl=data.get("unrealized_pnl"),
        )


class ShortTermMemory:
    """短期记忆容器。

    维护最近 N 次决策的记录，支持：
    - 记录新决策
    - 获取最近决策历史
    - 按 symbol 过滤
    - 获取决策统计摘要
    """

    def __init__(self, max_records: int = 10) -> None:
        """初始化短期记忆。

        Args:
            max_records: 最大记录数量（FIFO 淘汰）
        """
        self._max_records = max_records
        self._records: deque[DecisionRecord] = deque(maxlen=max_records)
        self._pending_signals: Dict[str, str] = {}  # symbol -> 待观察信号

    def record_decision(
        self,
        compose_id: str,
        cycle_index: int,
        timestamp_ms: int,
        instructions: List[TradeInstruction],
        trades: List[TradeHistoryEntry],
        rationale: Optional[str] = None,
    ) -> None:
        """记录一次决策周期的结果。

        Args:
            compose_id: 决策周期 ID
            cycle_index: 周期索引
            timestamp_ms: 时间戳
            instructions: 生成的交易指令
            trades: 实际执行的交易
            rationale: 决策理由
        """
        # 如果没有指令，记录 NOOP
        if not instructions:
            record = DecisionRecord(
                compose_id=compose_id,
                cycle_index=cycle_index,
                timestamp_ms=timestamp_ms,
                action=TradeDecisionAction.NOOP,
                rationale=rationale,
            )
            self._records.append(record)
            return

        # 为每个指令创建记录
        trade_map = {t.instruction_id: t for t in trades if t.instruction_id}

        for inst in instructions:
            trade = trade_map.get(inst.instruction_id)

            record = DecisionRecord(
                compose_id=compose_id,
                cycle_index=cycle_index,
                timestamp_ms=timestamp_ms,
                action=inst.action,
                symbol=inst.instrument.symbol,
                quantity=inst.quantity,
                rationale=inst.meta.get("rationale") if inst.meta else None,
                confidence=inst.meta.get("confidence") if inst.meta else None,
                executed=trade is not None,
                exec_price=trade.avg_exec_price if trade else None,
                exec_qty=trade.quantity if trade else None,
                fee_cost=trade.fee_cost if trade else None,
                realized_pnl=trade.realized_pnl if trade else None,
            )
            self._records.append(record)

    def get_recent(self, n: Optional[int] = None) -> List[DecisionRecord]:
        """获取最近 N 条决策记录。

        Args:
            n: 数量，默认返回全部

        Returns:
            决策记录列表（从旧到新）
        """
        records = list(self._records)
        if n is not None:
            records = records[-n:]
        return records

    def get_by_symbol(self, symbol: str, n: Optional[int] = None) -> List[DecisionRecord]:
        """获取指定 symbol 的决策记录。

        Args:
            symbol: 交易对
            n: 数量限制

        Returns:
            决策记录列表
        """
        records = [r for r in self._records if r.symbol == symbol]
        if n is not None:
            records = records[-n:]
        return records

    def get_last_action(self, symbol: str) -> Optional[DecisionRecord]:
        """获取指定 symbol 的最后一次非 NOOP 操作。

        Args:
            symbol: 交易对

        Returns:
            最后一次操作记录，如果没有则返回 None
        """
        for record in reversed(self._records):
            if record.symbol == symbol and record.action != TradeDecisionAction.NOOP:
                return record
        return None

    def set_pending_signal(self, symbol: str, signal: str) -> None:
        """设置待观察信号。

        例如："等待回调到 95000 再加仓"

        Args:
            symbol: 交易对
            signal: 信号描述
        """
        self._pending_signals[symbol] = signal

    def get_pending_signal(self, symbol: str) -> Optional[str]:
        """获取待观察信号。"""
        return self._pending_signals.get(symbol)

    def clear_pending_signal(self, symbol: str) -> None:
        """清除待观察信号。"""
        self._pending_signals.pop(symbol, None)

    def get_all_pending_signals(self) -> Dict[str, str]:
        """获取所有待观察信号。"""
        return dict(self._pending_signals)

    def get_stats(self) -> Dict:
        """获取记忆统计摘要。

        Returns:
            包含统计信息的字典
        """
        records = list(self._records)
        if not records:
            return {"total_decisions": 0}

        # 统计各类操作
        actions = {}
        executed_count = 0
        total_pnl = 0.0

        for r in records:
            action_key = r.action.value if r.action else "unknown"
            actions[action_key] = actions.get(action_key, 0) + 1
            if r.executed:
                executed_count += 1
            if r.realized_pnl:
                total_pnl += r.realized_pnl

        return {
            "total_decisions": len(records),
            "executed": executed_count,
            "actions": actions,
            "total_realized_pnl": round(total_pnl, 4),
            "pending_signals": len(self._pending_signals),
        }

    def clear(self) -> None:
        """清空所有记忆。"""
        self._records.clear()
        self._pending_signals.clear()

    # =========================================================================
    # 序列化/反序列化（用于持久化）
    # =========================================================================

    def to_state(self) -> Dict[str, Any]:
        """导出状态（用于持久化）。

        Returns:
            包含所有记忆数据的字典
        """
        return {
            "decisions": [r.to_full_dict() for r in self._records],
            "pending_signals": dict(self._pending_signals),
            "max_records": self._max_records,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ShortTermMemory":
        """从状态恢复（用于持久化恢复）。

        Args:
            state: 之前导出的状态字典

        Returns:
            恢复的 ShortTermMemory 实例
        """
        max_records = state.get("max_records", 10)
        memory = cls(max_records=max_records)

        # 恢复决策记录
        decisions = state.get("decisions", [])
        for d in decisions:
            record = DecisionRecord.from_dict(d)
            memory._records.append(record)

        # 恢复待观察信号
        memory._pending_signals = dict(state.get("pending_signals", {}))

        return memory

    def get_last_cycle_index(self) -> int:
        """获取最后一次决策的周期索引。"""
        if not self._records:
            return 0
        return self._records[-1].cycle_index

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"ShortTermMemory(records={len(self._records)}, max={self._max_records})"
