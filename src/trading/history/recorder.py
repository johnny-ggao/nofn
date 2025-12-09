"""In-memory history recorder implementation."""

from typing import Any, Dict, List

from ..models import HistoryRecord
from .interfaces import BaseHistoryRecorder


class InMemoryHistoryRecorder(BaseHistoryRecorder):
    """Simple in-memory history recorder.

    Stores history records in memory for digest building and analysis.
    """

    def __init__(self, max_records: int = 1000) -> None:
        """Initialize recorder.

        Args:
            max_records: Maximum records to keep (older ones are evicted)
        """
        self._records: List[HistoryRecord] = []
        self._max_records = max_records

    def record(self, entry: HistoryRecord) -> None:
        """Record a history entry."""
        self._records.append(entry)

        # Evict old records if over limit
        if len(self._records) > self._max_records:
            # Keep most recent records
            self._records = self._records[-self._max_records:]

    def get_records(self) -> List[HistoryRecord]:
        """Get all history records."""
        return list(self._records)

    def clear(self) -> None:
        """Clear all history records."""
        self._records.clear()

    def load_from_db_trades(self, db_trades: List[Dict[str, Any]]) -> int:
        """从数据库交易记录加载历史记录。

        将数据库交易记录转换为 HistoryRecord 格式，用于夏普率等统计计算。

        Args:
            db_trades: 从 PersistenceService.get_trades_for_digest() 获取的交易列表

        Returns:
            成功加载的记录数
        """
        loaded = 0
        for item in db_trades:
            ts = item.get("ts")
            compose_id = item.get("compose_id")
            trade = item.get("trade")

            if not ts or not trade:
                continue

            # 创建 execution 类型的 HistoryRecord
            record = HistoryRecord(
                ts=ts,
                kind="execution",
                reference_id=compose_id or "",
                payload={"trades": [trade]},
            )
            self._records.append(record)
            loaded += 1

        # 按时间排序
        self._records.sort(key=lambda r: r.ts)

        # 限制记录数
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        return loaded

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"InMemoryHistoryRecorder(records={len(self._records)})"
