"""In-memory history recorder implementation."""

from typing import List

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

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"InMemoryHistoryRecorder(records={len(self._records)})"
