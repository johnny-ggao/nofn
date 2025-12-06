"""Abstract base classes for history and digest."""

from abc import ABC, abstractmethod
from typing import List

from ..models import HistoryRecord, TradeDigest


class BaseHistoryRecorder(ABC):
    """Abstract base for recording trade history."""

    @abstractmethod
    def record(self, entry: HistoryRecord) -> None:
        """Record a history entry.

        Args:
            entry: History record to store
        """
        raise NotImplementedError

    @abstractmethod
    def get_records(self) -> List[HistoryRecord]:
        """Get all history records.

        Returns:
            List of all recorded history entries
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear all history records."""
        raise NotImplementedError


class BaseDigestBuilder(ABC):
    """Abstract base for building trade digests."""

    @abstractmethod
    def build(self, records: List[HistoryRecord]) -> TradeDigest:
        """Build a digest from history records.

        Args:
            records: History records to analyze

        Returns:
            TradeDigest with performance statistics
        """
        raise NotImplementedError
