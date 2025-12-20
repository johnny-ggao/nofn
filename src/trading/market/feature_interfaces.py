"""Interfaces for feature computation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models import FeaturesPipelineResult, FeatureVector

# 从 features/base.py 重新导出，保持向后兼容
from .features.base import CandleBasedFeatureComputer


class BaseFeaturesPipeline(ABC):
    """Abstract base class for feature computation pipelines.

    Pipelines fetch market data and compute feature vectors for decision making.
    """

    @abstractmethod
    async def build(self) -> FeaturesPipelineResult:
        """Build feature vectors from market data.

        Returns:
            FeaturesPipelineResult with computed features
        """
        raise NotImplementedError


class MarketSnapshotFeatureComputer(ABC):
    """Abstract base class for market snapshot feature computation.

    Implementations transform raw market data (ticker, funding rate, OI)
    into standardized FeatureVector format.
    """

    @abstractmethod
    def compute_features(
        self,
        snapshot: Dict[str, Dict],
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        """Compute features from market snapshot.

        Args:
            snapshot: Dict mapping symbol to market data
            meta: Optional metadata to include in feature vectors

        Returns:
            List of FeatureVector objects
        """
        ...
