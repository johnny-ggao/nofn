"""Abstract base class for portfolio services."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import FeatureVector, PortfolioView, TradeHistoryEntry


class BasePortfolioService(ABC):
    """Abstract base class for portfolio management services.

    Tracks positions, balances, and calculates portfolio metrics.
    """

    @abstractmethod
    def get_view(self) -> PortfolioView:
        """Get the current portfolio view.

        Returns:
            PortfolioView with current positions and metrics
        """
        raise NotImplementedError

    @abstractmethod
    def apply_trades(
        self,
        trades: List[TradeHistoryEntry],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> None:
        """Apply trades to update portfolio state.

        Args:
            trades: List of executed trades
            market_features: Optional market data for pricing
        """
        raise NotImplementedError

    @abstractmethod
    def update_prices(self, market_features: List[FeatureVector]) -> None:
        """Update position prices from market features.

        Args:
            market_features: Current market data
        """
        raise NotImplementedError
