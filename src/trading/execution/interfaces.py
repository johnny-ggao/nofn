"""Abstract base class for execution gateways."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..models import FeatureVector, TradeInstruction, TxResult


class BaseExecutionGateway(ABC):
    """Abstract base class for trade execution gateways.

    Implementations must provide:
    - execute: Execute trade instructions
    - fetch_balance: Get account balance
    - fetch_positions: Get current positions
    - close: Cleanup resources
    """

    @abstractmethod
    async def execute(
        self,
        instructions: List[TradeInstruction],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> List[TxResult]:
        """Execute trade instructions.

        Args:
            instructions: List of trade instructions to execute
            market_features: Optional market features for pricing (paper trading)

        Returns:
            List of transaction results
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch_balance(self) -> Dict:
        """Fetch account balance from exchange.

        Returns:
            Balance dictionary with free, used, and total amounts
        """
        raise NotImplementedError

    @abstractmethod
    async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Fetch current positions from exchange.

        Args:
            symbols: Optional list of symbols to filter

        Returns:
            List of position dictionaries
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Close the gateway and cleanup resources."""
        raise NotImplementedError

    async def test_connection(self) -> bool:
        """Test connectivity and authentication."""
        try:
            await self.fetch_balance()
            return True
        except Exception:
            return False
