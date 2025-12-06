"""Abstract base class for decision composers."""

from abc import ABC, abstractmethod

from ..models import ComposeContext, ComposeResult


class BaseComposer(ABC):
    """Abstract base class for trade decision composers.

    Composers take market context and produce trading instructions.
    """

    @abstractmethod
    async def compose(self, context: ComposeContext) -> ComposeResult:
        """Compose trading instructions from context.

        Args:
            context: Market context with features, portfolio, and digest

        Returns:
            ComposeResult with trading instructions
        """
        raise NotImplementedError
