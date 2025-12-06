"""LangGraph 交易决策图。

使用 LangGraph StateGraph 管理交易决策流程，
利用 Checkpointer 自动持久化状态。
"""

from .state import (
    TradingState,
    DecisionMemory,
    create_initial_state,
    create_memory_from_result,
    state_to_compose_context,
)
from .workflow import (
    create_trading_graph,
    TradingGraphConfig,
    get_sqlite_checkpointer_context,
    get_thread_config,
)
from .coordinator import (
    GraphDecisionCoordinator,
    GraphCoordinatorConfig,
)

__all__ = [
    # State
    "TradingState",
    "DecisionMemory",
    "create_initial_state",
    "create_memory_from_result",
    "state_to_compose_context",
    # Workflow
    "create_trading_graph",
    "TradingGraphConfig",
    "get_sqlite_checkpointer_context",
    "get_thread_config",
    # Coordinator
    "GraphDecisionCoordinator",
    "GraphCoordinatorConfig",
]
