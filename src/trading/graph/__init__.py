"""LangGraph 交易决策图。

使用 LangGraph StateGraph 管理交易决策流程，
利用 Checkpointer 自动持久化状态，
利用 Store 实现跨会话长期记忆。
支持可选的 LanceDB 向量搜索功能。
"""

from .state import (
    TradingState,
    ReflectionState,
    DecisionMemory,
    DecisionSummary,
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
from .long_term_memory import (
    MemoryType,
    LongTermMemoryConfig,
    LongTermMemoryManager,
    get_sqlite_store_context,
    create_in_memory_store,
    get_memory_namespace,
)

# 向量存储（可选，需要安装 lancedb）
try:
    from .vector_store import (
        BaseVectorStore,
        LanceDBVectorStore,
        InMemoryVectorStore,
        OpenAIEmbedder,
        VectorStoreConfig,
        VectorSearchResult,
        create_vector_store,
    )
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

__all__ = [
    # State
    "TradingState",
    "ReflectionState",
    "DecisionMemory",
    "DecisionSummary",
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
    # Long-term Memory
    "MemoryType",
    "LongTermMemoryConfig",
    "LongTermMemoryManager",
    "get_sqlite_store_context",
    "create_in_memory_store",
    "get_memory_namespace",
    # Vector Store (optional)
    "VECTOR_SEARCH_AVAILABLE",
]

# 动态添加向量存储相关导出
if VECTOR_SEARCH_AVAILABLE:
    __all__.extend([
        "BaseVectorStore",
        "LanceDBVectorStore",
        "InMemoryVectorStore",
        "OpenAIEmbedder",
        "VectorStoreConfig",
        "VectorSearchResult",
        "create_vector_store",
    ])
