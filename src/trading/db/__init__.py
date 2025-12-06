"""数据库模块。

提供 SQLite 数据持久化支持。
"""

from .connection import (
    DatabaseManager,
    get_database_manager,
    get_db,
    get_engine,
    create_tables,
)
from .models import (
    Base,
    Strategy,
    StrategyDetail,
    StrategyHolding,
)
from .service import (
    PersistenceService,
    get_persistence_service,
)

__all__ = [
    # Connection
    "DatabaseManager",
    "get_database_manager",
    "get_db",
    "get_engine",
    "create_tables",
    # Models
    "Base",
    "Strategy",
    "StrategyDetail",
    "StrategyHolding",
    # Service
    "PersistenceService",
    "get_persistence_service",
]
