"""
存储抽象层

参考 ValueCell 的存储设计，提供统一的存储接口和多种实现：
- InMemory: 用于测试和开发
- SQLite: 用于生产环境

设计优点：
1. 依赖注入 - 便于切换存储后端
2. 异步优先 - 所有操作都是 async
3. 类型安全 - 使用 Pydantic 模型
"""
from .base import BaseStore
from .order_store import OrderStore, InMemoryOrderStore, SQLiteOrderStore
from .trade_store import TradeStore, InMemoryTradeStore, SQLiteTradeStore
from .position_store import PositionStore, InMemoryPositionStore, SQLitePositionStore

__all__ = [
    "BaseStore",
    "OrderStore",
    "TradeStore",
    "PositionStore",
    "InMemoryOrderStore",
    "InMemoryTradeStore",
    "InMemoryPositionStore",
    "SQLiteOrderStore",
    "SQLiteTradeStore",
    "SQLitePositionStore",
]
