"""
订单存储

提供订单的持久化存储，支持内存和 SQLite 两种实现。
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
import json
import aiosqlite
from pathlib import Path

from .base import BaseStore, InMemoryStore
from ...models import Order, OrderStatus, PositionSide, OrderType


class OrderStore(BaseStore[Order], ABC):
    """
    订单存储抽象类

    定义订单特有的查询方法
    """

    @abstractmethod
    async def get_by_symbol(
        self,
        symbol: str,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        获取指定交易对的订单

        Args:
            symbol: 交易对
            status: 可选的订单状态过滤
            limit: 最大返回数量
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        获取所有未完成的订单

        Args:
            symbol: 可选的交易对过滤
        """
        pass

    @abstractmethod
    async def get_recent_orders(
        self,
        since: datetime,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        """
        获取指定时间后的订单

        Args:
            since: 开始时间
            symbol: 可选的交易对过滤
            limit: 最大返回数量
        """
        pass


class InMemoryOrderStore(InMemoryStore[Order], OrderStore):
    """
    内存订单存储

    用于测试和开发环境
    """

    def _get_item_id(self, item: Order) -> str:
        return item.order_id

    async def get_by_symbol(
        self,
        symbol: str,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
    ) -> List[Order]:
        orders = [o for o in self._items.values() if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]
        orders.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return orders[:limit]

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        open_statuses = {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}
        orders = [o for o in self._items.values() if o.status in open_statuses]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    async def get_recent_orders(
        self,
        since: datetime,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        orders = [
            o for o in self._items.values()
            if o.created_at and o.created_at >= since
        ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        orders.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return orders[:limit]


class SQLiteOrderStore(OrderStore):
    """
    SQLite 订单存储

    用于生产环境，提供持久化存储
    """

    def __init__(self, db_path: str = "data/orders.db"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """确保数据库表已创建"""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    price TEXT,
                    average_price TEXT,
                    filled TEXT NOT NULL DEFAULT '0',
                    remaining TEXT,
                    status TEXT NOT NULL,
                    fee TEXT,
                    fee_currency TEXT,
                    reduce_only INTEGER DEFAULT 0,
                    post_only INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    raw_data TEXT
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)")
            await db.commit()

        self._initialized = True

    def _order_to_row(self, order: Order) -> tuple:
        """将 Order 转换为数据库行"""
        return (
            order.order_id,
            order.symbol,
            order.order_type.value if order.order_type else None,
            order.side.value if order.side else None,
            str(order.amount) if order.amount else "0",
            str(order.price) if order.price else None,
            str(order.average_price) if order.average_price else None,
            str(order.filled) if order.filled else "0",
            str(order.remaining) if order.remaining else None,
            order.status.value if order.status else None,
            str(order.fee) if order.fee else None,
            order.fee_currency,
            1 if order.reduce_only else 0,
            1 if order.post_only else 0,
            order.created_at.isoformat() if order.created_at else None,
            order.updated_at.isoformat() if order.updated_at else None,
            json.dumps(order.raw_data) if order.raw_data else None,
        )

    def _row_to_order(self, row: tuple) -> Order:
        """将数据库行转换为 Order"""
        return Order(
            order_id=row[0],
            symbol=row[1],
            order_type=OrderType(row[2]) if row[2] else None,
            side=PositionSide(row[3]) if row[3] else None,
            amount=Decimal(row[4]) if row[4] else Decimal("0"),
            price=Decimal(row[5]) if row[5] else None,
            average_price=Decimal(row[6]) if row[6] else None,
            filled=Decimal(row[7]) if row[7] else Decimal("0"),
            remaining=Decimal(row[8]) if row[8] else None,
            status=OrderStatus(row[9]) if row[9] else None,
            fee=Decimal(row[10]) if row[10] else None,
            fee_currency=row[11],
            reduce_only=bool(row[12]),
            post_only=bool(row[13]),
            created_at=datetime.fromisoformat(row[14]) if row[14] else None,
            updated_at=datetime.fromisoformat(row[15]) if row[15] else None,
            raw_data=json.loads(row[16]) if row[16] else None,
        )

    async def save(self, order: Order) -> None:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO orders
                    (order_id, symbol, order_type, side, amount, price, average_price,
                     filled, remaining, status, fee, fee_currency, reduce_only, post_only,
                     created_at, updated_at, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, self._order_to_row(order))
                await db.commit()

    async def get(self, order_id: str) -> Optional[Order]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_order(row)
        return None

    async def delete(self, order_id: str) -> bool:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM orders WHERE order_id = ?", (order_id,)
                )
                await db.commit()
                return cursor.rowcount > 0

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = True,
    ) -> List[Order]:
        await self._ensure_initialized()

        query = "SELECT * FROM orders"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(Order, key):
                    conditions.append(f"{key} = ?")
                    params.append(value.value if hasattr(value, "value") else value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        order_field = order_by or "created_at"
        direction = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_field} {direction}"
        query += f" LIMIT {limit} OFFSET {offset}"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_order(row) for row in rows]

    async def exists(self, order_id: str) -> bool:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM orders WHERE order_id = ?", (order_id,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        await self._ensure_initialized()

        query = "SELECT COUNT(*) FROM orders"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value.value if hasattr(value, "value") else value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def get_by_symbol(
        self,
        symbol: str,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
    ) -> List[Order]:
        await self._ensure_initialized()

        query = "SELECT * FROM orders WHERE symbol = ?"
        params = [symbol]

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_order(row) for row in rows]

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        await self._ensure_initialized()

        query = "SELECT * FROM orders WHERE status IN (?, ?)"
        params = [OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_order(row) for row in rows]

    async def get_recent_orders(
        self,
        since: datetime,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Order]:
        await self._ensure_initialized()

        query = "SELECT * FROM orders WHERE created_at >= ?"
        params = [since.isoformat()]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_order(row) for row in rows]

    async def clear(self) -> int:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM orders")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                await db.execute("DELETE FROM orders")
                await db.commit()
                return count
