"""
成交记录存储

提供成交记录的持久化存储
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
import json
import aiosqlite
from pathlib import Path

from .base import BaseStore, InMemoryStore
from ...models import Trade, PositionSide


class TradeStore(BaseStore[Trade], ABC):
    """
    成交记录存储抽象类
    """

    @abstractmethod
    async def get_by_order(self, order_id: str) -> List[Trade]:
        """获取指定订单的所有成交"""
        pass

    @abstractmethod
    async def get_by_symbol(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """获取指定交易对的成交记录"""
        pass

    @abstractmethod
    async def get_total_volume(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> Decimal:
        """获取总成交量"""
        pass


class InMemoryTradeStore(InMemoryStore[Trade], TradeStore):
    """内存成交存储"""

    def _get_item_id(self, item: Trade) -> str:
        return item.trade_id

    async def get_by_order(self, order_id: str) -> List[Trade]:
        return [t for t in self._items.values() if t.order_id == order_id]

    async def get_by_symbol(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        trades = [t for t in self._items.values() if t.symbol == symbol]
        if since:
            trades = [t for t in trades if t.timestamp and t.timestamp >= since]
        trades.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
        return trades[:limit]

    async def get_total_volume(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> Decimal:
        trades = await self.get_by_symbol(symbol, since, limit=10000)
        return sum((t.amount for t in trades), Decimal("0"))


class SQLiteTradeStore(TradeStore):
    """SQLite 成交存储"""

    def __init__(self, db_path: str = "data/trades.db"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    price TEXT NOT NULL,
                    fee TEXT,
                    fee_currency TEXT,
                    is_maker INTEGER,
                    timestamp TEXT,
                    raw_data TEXT
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_order ON trades(order_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            await db.commit()

        self._initialized = True

    def _trade_to_row(self, trade: Trade) -> tuple:
        return (
            trade.trade_id,
            trade.order_id,
            trade.symbol,
            trade.side.value if trade.side else None,
            str(trade.amount) if trade.amount else "0",
            str(trade.price) if trade.price else "0",
            str(trade.fee) if trade.fee else None,
            trade.fee_currency,
            1 if trade.is_maker else 0 if trade.is_maker is not None else None,
            trade.timestamp.isoformat() if trade.timestamp else None,
            json.dumps(trade.raw_data) if trade.raw_data else None,
        )

    def _row_to_trade(self, row: tuple) -> Trade:
        return Trade(
            trade_id=row[0],
            order_id=row[1],
            symbol=row[2],
            side=PositionSide(row[3]) if row[3] else None,
            amount=Decimal(row[4]) if row[4] else Decimal("0"),
            price=Decimal(row[5]) if row[5] else Decimal("0"),
            fee=Decimal(row[6]) if row[6] else None,
            fee_currency=row[7],
            is_maker=bool(row[8]) if row[8] is not None else None,
            timestamp=datetime.fromisoformat(row[9]) if row[9] else None,
            raw_data=json.loads(row[10]) if row[10] else None,
        )

    async def save(self, trade: Trade) -> None:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO trades
                    (trade_id, order_id, symbol, side, amount, price, fee, fee_currency,
                     is_maker, timestamp, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, self._trade_to_row(trade))
                await db.commit()

    async def get(self, trade_id: str) -> Optional[Trade]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_trade(row)
        return None

    async def delete(self, trade_id: str) -> bool:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM trades WHERE trade_id = ?", (trade_id,)
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
    ) -> List[Trade]:
        await self._ensure_initialized()

        query = "SELECT * FROM trades"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value.value if hasattr(value, "value") else value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        order_field = order_by or "timestamp"
        direction = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_field} {direction}"
        query += f" LIMIT {limit} OFFSET {offset}"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_trade(row) for row in rows]

    async def exists(self, trade_id: str) -> bool:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM trades WHERE trade_id = ?", (trade_id,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        await self._ensure_initialized()

        query = "SELECT COUNT(*) FROM trades"
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

    async def get_by_order(self, order_id: str) -> List[Trade]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM trades WHERE order_id = ? ORDER BY timestamp DESC",
                (order_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_trade(row) for row in rows]

    async def get_by_symbol(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Trade]:
        await self._ensure_initialized()

        query = "SELECT * FROM trades WHERE symbol = ?"
        params = [symbol]

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_trade(row) for row in rows]

    async def get_total_volume(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> Decimal:
        await self._ensure_initialized()

        query = "SELECT SUM(CAST(amount AS REAL)) FROM trades WHERE symbol = ?"
        params = [symbol]

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return Decimal(str(row[0])) if row and row[0] else Decimal("0")

    async def clear(self) -> int:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM trades")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                await db.execute("DELETE FROM trades")
                await db.commit()
                return count
