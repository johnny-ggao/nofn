"""
持仓存储

提供持仓的持久化存储
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
import json
import aiosqlite
from pathlib import Path

from .base import BaseStore, InMemoryStore
from ...models import Position, PositionSide


class PositionStore(BaseStore[Position], ABC):
    """
    持仓存储抽象类
    """

    @abstractmethod
    async def get_active_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取所有活跃持仓（amount > 0）"""
        pass

    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """获取指定交易对的持仓"""
        pass

    @abstractmethod
    async def update_position(
        self,
        symbol: str,
        amount: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Optional[Position]:
        """更新持仓"""
        pass


class InMemoryPositionStore(InMemoryStore[Position], PositionStore):
    """内存持仓存储"""

    def _get_item_id(self, item: Position) -> str:
        return item.position_id or item.symbol

    async def get_active_positions(self, symbol: Optional[str] = None) -> List[Position]:
        positions = [p for p in self._items.values() if p.amount > 0]
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    async def get_by_symbol(self, symbol: str) -> Optional[Position]:
        for p in self._items.values():
            if p.symbol == symbol:
                return p
        return None

    async def update_position(
        self,
        symbol: str,
        amount: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Optional[Position]:
        position = await self.get_by_symbol(symbol)
        if not position:
            return None

        if amount is not None:
            position.amount = amount
        if entry_price is not None:
            position.entry_price = entry_price
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit

        await self.save(position)
        return position


class SQLitePositionStore(PositionStore):
    """SQLite 持仓存储"""

    def __init__(self, db_path: str = "data/positions.db"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    side TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    entry_price TEXT NOT NULL,
                    mark_price TEXT,
                    liquidation_price TEXT,
                    unrealized_pnl TEXT,
                    leverage INTEGER DEFAULT 1,
                    stop_loss TEXT,
                    take_profit TEXT,
                    opened_at TEXT,
                    updated_at TEXT,
                    raw_data TEXT
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")
            await db.commit()

        self._initialized = True

    def _position_to_row(self, position: Position) -> tuple:
        return (
            position.position_id or position.symbol,
            position.symbol,
            position.side.value if position.side else None,
            str(position.amount) if position.amount else "0",
            str(position.entry_price) if position.entry_price else "0",
            str(position.mark_price) if position.mark_price else None,
            str(position.liquidation_price) if position.liquidation_price else None,
            str(position.unrealized_pnl) if position.unrealized_pnl else None,
            position.leverage,
            str(position.stop_loss) if position.stop_loss else None,
            str(position.take_profit) if position.take_profit else None,
            position.opened_at.isoformat() if position.opened_at else None,
            datetime.now().isoformat(),
            json.dumps(position.raw_data) if position.raw_data else None,
        )

    def _row_to_position(self, row: tuple) -> Position:
        return Position(
            position_id=row[0],
            symbol=row[1],
            side=PositionSide(row[2]) if row[2] else None,
            amount=Decimal(row[3]) if row[3] else Decimal("0"),
            entry_price=Decimal(row[4]) if row[4] else Decimal("0"),
            mark_price=Decimal(row[5]) if row[5] else None,
            liquidation_price=Decimal(row[6]) if row[6] else None,
            unrealized_pnl=Decimal(row[7]) if row[7] else None,
            leverage=row[8] or 1,
            stop_loss=Decimal(row[9]) if row[9] else None,
            take_profit=Decimal(row[10]) if row[10] else None,
            opened_at=datetime.fromisoformat(row[11]) if row[11] else None,
            raw_data=json.loads(row[13]) if row[13] else None,
        )

    async def save(self, position: Position) -> None:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO positions
                    (position_id, symbol, side, amount, entry_price, mark_price,
                     liquidation_price, unrealized_pnl, leverage, stop_loss, take_profit,
                     opened_at, updated_at, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, self._position_to_row(position))
                await db.commit()

    async def get(self, position_id: str) -> Optional[Position]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM positions WHERE position_id = ?", (position_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_position(row)
        return None

    async def delete(self, position_id: str) -> bool:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM positions WHERE position_id = ?", (position_id,)
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
    ) -> List[Position]:
        await self._ensure_initialized()

        query = "SELECT * FROM positions"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value.value if hasattr(value, "value") else value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        order_field = order_by or "updated_at"
        direction = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_field} {direction}"
        query += f" LIMIT {limit} OFFSET {offset}"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_position(row) for row in rows]

    async def exists(self, position_id: str) -> bool:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM positions WHERE position_id = ?", (position_id,)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        await self._ensure_initialized()

        query = "SELECT COUNT(*) FROM positions"
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

    async def get_active_positions(self, symbol: Optional[str] = None) -> List[Position]:
        await self._ensure_initialized()

        query = "SELECT * FROM positions WHERE CAST(amount AS REAL) > 0"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY updated_at DESC"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_position(row) for row in rows]

    async def get_by_symbol(self, symbol: str) -> Optional[Position]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_position(row)
        return None

    async def update_position(
        self,
        symbol: str,
        amount: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Optional[Position]:
        await self._ensure_initialized()

        position = await self.get_by_symbol(symbol)
        if not position:
            return None

        updates = []
        params = []

        if amount is not None:
            updates.append("amount = ?")
            params.append(str(amount))
        if entry_price is not None:
            updates.append("entry_price = ?")
            params.append(str(entry_price))
        if stop_loss is not None:
            updates.append("stop_loss = ?")
            params.append(str(stop_loss))
        if take_profit is not None:
            updates.append("take_profit = ?")
            params.append(str(take_profit))

        if not updates:
            return position

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(symbol)

        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE positions SET {', '.join(updates)} WHERE symbol = ?",
                    params
                )
                await db.commit()

        return await self.get_by_symbol(symbol)

    async def clear(self) -> int:
        await self._ensure_initialized()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM positions")
                row = await cursor.fetchone()
                count = row[0] if row else 0
                await db.execute("DELETE FROM positions")
                await db.commit()
                return count
