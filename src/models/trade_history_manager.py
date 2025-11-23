"""
交易历史管理器（SQLite 版本）

负责记录、存储和查询交易历史和持仓信息
使用 SQLite 数据库进行持久化存储
"""
import sqlite3
from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import threading

from .trade_history import TradeRecord, PositionRecord


class TradeHistoryManager:
    """
    交易历史管理器（使用 SQLite）

    职责：
    1. 记录每笔交易
    2. 跟踪持仓状态（开仓、平仓）
    3. 计算已实现盈亏
    4. 持久化到 SQLite 数据库
    5. 提供查询和统计功能
    """

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()  # 线程本地存储

        # 初始化数据库
        self._init_database()

        # 加载统计信息
        stats = self._get_db_stats()
        print(f"✅ 加载了 {stats['trades_count']} 笔交易记录, {stats['positions_count']} 个持仓")

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn'):
            # 增加 timeout 到 30 秒以处理并发访问
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row

            # 启用 WAL 模式以支持并发读写
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA busy_timeout=30000')  # 30秒

        return self._local.conn

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建交易记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                leverage INTEGER,
                fee REAL DEFAULT 0,
                realized_pnl REAL,
                position_id TEXT,
                order_id TEXT,
                note TEXT,
                FOREIGN KEY (position_id) REFERENCES positions(position_id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id)')

        # 创建持仓记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                leverage INTEGER NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                close_time TEXT,
                close_price REAL,
                close_reason TEXT,
                realized_pnl REAL,
                realized_pnl_percent REAL,
                max_unrealized_pnl REAL DEFAULT 0,
                min_unrealized_pnl REAL DEFAULT 0,
                note TEXT
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_close_time ON positions(close_time DESC)')

        conn.commit()

    def _get_db_stats(self) -> dict:
        """获取数据库统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM trades')
        trades_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM positions')
        positions_count = cursor.fetchone()[0]

        return {
            'trades_count': trades_count,
            'positions_count': positions_count
        }

    def record_trade(
        self,
        symbol: str,
        side: str,
        action: str,
        price: Decimal,
        amount: Decimal,
        leverage: Optional[int] = None,
        fee: Decimal = Decimal('0'),
        position_id: Optional[str] = None,
        order_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> TradeRecord:
        """
        记录一笔交易

        Args:
            symbol: 交易对
            side: long/short
            action: open/close/stop_loss/take_profit
            price: 成交价格
            amount: 数量
            leverage: 杠杆
            fee: 手续费
            position_id: 关联持仓ID
            order_id: 交易所订单ID
            note: 备注

        Returns:
            TradeRecord: 交易记录
        """
        trade_id = f"trade_{int(datetime.now().timestamp() * 1000)}"
        timestamp = datetime.now()

        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            action=action,
            price=price,
            amount=amount,
            leverage=leverage,
            fee=fee,
            position_id=position_id,
            order_id=order_id,
            note=note,
        )

        # 保存到数据库
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (trade_id, timestamp, symbol, side, action, price, amount,
                              leverage, fee, realized_pnl, position_id, order_id, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id,
            trade.timestamp.isoformat(),
            trade.symbol,
            trade.side,
            trade.action,
            float(trade.price),
            float(trade.amount),
            trade.leverage,
            float(trade.fee),
            float(trade.realized_pnl) if trade.realized_pnl else None,
            trade.position_id,
            trade.order_id,
            trade.note,
        ))

        conn.commit()

        # 如果是开仓，创建新持仓
        if action == "open":
            self._create_position(trade)

        # 如果是平仓，更新持仓状态
        elif action in ["close", "stop_loss", "take_profit"]:
            self._close_position(trade, action)

        return trade

    def _create_position(self, trade: TradeRecord) -> PositionRecord:
        """创建新持仓"""
        position_id = f"pos_{trade.symbol}_{int(trade.timestamp.timestamp())}"

        position = PositionRecord(
            position_id=position_id,
            symbol=trade.symbol,
            side=trade.side,
            entry_time=trade.timestamp,
            entry_price=trade.price,
            amount=trade.amount,
            leverage=trade.leverage or 1,
            trade_ids=[trade.trade_id],
        )

        # 更新trade的position_id
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE trades SET position_id = ? WHERE trade_id = ?
        ''', (position_id, trade.trade_id))

        # 保存持仓
        cursor.execute('''
            INSERT INTO positions (position_id, symbol, side, entry_time, entry_price,
                                  amount, leverage, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            position.position_id,
            position.symbol,
            position.side,
            position.entry_time.isoformat(),
            float(position.entry_price),
            float(position.amount),
            position.leverage,
        ))

        conn.commit()

        return position

    def _close_position(self, trade: TradeRecord, reason: str):
        """关闭持仓"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 查找当前持仓
        cursor.execute('''
            SELECT * FROM positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_time DESC
            LIMIT 1
        ''', (trade.symbol,))

        row = cursor.fetchone()
        if not row:
            print(f"⚠️  未找到 {trade.symbol} 的开仓记录")
            return

        position_id = row['position_id']
        entry_price = Decimal(str(row['entry_price']))
        amount = Decimal(str(row['amount']))
        side = row['side']

        # 计算已实现盈亏
        if side == "long":
            pnl = (trade.price - entry_price) * amount
        else:  # short
            pnl = (entry_price - trade.price) * amount

        # 减去手续费（开仓 + 平仓）
        cursor.execute('''
            SELECT SUM(fee) as total_fee
            FROM trades
            WHERE position_id = ?
        ''', (position_id,))

        total_fee = cursor.fetchone()['total_fee'] or 0
        pnl = pnl - Decimal(str(total_fee)) - trade.fee

        # 计算盈亏百分比
        cost = entry_price * amount / row['leverage']
        pnl_percent = float(pnl / cost * 100) if cost > 0 else 0.0

        # 更新持仓状态
        status = {
            "close": "closed",
            "stop_loss": "stopped",
            "take_profit": "taken",
        }.get(reason, "closed")

        cursor.execute('''
            UPDATE positions
            SET status = ?,
                close_time = ?,
                close_price = ?,
                close_reason = ?,
                realized_pnl = ?,
                realized_pnl_percent = ?
            WHERE position_id = ?
        ''', (
            status,
            trade.timestamp.isoformat(),
            float(trade.price),
            reason,
            float(pnl),
            pnl_percent,
            position_id,
        ))

        # 更新trade的盈亏和position_id
        cursor.execute('''
            UPDATE trades
            SET realized_pnl = ?, position_id = ?
            WHERE trade_id = ?
        ''', (float(pnl), position_id, trade.trade_id))

        conn.commit()

    def update_position_sl_tp(
        self,
        symbol: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ):
        """更新持仓的止损止盈"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 查找当前持仓
        cursor.execute('''
            SELECT position_id FROM positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_time DESC
            LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        if row:
            updates = []
            params = []

            if stop_loss is not None:
                updates.append('stop_loss = ?')
                params.append(float(stop_loss))

            if take_profit is not None:
                updates.append('take_profit = ?')
                params.append(float(take_profit))

            if updates:
                params.append(row['position_id'])
                cursor.execute(f'''
                    UPDATE positions
                    SET {', '.join(updates)}
                    WHERE position_id = ?
                ''', params)
                conn.commit()

    def get_open_position(self, symbol: str) -> Optional[PositionRecord]:
        """获取当前持仓"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM positions
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_time DESC
            LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        if row:
            return self._row_to_position(row)
        return None

    def get_all_open_positions(self) -> List[PositionRecord]:
        """获取所有当前持仓"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM positions
            WHERE status = 'open'
            ORDER BY entry_time DESC
        ''')

        return [self._row_to_position(row) for row in cursor.fetchall()]

    def get_position_by_id(self, position_id: str) -> Optional[PositionRecord]:
        """根据ID获取持仓"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM positions WHERE position_id = ?', (position_id,))

        row = cursor.fetchone()
        if row:
            return self._row_to_position(row)
        return None

    def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[PositionRecord]:
        """
        获取已平仓的持仓

        Args:
            symbol: 筛选交易对
            days: 最近N天
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM positions WHERE status != 'open'"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND close_time >= ?"
            params.append(cutoff)

        query += " ORDER BY close_time DESC"

        cursor.execute(query, params)

        return [self._row_to_position(row) for row in cursor.fetchall()]

    def get_trades(
        self,
        symbol: Optional[str] = None,
        action: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[TradeRecord]:
        """
        获取交易记录

        Args:
            symbol: 筛选交易对
            action: 筛选动作类型
            days: 最近N天
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if action:
            query += " AND action = ?"
            params.append(action)

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)

        return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_statistics(self, days: Optional[int] = None) -> dict:
        """
        获取统计数据

        Args:
            days: 统计最近N天，None表示全部

        Returns:
            dict: 统计结果
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM positions WHERE status != 'open'"
        params = []

        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND close_time >= ?"
            params.append(cutoff)

        cursor.execute(query, params)
        closed_positions = cursor.fetchall()

        if not closed_positions:
            return {
                'total_positions': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'avg_holding_time_hours': 0.0,
            }

        # 计算统计
        win_count = sum(1 for p in closed_positions if p['realized_pnl'] and p['realized_pnl'] > 0)
        loss_count = sum(1 for p in closed_positions if p['realized_pnl'] and p['realized_pnl'] < 0)

        pnls = [p['realized_pnl'] for p in closed_positions if p['realized_pnl'] is not None]
        total_pnl = sum(pnls) if pnls else 0.0
        max_profit = max(pnls) if pnls else 0.0
        max_loss = min(pnls) if pnls else 0.0

        # 平均持仓时间
        holding_times = []
        for p in closed_positions:
            if p['close_time']:
                entry_time = datetime.fromisoformat(p['entry_time'])
                close_time = datetime.fromisoformat(p['close_time'])
                hours = (close_time - entry_time).total_seconds() / 3600
                holding_times.append(hours)

        return {
            'total_positions': len(closed_positions),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_count / len(closed_positions) if closed_positions else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(closed_positions) if closed_positions else 0.0,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_holding_time_hours': sum(holding_times) / len(holding_times) if holding_times else 0.0,
        }

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        """将数据库行转换为 TradeRecord"""
        return TradeRecord(
            trade_id=row['trade_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            symbol=row['symbol'],
            side=row['side'],
            action=row['action'],
            price=Decimal(str(row['price'])),
            amount=Decimal(str(row['amount'])),
            leverage=row['leverage'],
            fee=Decimal(str(row['fee'])) if row['fee'] else Decimal('0'),
            realized_pnl=Decimal(str(row['realized_pnl'])) if row['realized_pnl'] else None,
            position_id=row['position_id'],
            order_id=row['order_id'],
            note=row['note'],
        )

    def _row_to_position(self, row: sqlite3.Row) -> PositionRecord:
        """将数据库行转换为 PositionRecord"""
        # 获取关联的交易ID
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT trade_id FROM trades WHERE position_id = ?', (row['position_id'],))
        trade_ids = [r['trade_id'] for r in cursor.fetchall()]

        return PositionRecord(
            position_id=row['position_id'],
            symbol=row['symbol'],
            side=row['side'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            entry_price=Decimal(str(row['entry_price'])),
            amount=Decimal(str(row['amount'])),
            leverage=row['leverage'],
            stop_loss=Decimal(str(row['stop_loss'])) if row['stop_loss'] else None,
            take_profit=Decimal(str(row['take_profit'])) if row['take_profit'] else None,
            status=row['status'],
            close_time=datetime.fromisoformat(row['close_time']) if row['close_time'] else None,
            close_price=Decimal(str(row['close_price'])) if row['close_price'] else None,
            close_reason=row['close_reason'],
            realized_pnl=Decimal(str(row['realized_pnl'])) if row['realized_pnl'] else None,
            realized_pnl_percent=row['realized_pnl_percent'],
            trade_ids=trade_ids,
            max_unrealized_pnl=Decimal(str(row['max_unrealized_pnl'])) if row['max_unrealized_pnl'] else Decimal('0'),
            min_unrealized_pnl=Decimal(str(row['min_unrealized_pnl'])) if row['min_unrealized_pnl'] else Decimal('0'),
            note=row['note'],
        )

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')

    def __del__(self):
        """析构时关闭连接"""
        self.close()
