"""数据持久化服务。

提供策略、交易详情、持仓快照的持久化操作。
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

from loguru import logger
from sqlalchemy.orm import Session

from .connection import get_database_manager
from .models import Strategy, StrategyDetail, StrategyHolding, StrategyMemory


class PersistenceService:
    """数据持久化服务。

    负责将策略运行数据保存到 SQLite 数据库。
    """

    def __init__(self, db_url: Optional[str] = None):
        """初始化持久化服务。

        Args:
            db_url: 数据库 URL，默认使用 SQLite
        """
        self._db_manager = get_database_manager(db_url)
        self._db_manager.create_tables()

    def _get_session(self) -> Session:
        """获取数据库会话。"""
        return self._db_manager.get_session()

    # =========================================================================
    # 策略操作
    # =========================================================================

    def create_strategy(
        self,
        strategy_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Strategy:
        """创建策略记录。

        Args:
            strategy_id: 策略 ID
            name: 策略名称
            description: 策略描述
            config: UserRequest 配置
            metadata: 额外元数据

        Returns:
            创建的策略记录
        """
        with self._get_session() as session:
            strategy = Strategy(
                strategy_id=strategy_id,
                name=name,
                description=description,
                status="running",
                config=config,
                metadata_=metadata,
            )
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            logger.debug(f"创建策略记录: {strategy_id}")
            return strategy

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """获取策略记录。"""
        with self._get_session() as session:
            return session.query(Strategy).filter(
                Strategy.strategy_id == strategy_id
            ).first()

    def update_strategy_status(self, strategy_id: str, status: str) -> bool:
        """更新策略状态。"""
        with self._get_session() as session:
            strategy = session.query(Strategy).filter(
                Strategy.strategy_id == strategy_id
            ).first()
            if strategy:
                strategy.status = status
                session.commit()
                logger.debug(f"更新策略状态: {strategy_id} -> {status}")
                return True
            return False

    def list_strategies(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Strategy]:
        """列出策略。"""
        with self._get_session() as session:
            query = session.query(Strategy)
            if status:
                query = query.filter(Strategy.status == status)
            query = query.order_by(Strategy.created_at.desc()).limit(limit)
            return query.all()

    # =========================================================================
    # 交易详情操作
    # =========================================================================

    def save_trade(
        self,
        strategy_id: str,
        trade_id: str,
        symbol: str,
        trade_type: str,
        side: str,
        quantity: float,
        compose_id: Optional[str] = None,
        instruction_id: Optional[str] = None,
        leverage: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        avg_exec_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None,
        notional_entry: Optional[float] = None,
        notional_exit: Optional[float] = None,
        fee_cost: Optional[float] = None,
        holding_ms: Optional[int] = None,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        note: Optional[str] = None,
    ) -> StrategyDetail:
        """保存交易详情。

        Args:
            strategy_id: 策略 ID
            trade_id: 交易 ID
            symbol: 交易对
            trade_type: 持仓类型 (LONG/SHORT)
            side: 交易方向 (BUY/SELL)
            quantity: 交易数量
            其他参数: 交易详情字段

        Returns:
            创建的交易详情记录
        """
        with self._get_session() as session:
            detail = StrategyDetail(
                strategy_id=strategy_id,
                trade_id=trade_id,
                compose_id=compose_id,
                instruction_id=instruction_id,
                symbol=symbol,
                type=trade_type,
                side=side,
                leverage=Decimal(str(leverage)) if leverage else None,
                quantity=Decimal(str(quantity)),
                entry_price=Decimal(str(entry_price)) if entry_price else None,
                exit_price=Decimal(str(exit_price)) if exit_price else None,
                avg_exec_price=Decimal(str(avg_exec_price)) if avg_exec_price else None,
                unrealized_pnl=Decimal(str(unrealized_pnl)) if unrealized_pnl else None,
                realized_pnl=Decimal(str(realized_pnl)) if realized_pnl else None,
                realized_pnl_pct=Decimal(str(realized_pnl_pct)) if realized_pnl_pct else None,
                notional_entry=Decimal(str(notional_entry)) if notional_entry else None,
                notional_exit=Decimal(str(notional_exit)) if notional_exit else None,
                fee_cost=Decimal(str(fee_cost)) if fee_cost else None,
                holding_ms=holding_ms,
                entry_time=entry_time,
                exit_time=exit_time,
                note=note,
            )
            session.add(detail)
            session.commit()
            session.refresh(detail)
            logger.debug(f"保存交易: {trade_id} for {symbol}")
            return detail

    def save_trades_batch(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[StrategyDetail]:
        """批量保存交易详情。

        Args:
            trades: 交易详情列表，每个元素是 save_trade 的参数字典

        Returns:
            创建的交易详情记录列表
        """
        if not trades:
            return []

        with self._get_session() as session:
            details = []
            for trade_data in trades:
                detail = StrategyDetail(
                    strategy_id=trade_data["strategy_id"],
                    trade_id=trade_data["trade_id"],
                    compose_id=trade_data.get("compose_id"),
                    instruction_id=trade_data.get("instruction_id"),
                    symbol=trade_data["symbol"],
                    type=trade_data["trade_type"],
                    side=trade_data["side"],
                    leverage=Decimal(str(trade_data["leverage"])) if trade_data.get("leverage") else None,
                    quantity=Decimal(str(trade_data["quantity"])),
                    entry_price=Decimal(str(trade_data["entry_price"])) if trade_data.get("entry_price") else None,
                    exit_price=Decimal(str(trade_data["exit_price"])) if trade_data.get("exit_price") else None,
                    avg_exec_price=Decimal(str(trade_data["avg_exec_price"])) if trade_data.get("avg_exec_price") else None,
                    unrealized_pnl=Decimal(str(trade_data["unrealized_pnl"])) if trade_data.get("unrealized_pnl") else None,
                    realized_pnl=Decimal(str(trade_data["realized_pnl"])) if trade_data.get("realized_pnl") else None,
                    realized_pnl_pct=Decimal(str(trade_data["realized_pnl_pct"])) if trade_data.get("realized_pnl_pct") else None,
                    notional_entry=Decimal(str(trade_data["notional_entry"])) if trade_data.get("notional_entry") else None,
                    notional_exit=Decimal(str(trade_data["notional_exit"])) if trade_data.get("notional_exit") else None,
                    fee_cost=Decimal(str(trade_data["fee_cost"])) if trade_data.get("fee_cost") else None,
                    holding_ms=trade_data.get("holding_ms"),
                    entry_time=trade_data.get("entry_time"),
                    exit_time=trade_data.get("exit_time"),
                    note=trade_data.get("note"),
                )
                details.append(detail)

            session.add_all(details)
            session.commit()
            for detail in details:
                session.refresh(detail)
            logger.debug(f"批量保存 {len(details)} 笔交易")
            return details

    def get_trades(
        self,
        strategy_id: str,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[StrategyDetail]:
        """获取策略的交易记录。"""
        with self._get_session() as session:
            query = session.query(StrategyDetail).filter(
                StrategyDetail.strategy_id == strategy_id
            )
            if symbol:
                query = query.filter(StrategyDetail.symbol == symbol)
            query = query.order_by(StrategyDetail.created_at.desc()).limit(limit)
            return query.all()

    # =========================================================================
    # 持仓快照操作
    # =========================================================================

    def save_holding_snapshot(
        self,
        strategy_id: str,
        symbol: str,
        holding_type: str,
        quantity: float,
        leverage: Optional[float] = None,
        entry_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        unrealized_pnl_pct: Optional[float] = None,
        snapshot_ts: Optional[datetime] = None,
    ) -> StrategyHolding:
        """保存持仓快照。

        Args:
            strategy_id: 策略 ID
            symbol: 交易对
            holding_type: 持仓类型 (LONG/SHORT)
            quantity: 持仓数量
            其他参数: 持仓详情字段

        Returns:
            创建的持仓快照记录
        """
        with self._get_session() as session:
            holding = StrategyHolding(
                strategy_id=strategy_id,
                symbol=symbol,
                type=holding_type,
                leverage=Decimal(str(leverage)) if leverage else None,
                entry_price=Decimal(str(entry_price)) if entry_price else None,
                quantity=Decimal(str(quantity)),
                unrealized_pnl=Decimal(str(unrealized_pnl)) if unrealized_pnl else None,
                unrealized_pnl_pct=Decimal(str(unrealized_pnl_pct)) if unrealized_pnl_pct else None,
                snapshot_ts=snapshot_ts or datetime.now(timezone.utc),
            )
            session.add(holding)
            session.commit()
            session.refresh(holding)
            logger.debug(f"保存持仓快照: {symbol} qty={quantity}")
            return holding

    def save_holdings_batch(
        self,
        holdings: List[Dict[str, Any]],
    ) -> List[StrategyHolding]:
        """批量保存持仓快照。

        Args:
            holdings: 持仓快照列表

        Returns:
            创建的持仓快照记录列表
        """
        if not holdings:
            return []

        now = datetime.now(timezone.utc)
        with self._get_session() as session:
            records = []
            for h in holdings:
                record = StrategyHolding(
                    strategy_id=h["strategy_id"],
                    symbol=h["symbol"],
                    type=h["holding_type"],
                    leverage=Decimal(str(h["leverage"])) if h.get("leverage") else None,
                    entry_price=Decimal(str(h["entry_price"])) if h.get("entry_price") else None,
                    quantity=Decimal(str(h["quantity"])),
                    unrealized_pnl=Decimal(str(h["unrealized_pnl"])) if h.get("unrealized_pnl") else None,
                    unrealized_pnl_pct=Decimal(str(h["unrealized_pnl_pct"])) if h.get("unrealized_pnl_pct") else None,
                    snapshot_ts=h.get("snapshot_ts") or now,
                )
                records.append(record)

            session.add_all(records)
            session.commit()
            for record in records:
                session.refresh(record)
            logger.debug(f"批量保存 {len(records)} 条持仓快照")
            return records

    def get_latest_holdings(
        self,
        strategy_id: str,
    ) -> List[StrategyHolding]:
        """获取策略的最新持仓快照。"""
        with self._get_session() as session:
            # 使用子查询获取每个 symbol 的最新快照
            from sqlalchemy import func

            subquery = (
                session.query(
                    StrategyHolding.symbol,
                    func.max(StrategyHolding.snapshot_ts).label("max_ts")
                )
                .filter(StrategyHolding.strategy_id == strategy_id)
                .group_by(StrategyHolding.symbol)
                .subquery()
            )

            return (
                session.query(StrategyHolding)
                .join(
                    subquery,
                    (StrategyHolding.symbol == subquery.c.symbol) &
                    (StrategyHolding.snapshot_ts == subquery.c.max_ts)
                )
                .filter(StrategyHolding.strategy_id == strategy_id)
                .all()
            )

    def get_holdings_history(
        self,
        strategy_id: str,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[StrategyHolding]:
        """获取持仓历史。"""
        with self._get_session() as session:
            query = session.query(StrategyHolding).filter(
                StrategyHolding.strategy_id == strategy_id
            )
            if symbol:
                query = query.filter(StrategyHolding.symbol == symbol)
            query = query.order_by(StrategyHolding.snapshot_ts.desc()).limit(limit)
            return query.all()

    # =========================================================================
    # 记忆持久化操作
    # =========================================================================

    def save_memory(
        self,
        strategy_id: str,
        decisions: List[Dict[str, Any]],
        pending_signals: Dict[str, str],
        cycle_index: int,
    ) -> StrategyMemory:
        """保存策略记忆状态。

        使用 upsert 语义：如果存在则更新，否则创建。

        Args:
            strategy_id: 策略 ID
            decisions: 决策记录列表
            pending_signals: 待观察信号
            cycle_index: 最后决策周期索引

        Returns:
            保存的记忆记录
        """
        with self._get_session() as session:
            # 查找现有记录
            memory = session.query(StrategyMemory).filter(
                StrategyMemory.strategy_id == strategy_id
            ).first()

            if memory:
                # 更新现有记录
                memory.decisions = decisions
                memory.pending_signals = pending_signals
                memory.cycle_index = cycle_index
                logger.debug(f"更新策略记忆: {strategy_id}, {len(decisions)} 条决策")
            else:
                # 创建新记录
                memory = StrategyMemory(
                    strategy_id=strategy_id,
                    decisions=decisions,
                    pending_signals=pending_signals,
                    cycle_index=cycle_index,
                )
                session.add(memory)
                logger.debug(f"创建策略记忆: {strategy_id}, {len(decisions)} 条决策")

            session.commit()
            session.refresh(memory)
            return memory

    def load_memory(
        self,
        strategy_id: str,
    ) -> Optional[Dict[str, Any]]:
        """加载策略记忆状态。

        Args:
            strategy_id: 策略 ID

        Returns:
            记忆状态字典，包含 decisions, pending_signals, cycle_index。
            如果不存在返回 None。
        """
        with self._get_session() as session:
            memory = session.query(StrategyMemory).filter(
                StrategyMemory.strategy_id == strategy_id
            ).first()

            if not memory:
                logger.debug(f"未找到策略记忆: {strategy_id}")
                return None

            logger.debug(
                f"加载策略记忆: {strategy_id}, "
                f"{len(memory.decisions or [])} 条决策, "
                f"cycle={memory.cycle_index}"
            )

            return {
                "decisions": memory.decisions or [],
                "pending_signals": memory.pending_signals or {},
                "max_records": 10,  # 默认值，与 ShortTermMemory 保持一致
            }

    def delete_memory(self, strategy_id: str) -> bool:
        """删除策略记忆。

        Args:
            strategy_id: 策略 ID

        Returns:
            是否成功删除
        """
        with self._get_session() as session:
            memory = session.query(StrategyMemory).filter(
                StrategyMemory.strategy_id == strategy_id
            ).first()

            if memory:
                session.delete(memory)
                session.commit()
                logger.debug(f"删除策略记忆: {strategy_id}")
                return True

            return False


# 全局持久化服务实例
_persistence_service: Optional[PersistenceService] = None


def get_persistence_service(db_url: Optional[str] = None) -> PersistenceService:
    """获取全局持久化服务实例。"""
    global _persistence_service
    if _persistence_service is None:
        _persistence_service = PersistenceService(db_url)
    return _persistence_service
