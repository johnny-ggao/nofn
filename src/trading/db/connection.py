"""数据库连接和会话管理。"""

import os
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models.base import Base


def get_default_db_path() -> str:
    """获取默认数据库路径。"""
    # 优先使用环境变量
    if db_path := os.environ.get("NOFN_DB_PATH"):
        return db_path

    # 默认使用项目根目录下的 data 目录
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "nofn.db")


class DatabaseManager:
    """数据库连接和会话管理器。"""

    def __init__(self, db_url: Optional[str] = None):
        """初始化数据库管理器。

        Args:
            db_url: 数据库 URL，默认使用 SQLite
        """
        self.db_url = db_url or f"sqlite:///{get_default_db_path()}"
        self.engine: Optional[Engine] = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """初始化数据库引擎。"""
        connect_args = {}

        # SQLite 特殊配置
        if self.db_url.startswith("sqlite"):
            connect_args = {
                "check_same_thread": False,
                "timeout": 20,
            }
            self.engine = create_engine(
                self.db_url,
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(self.db_url)

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_engine(self) -> Engine:
        """获取数据库引擎。"""
        return self.engine

    def create_tables(self) -> None:
        """创建所有表。"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """删除所有表。"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """获取新的数据库会话。"""
        return self.SessionLocal()

    def get_db_session(self) -> Generator[Session, None, None]:
        """获取数据库会话（用于依赖注入）。"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(db_url: Optional[str] = None) -> DatabaseManager:
    """获取全局数据库管理器实例。"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_url)
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话。"""
    db_manager = get_database_manager()
    yield from db_manager.get_db_session()


def get_engine() -> Engine:
    """获取数据库引擎。"""
    return get_database_manager().get_engine()


def create_tables() -> None:
    """创建所有数据库表。"""
    get_database_manager().create_tables()


def drop_tables() -> None:
    """删除所有数据库表。"""
    get_database_manager().drop_tables()
