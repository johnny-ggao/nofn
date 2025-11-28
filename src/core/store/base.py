"""
存储基类

定义所有存储的通用接口和行为
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any
from datetime import datetime
import asyncio

T = TypeVar("T")


class BaseStore(ABC, Generic[T]):
    """
    存储抽象基类

    所有存储实现都必须继承此类。提供：
    - 基本的 CRUD 操作
    - 查询和过滤
    - 批量操作
    - 并发安全（通过 asyncio.Lock）
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    @abstractmethod
    async def save(self, item: T) -> None:
        """
        保存单个项目

        如果项目已存在（基于 ID），则更新；否则插入新记录。

        Args:
            item: 要保存的项目
        """
        pass

    @abstractmethod
    async def get(self, item_id: str) -> Optional[T]:
        """
        根据 ID 获取项目

        Args:
            item_id: 项目的唯一标识符

        Returns:
            找到的项目，或 None
        """
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """
        删除项目

        Args:
            item_id: 项目的唯一标识符

        Returns:
            是否成功删除
        """
        pass

    @abstractmethod
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = True,
    ) -> List[T]:
        """
        列出项目

        Args:
            filters: 过滤条件（字段名 -> 值）
            limit: 最大返回数量
            offset: 跳过的数量
            order_by: 排序字段
            descending: 是否降序

        Returns:
            项目列表
        """
        pass

    @abstractmethod
    async def exists(self, item_id: str) -> bool:
        """
        检查项目是否存在

        Args:
            item_id: 项目的唯一标识符

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        计算项目数量

        Args:
            filters: 过滤条件

        Returns:
            数量
        """
        pass

    async def save_batch(self, items: List[T]) -> None:
        """
        批量保存项目

        默认实现逐个保存，子类可以覆盖以优化性能。

        Args:
            items: 要保存的项目列表
        """
        async with self._lock:
            for item in items:
                await self.save(item)

    async def delete_batch(self, item_ids: List[str]) -> int:
        """
        批量删除项目

        Args:
            item_ids: 项目 ID 列表

        Returns:
            成功删除的数量
        """
        async with self._lock:
            count = 0
            for item_id in item_ids:
                if await self.delete(item_id):
                    count += 1
            return count

    async def clear(self) -> int:
        """
        清空所有项目

        Returns:
            删除的项目数量
        """
        pass


class InMemoryStore(BaseStore[T]):
    """
    内存存储基类

    用于测试和开发环境。数据存储在内存中，程序退出后丢失。
    """

    def __init__(self):
        super().__init__()
        self._items: Dict[str, T] = {}

    def _get_item_id(self, item: T) -> str:
        """获取项目的 ID，子类需要实现"""
        raise NotImplementedError("Subclass must implement _get_item_id")

    def _matches_filters(self, item: T, filters: Dict[str, Any]) -> bool:
        """检查项目是否匹配过滤条件"""
        for key, value in filters.items():
            item_value = getattr(item, key, None)
            if item_value != value:
                return False
        return True

    async def save(self, item: T) -> None:
        async with self._lock:
            item_id = self._get_item_id(item)
            self._items[item_id] = item

    async def get(self, item_id: str) -> Optional[T]:
        return self._items.get(item_id)

    async def delete(self, item_id: str) -> bool:
        async with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
            return False

    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = True,
    ) -> List[T]:
        items = list(self._items.values())

        # 应用过滤
        if filters:
            items = [item for item in items if self._matches_filters(item, filters)]

        # 应用排序
        if order_by:
            items.sort(
                key=lambda x: getattr(x, order_by, None) or "",
                reverse=descending,
            )

        # 应用分页
        return items[offset : offset + limit]

    async def exists(self, item_id: str) -> bool:
        return item_id in self._items

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        if not filters:
            return len(self._items)
        return len([item for item in self._items.values() if self._matches_filters(item, filters)])

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._items)
            self._items.clear()
            return count
