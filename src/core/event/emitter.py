"""
事件发射器

实现发布/订阅模式，支持：
- 同步和异步监听器
- 事件过滤
- 事件历史记录
"""
from typing import Callable, Dict, List, Set, Optional, Union, Awaitable
from collections import defaultdict
import asyncio
from loguru import logger

from .types import (
    AnyEvent,
    AnyEventType,
    TradingEvent,
    TradingEventType,
    SystemEvent,
    SystemEventType,
)


# 监听器类型
SyncListener = Callable[[AnyEvent], None]
AsyncListener = Callable[[AnyEvent], Awaitable[None]]
Listener = Union[SyncListener, AsyncListener]


class EventEmitter:
    """
    事件发射器

    支持发布/订阅模式，可以：
    - 注册同步或异步监听器
    - 按事件类型过滤
    - 保留事件历史（可选）

    使用示例:
        emitter = EventEmitter()

        # 注册监听器
        @emitter.on(TradingEventType.ORDER_FILLED)
        async def on_order_filled(event: TradingEvent):
            print(f"Order filled: {event.order_id}")

        # 发射事件
        await emitter.emit(TradingEvent(
            event_type=TradingEventType.ORDER_FILLED,
            symbol="BTC/USDT",
            order_id="12345",
        ))
    """

    def __init__(
        self,
        keep_history: bool = False,
        max_history: int = 1000,
    ):
        """
        初始化事件发射器

        Args:
            keep_history: 是否保留事件历史
            max_history: 最大历史记录数
        """
        self._listeners: Dict[AnyEventType, List[Listener]] = defaultdict(list)
        self._global_listeners: List[Listener] = []
        self._keep_history = keep_history
        self._max_history = max_history
        self._history: List[AnyEvent] = []
        self._lock = asyncio.Lock()

    def on(
        self,
        event_type: Optional[AnyEventType] = None,
    ) -> Callable[[Listener], Listener]:
        """
        装饰器：注册事件监听器

        Args:
            event_type: 要监听的事件类型（None 表示监听所有事件）

        Returns:
            装饰器函数

        Example:
            @emitter.on(TradingEventType.ORDER_FILLED)
            async def handle_fill(event):
                ...
        """
        def decorator(func: Listener) -> Listener:
            self.add_listener(func, event_type)
            return func
        return decorator

    def add_listener(
        self,
        listener: Listener,
        event_type: Optional[AnyEventType] = None,
    ) -> None:
        """
        添加事件监听器

        Args:
            listener: 监听器函数
            event_type: 要监听的事件类型（None 表示监听所有事件）
        """
        if event_type is None:
            self._global_listeners.append(listener)
        else:
            self._listeners[event_type].append(listener)

    def remove_listener(
        self,
        listener: Listener,
        event_type: Optional[AnyEventType] = None,
    ) -> bool:
        """
        移除事件监听器

        Args:
            listener: 要移除的监听器
            event_type: 事件类型（None 表示从全局监听器移除）

        Returns:
            是否成功移除
        """
        try:
            if event_type is None:
                self._global_listeners.remove(listener)
            else:
                self._listeners[event_type].remove(listener)
            return True
        except ValueError:
            return False

    async def emit(self, event: AnyEvent) -> None:
        """
        发射事件

        异步调用所有匹配的监听器

        Args:
            event: 要发射的事件
        """
        # 记录历史
        if self._keep_history:
            async with self._lock:
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        # 获取事件类型
        event_type = event.event_type

        # 收集所有匹配的监听器
        listeners = list(self._global_listeners) + list(self._listeners.get(event_type, []))

        # 并发调用所有监听器
        tasks = []
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    tasks.append(asyncio.create_task(listener(event)))
                else:
                    # 同步监听器在线程池中执行
                    listener(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")

        # 等待所有异步任务完成
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in async event listener: {result}")

    def emit_sync(self, event: AnyEvent) -> None:
        """
        同步发射事件（在事件循环中调度）

        Args:
            event: 要发射的事件
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event))
        except RuntimeError:
            # 没有运行的事件循环，直接运行
            asyncio.run(self.emit(event))

    def get_history(
        self,
        event_type: Optional[AnyEventType] = None,
        limit: int = 100,
    ) -> List[AnyEvent]:
        """
        获取事件历史

        Args:
            event_type: 过滤的事件类型（None 表示所有）
            limit: 最大返回数量

        Returns:
            事件列表（最新的在前）
        """
        if not self._keep_history:
            return []

        events = list(reversed(self._history))

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[:limit]

    def clear_history(self) -> int:
        """
        清除事件历史

        Returns:
            清除的事件数量
        """
        count = len(self._history)
        self._history.clear()
        return count

    def listener_count(self, event_type: Optional[AnyEventType] = None) -> int:
        """
        获取监听器数量

        Args:
            event_type: 事件类型（None 表示全局监听器）

        Returns:
            监听器数量
        """
        if event_type is None:
            return len(self._global_listeners)
        return len(self._listeners.get(event_type, []))

    def clear_listeners(self, event_type: Optional[AnyEventType] = None) -> int:
        """
        清除监听器

        Args:
            event_type: 事件类型（None 表示清除所有）

        Returns:
            清除的监听器数量
        """
        if event_type is None:
            count = len(self._global_listeners)
            for listeners in self._listeners.values():
                count += len(listeners)
            self._global_listeners.clear()
            self._listeners.clear()
            return count
        else:
            count = len(self._listeners.get(event_type, []))
            self._listeners[event_type].clear()
            return count


# 全局事件发射器实例
_global_emitter: Optional[EventEmitter] = None


def get_global_emitter() -> EventEmitter:
    """获取全局事件发射器"""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter(keep_history=True)
    return _global_emitter


def set_global_emitter(emitter: EventEmitter) -> None:
    """设置全局事件发射器"""
    global _global_emitter
    _global_emitter = emitter
