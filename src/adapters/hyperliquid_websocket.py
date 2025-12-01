"""
Hyperliquid WebSocket 订阅处理

实时监听用户的订单和仓位变化
"""
import asyncio
import json
from typing import Optional, Callable, Awaitable, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from termcolor import cprint

from hyperliquid.info import Info

from ..models import (
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeUpdateEvent,
    AccountUpdateEvent,
    WebSocketEventType,
    StopTriggerReason,
    PositionSide,
    OrderStatus,
)


class HyperliquidWebSocketManager:
    """
    Hyperliquid WebSocket 管理器

    负责订阅和处理 Hyperliquid 的 WebSocket 消息
    """

    def __init__(
        self,
        hl_info: Info,
        user_address: str,
    ):
        """
        初始化 WebSocket 管理器

        Args:
            hl_info: Hyperliquid Info API 实例
            user_address: 用户钱包地址
        """
        self.hl_info = hl_info
        self.user_address = user_address

        # 回调函数
        self.on_order_update: Optional[Callable[[OrderUpdateEvent], Awaitable[None]]] = None
        self.on_position_update: Optional[Callable[[PositionUpdateEvent], Awaitable[None]]] = None
        self.on_trade_update: Optional[Callable[[TradeUpdateEvent], Awaitable[None]]] = None
        self.on_account_update: Optional[Callable[[AccountUpdateEvent], Awaitable[None]]] = None

        # WebSocket 连接状态
        self._ws_task: Optional[asyncio.Task] = None
        self._is_running = False

        # 记录上次的仓位状态，用于检测变化
        self._last_positions: Dict[str, Dict] = {}

    async def subscribe(
        self,
        on_order_update: Optional[Callable[[OrderUpdateEvent], Awaitable[None]]] = None,
        on_position_update: Optional[Callable[[PositionUpdateEvent], Awaitable[None]]] = None,
        on_trade_update: Optional[Callable[[TradeUpdateEvent], Awaitable[None]]] = None,
        on_account_update: Optional[Callable[[AccountUpdateEvent], Awaitable[None]]] = None,
    ) -> bool:
        """
        启动 WebSocket 订阅

        Args:
            on_order_update: 订单更新回调
            on_position_update: 仓位更新回调
            on_trade_update: 成交更新回调
            on_account_update: 账户更新回调

        Returns:
            bool: 是否订阅成功
        """
        self.on_order_update = on_order_update
        self.on_position_update = on_position_update
        self.on_trade_update = on_trade_update
        self.on_account_update = on_account_update

        try:
            # 启动 WebSocket 订阅任务
            self._is_running = True
            self._ws_task = asyncio.create_task(self._run_websocket())

            cprint("✅ Hyperliquid WebSocket 订阅已启动", "green")
            return True

        except Exception as e:
            cprint(f"❌ WebSocket 订阅失败: {e}", "red")
            return False

    async def unsubscribe(self) -> bool:
        """取消订阅"""
        self._is_running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        cprint("⚠️  Hyperliquid WebSocket 已取消订阅", "yellow")
        return True

    async def _run_websocket(self) -> None:
        """运行 WebSocket 订阅循环"""
        try:
            # 订阅用户更新
            subscription = {"type": "userEvents", "user": self.user_address}

            # 创建 WebSocket 连接
            def callback(message: Dict[str, Any]):
                """WebSocket 消息回调"""
                asyncio.create_task(self._handle_message(message))

            # 使用 Hyperliquid SDK 的 subscribe 方法
            self.hl_info.subscribe(subscription, callback)

            # 保持运行
            while self._is_running:
                await asyncio.sleep(1)

        except Exception as e:
            cprint(f"❌ WebSocket 运行错误: {e}", "red")
            self._is_running = False

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        处理 WebSocket 消息

        Args:
            message: WebSocket 消息
        """
        try:
            channel = message.get("channel")
            data = message.get("data", {})

            if channel == "user":
                await self._handle_user_event(data)

        except Exception as e:
            cprint(f"⚠️  处理 WebSocket 消息失败: {e}", "yellow")

    async def _handle_user_event(self, data: Dict[str, Any]) -> None:
        """
        处理用户事件

        Args:
            data: 事件数据
        """
        # Hyperliquid 的用户事件包含订单、成交、仓位等信息
        fills = data.get("fills", [])        # 成交记录
        orders = data.get("orders", [])      # 订单更新
        positions = data.get("assetPositions", [])  # 仓位更新

        # 处理成交记录
        for fill in fills:
            await self._handle_fill(fill)

        # 处理订单更新
        for order in orders:
            await self._handle_order(order)

        # 处理仓位更新
        if positions:
            await self._handle_positions(positions)

    async def _handle_fill(self, fill: Dict[str, Any]) -> None:
        """
        处理成交记录

        Args:
            fill: 成交数据
        """
        try:
            # 解析成交数据
            event = TradeUpdateEvent(
                trade_id=fill.get("oid", ""),
                order_id=fill.get("oid", ""),
                symbol=fill.get("coin", ""),
                side=PositionSide.LONG if fill.get("side") == "B" else PositionSide.SHORT,
                price=Decimal(str(fill.get("px", 0))),
                amount=Decimal(str(fill.get("sz", 0))),
                fee=Decimal(str(fill.get("fee", 0))),
                is_closing=fill.get("closedPnl") is not None,
                realized_pnl=Decimal(str(fill.get("closedPnl", 0))) if fill.get("closedPnl") else None,
                raw_data=fill,
            )

            # 触发回调
            if self.on_trade_update:
                await self.on_trade_update(event)

        except Exception as e:
            cprint(f"⚠️  处理成交记录失败: {e}", "yellow")

    async def _handle_order(self, order: Dict[str, Any]) -> None:
        """
        处理订单更新

        Args:
            order: 订单数据
        """
        try:
            # 解析订单数据
            order_type = order.get("orderType", {})
            is_stop_loss = "trigger" in order_type and order_type.get("trigger", {}).get("triggerPx")
            is_take_profit = "trigger" in order_type and not is_stop_loss

            # 判断触发原因
            trigger_reason = None
            if is_stop_loss:
                trigger_reason = StopTriggerReason.STOP_LOSS
            elif is_take_profit:
                trigger_reason = StopTriggerReason.TAKE_PROFIT

            # 创建订单更新事件
            event = OrderUpdateEvent(
                order_id=str(order.get("oid", "")),
                symbol=order.get("coin", ""),
                side=PositionSide.LONG if order.get("side") == "B" else PositionSide.SHORT,
                order_type=order.get("orderType", {}).get("limit", "market"),
                status=self._parse_order_status(order.get("status")),
                price=Decimal(str(order.get("limitPx", 0))) if order.get("limitPx") else None,
                amount=Decimal(str(order.get("sz", 0))),
                filled=Decimal(str(order.get("szFilled", 0))),
                remaining=Decimal(str(order.get("szRemaining", 0))),
                is_stop_loss=is_stop_loss,
                is_take_profit=is_take_profit,
                trigger_reason=trigger_reason,
                raw_data=order,
            )

            # 如果是止损/止盈触发，输出日志
            if event.is_stop_triggered() and event.is_filled():
                reason_text = "止损" if is_stop_loss else "止盈"
                cprint(f"\n⚠️  检测到{reason_text}触发: {event.symbol}", "yellow")
                cprint(f"   订单ID: {event.order_id}", "yellow")
                cprint(f"   成交数量: {event.filled}", "yellow")

            # 触发回调
            if self.on_order_update:
                await self.on_order_update(event)

        except Exception as e:
            cprint(f"⚠️  处理订单更新失败: {e}", "yellow")

    async def _handle_positions(self, positions: List[Dict[str, Any]]) -> None:
        """
        处理仓位更新

        Args:
            positions: 仓位列表
        """
        try:
            current_positions = {}

            # 遍历当前仓位
            for pos in positions:
                symbol = pos.get("position", {}).get("coin", "")
                szi = Decimal(str(pos.get("position", {}).get("szi", 0)))

                current_positions[symbol] = pos

                # 检查是否有仓位变化
                if symbol in self._last_positions:
                    last_szi = Decimal(str(self._last_positions[symbol].get("position", {}).get("szi", 0)))

                    # 仓位从有到无 -> 平仓
                    if last_szi != 0 and szi == 0:
                        await self._handle_position_closed(symbol, pos, self._last_positions[symbol])

            # 更新记录
            self._last_positions = current_positions

        except Exception as e:
            cprint(f"⚠️  处理仓位更新失败: {e}", "yellow")

    async def _handle_position_closed(
        self,
        symbol: str,
        current_pos: Dict[str, Any],
        last_pos: Dict[str, Any],
    ) -> None:
        """
        处理仓位平仓事件

        Args:
            symbol: 交易对
            current_pos: 当前仓位数据（已平仓）
            last_pos: 上次仓位数据
        """
        try:
            # 创建仓位更新事件
            previous_szi = Decimal(str(last_pos.get("position", {}).get("szi", 0)))
            side = PositionSide.LONG if previous_szi > 0 else PositionSide.SHORT

            event = PositionUpdateEvent(
                symbol=symbol,
                side=side,
                position_amount=Decimal(0),
                previous_amount=abs(previous_szi),
                amount_change=-abs(previous_szi),
                is_closed=True,
                close_reason=StopTriggerReason.UNKNOWN,  # 需要结合订单数据判断
                raw_data=current_pos,
            )

            cprint(f"\n⚠️  检测到仓位平仓: {symbol}", "yellow")
            cprint(f"   方向: {side.value}", "yellow")
            cprint(f"   平仓数量: {abs(previous_szi)}", "yellow")

            # 触发回调
            if self.on_position_update:
                await self.on_position_update(event)

        except Exception as e:
            cprint(f"⚠️  处理仓位平仓失败: {e}", "yellow")

    def _parse_order_status(self, status: str) -> OrderStatus:
        """解析订单状态"""
        status_map = {
            "open": OrderStatus.OPEN,
            "filled": OrderStatus.CLOSED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(status, OrderStatus.OPEN)