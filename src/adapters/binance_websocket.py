"""
Binance Futures WebSocket 管理器

支持:
- 行情数据流 (ticker, klines)
- 用户数据流 (orders, positions, account)

Binance WebSocket 端点:
- 公开流: wss://fstream.binance.com/ws/<streamName>
- 用户流: wss://fstream.binance.com/ws/<listenKey>
"""
import asyncio
import json
import time
import hmac
import hashlib
from typing import Optional, Callable, Awaitable, Dict, Any, List, Set
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from termcolor import cprint

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


class BinanceStreamType(Enum):
    """Binance 数据流类型"""
    TICKER = "ticker"           # 24小时 ticker
    MINI_TICKER = "miniTicker"  # 精简 ticker
    KLINE = "kline"             # K线数据
    MARK_PRICE = "markPrice"    # 标记价格
    BOOK_TICKER = "bookTicker"  # 最优挂单
    USER_DATA = "userData"      # 用户数据


@dataclass
class TickerData:
    """实时 Ticker 数据"""
    symbol: str
    last_price: Decimal
    price_change: Decimal
    price_change_percent: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    timestamp: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KlineData:
    """K线数据"""
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool  # K线是否已收盘
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarkPriceData:
    """标记价格数据"""
    symbol: str
    mark_price: Decimal
    index_price: Decimal
    funding_rate: Decimal
    next_funding_time: datetime
    timestamp: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BinanceWebSocketManager:
    """
    Binance Futures WebSocket 管理器

    功能:
    - 订阅多个交易对的行情数据
    - 订阅用户数据流 (订单、持仓、账户)
    - 自动重连和心跳保活
    - listenKey 自动续期
    """

    # WebSocket 端点
    WS_BASE_URL = "wss://fstream.binance.com"
    WS_TESTNET_URL = "wss://stream.binancefuture.com"

    # REST API 端点 (用于获取 listenKey)
    REST_BASE_URL = "https://fapi.binance.com"
    REST_TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ):
        """
        初始化 WebSocket 管理器

        Args:
            api_key: API 密钥
            api_secret: API 私钥
            testnet: 是否使用测试网
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # 设置端点
        self.ws_base_url = self.WS_TESTNET_URL if testnet else self.WS_BASE_URL
        self.rest_base_url = self.REST_TESTNET_URL if testnet else self.REST_BASE_URL

        # WebSocket 会话
        self._session: Optional[aiohttp.ClientSession] = None
        self._market_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._user_ws: Optional[aiohttp.ClientWebSocketResponse] = None

        # listenKey 管理
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task] = None

        # 运行状态
        self._is_running = False
        self._market_task: Optional[asyncio.Task] = None
        self._user_task: Optional[asyncio.Task] = None

        # 订阅的 streams
        self._subscribed_streams: Set[str] = set()

        # 回调函数 - 行情数据
        self.on_ticker: Optional[Callable[[TickerData], Awaitable[None]]] = None
        self.on_kline: Optional[Callable[[KlineData], Awaitable[None]]] = None
        self.on_mark_price: Optional[Callable[[MarkPriceData], Awaitable[None]]] = None

        # 回调函数 - 用户数据
        self.on_order_update: Optional[Callable[[OrderUpdateEvent], Awaitable[None]]] = None
        self.on_position_update: Optional[Callable[[PositionUpdateEvent], Awaitable[None]]] = None
        self.on_trade_update: Optional[Callable[[TradeUpdateEvent], Awaitable[None]]] = None
        self.on_account_update: Optional[Callable[[AccountUpdateEvent], Awaitable[None]]] = None

        # 缓存
        self._last_positions: Dict[str, Dict] = {}

    # ==================== 公共方法 ====================

    async def start(self) -> bool:
        """
        启动 WebSocket 连接

        Returns:
            bool: 是否启动成功
        """
        try:
            self._is_running = True
            self._session = aiohttp.ClientSession()

            cprint("✅ Binance WebSocket 管理器已启动", "green")
            return True

        except Exception as e:
            cprint(f"❌ WebSocket 启动失败: {e}", "red")
            return False

    async def stop(self) -> None:
        """停止 WebSocket 连接"""
        self._is_running = False

        # 取消任务
        for task in [self._market_task, self._user_task, self._listen_key_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 关闭 WebSocket
        if self._market_ws:
            await self._market_ws.close()
        if self._user_ws:
            await self._user_ws.close()

        # 关闭 session
        if self._session:
            await self._session.close()

        cprint("⚠️  Binance WebSocket 已停止", "yellow")

    async def subscribe_market_streams(
        self,
        symbols: List[str],
        streams: List[str] = None,
        on_ticker: Optional[Callable[[TickerData], Awaitable[None]]] = None,
        on_kline: Optional[Callable[[KlineData], Awaitable[None]]] = None,
        on_mark_price: Optional[Callable[[MarkPriceData], Awaitable[None]]] = None,
    ) -> bool:
        """
        订阅行情数据流

        Args:
            symbols: 交易对列表，如 ["BTCUSDT", "ETHUSDT"]
            streams: 数据流类型列表，如 ["ticker", "kline_1m", "markPrice"]
                     默认: ["miniTicker", "markPrice"]
            on_ticker: Ticker 回调
            on_kline: K线回调
            on_mark_price: 标记价格回调

        Returns:
            bool: 是否订阅成功
        """
        if not self._is_running:
            await self.start()

        # 设置回调
        if on_ticker:
            self.on_ticker = on_ticker
        if on_kline:
            self.on_kline = on_kline
        if on_mark_price:
            self.on_mark_price = on_mark_price

        # 默认订阅流
        if streams is None:
            streams = ["miniTicker", "markPrice"]

        # 构建订阅列表
        subscribe_list = []
        for symbol in symbols:
            symbol_lower = symbol.lower().replace("/", "").replace(":", "")
            for stream in streams:
                if stream == "miniTicker":
                    subscribe_list.append(f"{symbol_lower}@miniTicker")
                elif stream == "ticker":
                    subscribe_list.append(f"{symbol_lower}@ticker")
                elif stream.startswith("kline_"):
                    interval = stream.split("_")[1]
                    subscribe_list.append(f"{symbol_lower}@kline_{interval}")
                elif stream == "markPrice":
                    subscribe_list.append(f"{symbol_lower}@markPrice")
                elif stream == "bookTicker":
                    subscribe_list.append(f"{symbol_lower}@bookTicker")

        self._subscribed_streams.update(subscribe_list)

        # 启动行情 WebSocket
        self._market_task = asyncio.create_task(
            self._run_market_websocket(subscribe_list)
        )

        cprint(f"✅ 已订阅行情流: {', '.join(subscribe_list[:3])}{'...' if len(subscribe_list) > 3 else ''}", "green")
        return True

    async def subscribe_user_stream(
        self,
        on_order_update: Optional[Callable[[OrderUpdateEvent], Awaitable[None]]] = None,
        on_position_update: Optional[Callable[[PositionUpdateEvent], Awaitable[None]]] = None,
        on_trade_update: Optional[Callable[[TradeUpdateEvent], Awaitable[None]]] = None,
        on_account_update: Optional[Callable[[AccountUpdateEvent], Awaitable[None]]] = None,
    ) -> bool:
        """
        订阅用户数据流

        Args:
            on_order_update: 订单更新回调
            on_position_update: 仓位更新回调
            on_trade_update: 成交更新回调
            on_account_update: 账户更新回调

        Returns:
            bool: 是否订阅成功
        """
        if not self._is_running:
            await self.start()

        # 设置回调
        if on_order_update:
            self.on_order_update = on_order_update
        if on_position_update:
            self.on_position_update = on_position_update
        if on_trade_update:
            self.on_trade_update = on_trade_update
        if on_account_update:
            self.on_account_update = on_account_update

        try:
            # 获取 listenKey
            self._listen_key = await self._get_listen_key()
            if not self._listen_key:
                cprint("❌ 获取 listenKey 失败", "red")
                return False

            # 启动 listenKey 续期任务
            self._listen_key_task = asyncio.create_task(self._keepalive_listen_key())

            # 启动用户数据 WebSocket
            self._user_task = asyncio.create_task(self._run_user_websocket())

            cprint("✅ 已订阅用户数据流", "green")
            return True

        except Exception as e:
            cprint(f"❌ 订阅用户数据流失败: {e}", "red")
            return False

    # ==================== 内部方法 - listenKey 管理 ====================

    async def _get_listen_key(self) -> Optional[str]:
        """获取 listenKey"""
        try:
            url = f"{self.rest_base_url}/fapi/v1/listenKey"
            headers = {"X-MBX-APIKEY": self.api_key}

            async with self._session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("listenKey")
                else:
                    error = await resp.text()
                    cprint(f"❌ 获取 listenKey 失败: {error}", "red")
                    return None

        except Exception as e:
            cprint(f"❌ 获取 listenKey 异常: {e}", "red")
            return None

    async def _keepalive_listen_key(self) -> None:
        """保持 listenKey 活跃 (每30分钟续期一次)"""
        while self._is_running and self._listen_key:
            try:
                await asyncio.sleep(30 * 60)  # 30分钟

                url = f"{self.rest_base_url}/fapi/v1/listenKey"
                headers = {"X-MBX-APIKEY": self.api_key}

                async with self._session.put(url, headers=headers) as resp:
                    if resp.status == 200:
                        cprint("✅ listenKey 续期成功", "cyan")
                    else:
                        error = await resp.text()
                        cprint(f"⚠️  listenKey 续期失败: {error}", "yellow")
                        # 重新获取 listenKey
                        self._listen_key = await self._get_listen_key()

            except asyncio.CancelledError:
                break
            except Exception as e:
                cprint(f"⚠️  listenKey 续期异常: {e}", "yellow")

    # ==================== 内部方法 - WebSocket 运行 ====================

    async def _run_market_websocket(self, streams: List[str]) -> None:
        """运行行情 WebSocket"""
        stream_path = "/".join(streams)
        url = f"{self.ws_base_url}/stream?streams={stream_path}"

        while self._is_running:
            try:
                async with self._session.ws_connect(url) as ws:
                    self._market_ws = ws
                    cprint(f"✅ 行情 WebSocket 已连接", "green")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_market_message(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            cprint(f"⚠️  行情 WebSocket 错误: {ws.exception()}", "yellow")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            cprint("⚠️  行情 WebSocket 已关闭", "yellow")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                cprint(f"❌ 行情 WebSocket 异常: {e}", "red")

            # 重连等待
            if self._is_running:
                cprint("🔄 行情 WebSocket 将在 5 秒后重连...", "yellow")
                await asyncio.sleep(5)

    async def _run_user_websocket(self) -> None:
        """运行用户数据 WebSocket"""
        url = f"{self.ws_base_url}/ws/{self._listen_key}"

        while self._is_running:
            try:
                async with self._session.ws_connect(url) as ws:
                    self._user_ws = ws
                    cprint(f"✅ 用户数据 WebSocket 已连接", "green")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_user_message(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            cprint(f"⚠️  用户数据 WebSocket 错误: {ws.exception()}", "yellow")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            cprint("⚠️  用户数据 WebSocket 已关闭", "yellow")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                cprint(f"❌ 用户数据 WebSocket 异常: {e}", "red")

            # 重连等待
            if self._is_running:
                cprint("🔄 用户数据 WebSocket 将在 5 秒后重连...", "yellow")
                await asyncio.sleep(5)
                # 重新获取 listenKey
                self._listen_key = await self._get_listen_key()
                if self._listen_key:
                    url = f"{self.ws_base_url}/ws/{self._listen_key}"

    # ==================== 内部方法 - 消息处理 ====================

    async def _handle_market_message(self, message: Dict[str, Any]) -> None:
        """处理行情消息"""
        try:
            stream = message.get("stream", "")
            data = message.get("data", {})

            if "@miniTicker" in stream or "@ticker" in stream:
                await self._handle_ticker(data)
            elif "@kline_" in stream:
                await self._handle_kline(data)
            elif "@markPrice" in stream:
                await self._handle_mark_price(data)

        except Exception as e:
            cprint(f"⚠️  处理行情消息失败: {e}", "yellow")

    async def _handle_ticker(self, data: Dict[str, Any]) -> None:
        """处理 Ticker 数据"""
        try:
            ticker = TickerData(
                symbol=data.get("s", ""),
                last_price=Decimal(str(data.get("c", 0))),
                price_change=Decimal(str(data.get("p", 0))),
                price_change_percent=Decimal(str(data.get("P", 0))),
                high_price=Decimal(str(data.get("h", 0))),
                low_price=Decimal(str(data.get("l", 0))),
                volume=Decimal(str(data.get("v", 0))),
                quote_volume=Decimal(str(data.get("q", 0))),
                timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000),
                raw_data=data,
            )

            if self.on_ticker:
                await self.on_ticker(ticker)

        except Exception as e:
            cprint(f"⚠️  处理 Ticker 失败: {e}", "yellow")

    async def _handle_kline(self, data: Dict[str, Any]) -> None:
        """处理 K线数据"""
        try:
            k = data.get("k", {})
            kline = KlineData(
                symbol=data.get("s", ""),
                interval=k.get("i", ""),
                open_time=datetime.fromtimestamp(k.get("t", 0) / 1000),
                close_time=datetime.fromtimestamp(k.get("T", 0) / 1000),
                open=Decimal(str(k.get("o", 0))),
                high=Decimal(str(k.get("h", 0))),
                low=Decimal(str(k.get("l", 0))),
                close=Decimal(str(k.get("c", 0))),
                volume=Decimal(str(k.get("v", 0))),
                is_closed=k.get("x", False),
                raw_data=data,
            )

            if self.on_kline:
                await self.on_kline(kline)

        except Exception as e:
            cprint(f"⚠️  处理 K线失败: {e}", "yellow")

    async def _handle_mark_price(self, data: Dict[str, Any]) -> None:
        """处理标记价格数据"""
        try:
            mark_price = MarkPriceData(
                symbol=data.get("s", ""),
                mark_price=Decimal(str(data.get("p", 0))),
                index_price=Decimal(str(data.get("i", 0))),
                funding_rate=Decimal(str(data.get("r", 0))),
                next_funding_time=datetime.fromtimestamp(data.get("T", 0) / 1000),
                timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000),
                raw_data=data,
            )

            if self.on_mark_price:
                await self.on_mark_price(mark_price)

        except Exception as e:
            cprint(f"⚠️  处理标记价格失败: {e}", "yellow")

    async def _handle_user_message(self, message: Dict[str, Any]) -> None:
        """处理用户数据消息"""
        try:
            event_type = message.get("e", "")

            if event_type == "ORDER_TRADE_UPDATE":
                await self._handle_order_trade_update(message)
            elif event_type == "ACCOUNT_UPDATE":
                await self._handle_account_update(message)
            elif event_type == "listenKeyExpired":
                cprint("⚠️  listenKey 已过期，正在重新获取...", "yellow")
                self._listen_key = await self._get_listen_key()

        except Exception as e:
            cprint(f"⚠️  处理用户消息失败: {e}", "yellow")

    async def _handle_order_trade_update(self, message: Dict[str, Any]) -> None:
        """
        处理订单/成交更新

        Binance ORDER_TRADE_UPDATE 包含订单状态变化和成交信息
        """
        try:
            order_data = message.get("o", {})

            # 解析基本信息
            symbol = order_data.get("s", "")
            order_id = str(order_data.get("i", ""))
            client_order_id = order_data.get("c", "")
            side = PositionSide.LONG if order_data.get("S") == "BUY" else PositionSide.SHORT
            order_type = order_data.get("o", "")  # MARKET, LIMIT, STOP_MARKET, TAKE_PROFIT_MARKET
            order_status = order_data.get("X", "")  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED
            execution_type = order_data.get("x", "")  # NEW, TRADE, CANCELED, EXPIRED

            # 判断是否是止损/止盈订单
            is_stop_loss = order_type in ["STOP_MARKET", "STOP"]
            is_take_profit = order_type in ["TAKE_PROFIT_MARKET", "TAKE_PROFIT"]

            # 判断触发原因
            trigger_reason = None
            if is_stop_loss:
                trigger_reason = StopTriggerReason.STOP_LOSS
            elif is_take_profit:
                trigger_reason = StopTriggerReason.TAKE_PROFIT

            # 创建订单更新事件
            order_event = OrderUpdateEvent(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=self._parse_order_status(order_status),
                price=Decimal(str(order_data.get("p", 0))) if order_data.get("p") else None,
                amount=Decimal(str(order_data.get("q", 0))),
                filled=Decimal(str(order_data.get("z", 0))),
                remaining=Decimal(str(order_data.get("q", 0))) - Decimal(str(order_data.get("z", 0))),
                stop_price=Decimal(str(order_data.get("sp", 0))) if order_data.get("sp") else None,
                is_stop_loss=is_stop_loss,
                is_take_profit=is_take_profit,
                trigger_reason=trigger_reason,
                raw_data=message,
            )

            # 输出止损/止盈触发日志
            if order_event.is_stop_triggered() and order_event.is_filled():
                reason_text = "止损" if is_stop_loss else "止盈"
                cprint(f"\n⚠️  检测到{reason_text}触发: {symbol}", "yellow")
                cprint(f"   订单ID: {order_id}", "yellow")
                cprint(f"   成交数量: {order_event.filled}", "yellow")
                cprint(f"   成交价格: ${order_data.get('ap', 'N/A')}", "yellow")

            # 触发订单回调
            if self.on_order_update:
                await self.on_order_update(order_event)

            # 如果是成交，触发成交回调
            if execution_type == "TRADE":
                realized_pnl = Decimal(str(order_data.get("rp", 0))) if order_data.get("rp") else None
                is_closing = realized_pnl is not None and realized_pnl != 0

                trade_event = TradeUpdateEvent(
                    trade_id=str(order_data.get("t", "")),
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    price=Decimal(str(order_data.get("L", 0))),  # 最新成交价
                    amount=Decimal(str(order_data.get("l", 0))),  # 最新成交量
                    fee=Decimal(str(order_data.get("n", 0))),  # 手续费
                    fee_currency=order_data.get("N", ""),  # 手续费资产
                    is_closing=is_closing,
                    realized_pnl=realized_pnl,
                    raw_data=message,
                )

                if self.on_trade_update:
                    await self.on_trade_update(trade_event)

                # 输出成交日志
                if is_closing and realized_pnl:
                    pnl_emoji = "🟢" if realized_pnl > 0 else "🔴"
                    cprint(f"   {pnl_emoji} 已实现盈亏: ${float(realized_pnl):.2f}", "green" if realized_pnl > 0 else "red")

        except Exception as e:
            cprint(f"⚠️  处理订单更新失败: {e}", "yellow")

    async def _handle_account_update(self, message: Dict[str, Any]) -> None:
        """
        处理账户更新

        Binance ACCOUNT_UPDATE 包含余额和仓位变化
        """
        try:
            account_data = message.get("a", {})
            update_reason = account_data.get("m", "")  # DEPOSIT, WITHDRAW, ORDER, FUNDING_FEE, etc.

            # 处理余额更新
            balances = account_data.get("B", [])
            for balance in balances:
                if self.on_account_update:
                    account_event = AccountUpdateEvent(
                        total_balance=Decimal(str(balance.get("wb", 0))),
                        available_balance=Decimal(str(balance.get("cw", 0))),
                        raw_data=balance,
                    )
                    await self.on_account_update(account_event)

            # 处理仓位更新
            positions = account_data.get("P", [])
            for pos in positions:
                symbol = pos.get("s", "")
                position_amt = Decimal(str(pos.get("pa", 0)))
                entry_price = Decimal(str(pos.get("ep", 0)))
                unrealized_pnl = Decimal(str(pos.get("up", 0)))

                # 判断仓位方向
                if position_amt > 0:
                    side = PositionSide.LONG
                elif position_amt < 0:
                    side = PositionSide.SHORT
                else:
                    side = None

                # 检查仓位变化
                previous_amt = Decimal(0)
                if symbol in self._last_positions:
                    previous_amt = Decimal(str(self._last_positions[symbol].get("pa", 0)))

                amount_change = position_amt - previous_amt
                is_closed = previous_amt != 0 and position_amt == 0

                # 判断平仓原因
                close_reason = None
                if is_closed:
                    if update_reason == "ORDER":
                        # 需要结合订单信息判断具体原因
                        close_reason = StopTriggerReason.UNKNOWN
                    elif update_reason == "LIQUIDATION":
                        close_reason = StopTriggerReason.LIQUIDATION

                # 创建仓位更新事件
                position_event = PositionUpdateEvent(
                    symbol=symbol,
                    side=side if side else (PositionSide.LONG if previous_amt > 0 else PositionSide.SHORT),
                    position_amount=abs(position_amt),
                    previous_amount=abs(previous_amt),
                    amount_change=amount_change,
                    entry_price=entry_price,
                    unrealized_pnl=unrealized_pnl,
                    is_closed=is_closed,
                    close_reason=close_reason,
                    raw_data=pos,
                )

                # 更新缓存
                self._last_positions[symbol] = pos

                # 输出仓位变化日志
                if is_closed:
                    cprint(f"\n⚠️  仓位已平仓: {symbol}", "yellow")
                    cprint(f"   原持仓: {abs(previous_amt)}", "yellow")
                elif amount_change != 0:
                    action = "加仓" if amount_change > 0 else "减仓"
                    cprint(f"\n📊 仓位变化: {symbol} {action}", "cyan")
                    cprint(f"   变化: {amount_change}, 当前: {abs(position_amt)}", "cyan")

                if self.on_position_update:
                    await self.on_position_update(position_event)

        except Exception as e:
            cprint(f"⚠️  处理账户更新失败: {e}", "yellow")

    def _parse_order_status(self, status: str) -> OrderStatus:
        """解析订单状态"""
        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.CLOSED,
            "CANCELED": OrderStatus.CANCELED,
            "EXPIRED": OrderStatus.EXPIRED,
            "REJECTED": OrderStatus.REJECTED,
        }
        return status_map.get(status, OrderStatus.OPEN)

    def __repr__(self) -> str:
        return f"<BinanceWebSocketManager testnet={self.testnet} running={self._is_running}>"