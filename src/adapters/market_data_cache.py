"""
行情数据缓存管理器

缓存 WebSocket 推送的行情数据，支持：
- K线数据缓存（按交易对和时间周期）
- Ticker 数据缓存（最新价格）
- 标记价格缓存
- 缓存数量限制和自动清理
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Deque, Any, Callable, Awaitable
from threading import Lock
from termcolor import cprint

from .binance_websocket import TickerData, KlineData, MarkPriceData


@dataclass
class CacheConfig:
    """缓存配置"""
    # K线缓存数量限制（每个交易对每个周期）
    max_klines_per_symbol: int = 500

    # Ticker 历史记录数量（用于计算短期变化）
    max_ticker_history: int = 60

    # 标记价格历史记录数量
    max_mark_price_history: int = 100

    # 缓存过期时间（秒），0 表示不过期
    ticker_ttl: int = 60
    kline_ttl: int = 0  # K线不过期
    mark_price_ttl: int = 60


@dataclass
class CachedTicker:
    """缓存的 Ticker 数据"""
    symbol: str
    last_price: Decimal
    price_change_percent: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    timestamp: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedKline:
    """缓存的 K线数据"""
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool


class MarketDataCache:
    """
    行情数据缓存管理器

    功能：
    - 缓存实时 Ticker 数据
    - 缓存 K线数据（支持多时间周期）
    - 缓存标记价格和资金费率
    - 自动控制缓存数量
    - 提供便捷的查询接口
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化缓存管理器

        Args:
            config: 缓存配置，None 使用默认配置
        """
        self.config = config or CacheConfig()

        # Ticker 缓存: symbol -> CachedTicker
        self._tickers: Dict[str, CachedTicker] = {}
        # Ticker 历史: symbol -> deque of (timestamp, price)
        self._ticker_history: Dict[str, Deque] = {}

        # K线缓存: symbol -> interval -> deque of CachedKline
        self._klines: Dict[str, Dict[str, Deque[CachedKline]]] = {}

        # 标记价格缓存: symbol -> MarkPriceData
        self._mark_prices: Dict[str, MarkPriceData] = {}
        # 标记价格历史: symbol -> deque of MarkPriceData
        self._mark_price_history: Dict[str, Deque[MarkPriceData]] = {}

        # 锁（线程安全）
        self._lock = Lock()

        # 统计
        self._stats = {
            "ticker_updates": 0,
            "kline_updates": 0,
            "mark_price_updates": 0,
        }

    # ==================== 数据更新方法 ====================

    def update_ticker(self, ticker: TickerData) -> None:
        """
        更新 Ticker 数据

        Args:
            ticker: WebSocket 推送的 Ticker 数据
        """
        with self._lock:
            symbol = ticker.symbol

            # 更新最新 Ticker
            self._tickers[symbol] = CachedTicker(
                symbol=symbol,
                last_price=ticker.last_price,
                price_change_percent=ticker.price_change_percent,
                high_price=ticker.high_price,
                low_price=ticker.low_price,
                volume=ticker.volume,
                timestamp=ticker.timestamp,
                raw_data=ticker.raw_data,
            )

            # 更新历史记录
            if symbol not in self._ticker_history:
                self._ticker_history[symbol] = deque(maxlen=self.config.max_ticker_history)

            self._ticker_history[symbol].append((ticker.timestamp, ticker.last_price))
            self._stats["ticker_updates"] += 1

    def update_kline(self, kline: KlineData) -> None:
        """
        更新 K线数据

        Args:
            kline: WebSocket 推送的 K线数据
        """
        with self._lock:
            symbol = kline.symbol
            interval = kline.interval

            # 初始化结构
            if symbol not in self._klines:
                self._klines[symbol] = {}
            if interval not in self._klines[symbol]:
                self._klines[symbol][interval] = deque(maxlen=self.config.max_klines_per_symbol)

            kline_deque = self._klines[symbol][interval]

            # 创建缓存对象
            cached = CachedKline(
                symbol=symbol,
                interval=interval,
                open_time=kline.open_time,
                close_time=kline.close_time,
                open=kline.open,
                high=kline.high,
                low=kline.low,
                close=kline.close,
                volume=kline.volume,
                is_closed=kline.is_closed,
            )

            # 如果是同一根K线的更新，替换最后一条
            if kline_deque and kline_deque[-1].open_time == kline.open_time:
                kline_deque[-1] = cached
            else:
                # 新K线，添加到队列
                kline_deque.append(cached)

            self._stats["kline_updates"] += 1

    def update_mark_price(self, mark_price: MarkPriceData) -> None:
        """
        更新标记价格数据

        Args:
            mark_price: WebSocket 推送的标记价格数据
        """
        with self._lock:
            symbol = mark_price.symbol

            # 更新最新标记价格
            self._mark_prices[symbol] = mark_price

            # 更新历史记录
            if symbol not in self._mark_price_history:
                self._mark_price_history[symbol] = deque(maxlen=self.config.max_mark_price_history)

            self._mark_price_history[symbol].append(mark_price)
            self._stats["mark_price_updates"] += 1

    # ==================== 数据查询方法 ====================

    def get_ticker(self, symbol: str) -> Optional[CachedTicker]:
        """
        获取最新 Ticker

        Args:
            symbol: 交易对

        Returns:
            CachedTicker 或 None
        """
        with self._lock:
            ticker = self._tickers.get(symbol)

            # 检查是否过期
            if ticker and self.config.ticker_ttl > 0:
                age = (datetime.now() - ticker.timestamp).total_seconds()
                if age > self.config.ticker_ttl:
                    return None

            return ticker

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """
        获取最新价格

        Args:
            symbol: 交易对

        Returns:
            最新价格或 None
        """
        ticker = self.get_ticker(symbol)
        return ticker.last_price if ticker else None

    def get_all_tickers(self) -> Dict[str, CachedTicker]:
        """获取所有 Ticker"""
        with self._lock:
            return dict(self._tickers)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
    ) -> List[CachedKline]:
        """
        获取 K线数据

        Args:
            symbol: 交易对
            interval: 时间周期（如 "1m", "5m", "1h"）
            limit: 返回数量限制，None 返回全部

        Returns:
            K线列表（时间升序）
        """
        with self._lock:
            if symbol not in self._klines or interval not in self._klines[symbol]:
                return []

            klines = list(self._klines[symbol][interval])

            if limit:
                klines = klines[-limit:]

            return klines

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        获取 OHLCV 数据（适配策略计算格式）

        Args:
            symbol: 交易对
            interval: 时间周期
            limit: 数量限制

        Returns:
            Dict with keys: open, high, low, close, volume
        """
        klines = self.get_klines(symbol, interval, limit)

        return {
            "open": [float(k.open) for k in klines],
            "high": [float(k.high) for k in klines],
            "low": [float(k.low) for k in klines],
            "close": [float(k.close) for k in klines],
            "volume": [float(k.volume) for k in klines],
        }

    def get_mark_price(self, symbol: str) -> Optional[MarkPriceData]:
        """
        获取标记价格

        Args:
            symbol: 交易对

        Returns:
            MarkPriceData 或 None
        """
        with self._lock:
            mark_price = self._mark_prices.get(symbol)

            # 检查是否过期
            if mark_price and self.config.mark_price_ttl > 0:
                age = (datetime.now() - mark_price.timestamp).total_seconds()
                if age > self.config.mark_price_ttl:
                    return None

            return mark_price

    def get_funding_rate(self, symbol: str) -> Optional[Decimal]:
        """
        获取资金费率

        Args:
            symbol: 交易对

        Returns:
            资金费率或 None
        """
        mark_price = self.get_mark_price(symbol)
        return mark_price.funding_rate if mark_price else None

    # ==================== 计算方法 ====================

    def get_price_change(
        self,
        symbol: str,
        seconds: int = 60,
    ) -> Optional[Decimal]:
        """
        计算价格变化百分比

        Args:
            symbol: 交易对
            seconds: 时间范围（秒）

        Returns:
            价格变化百分比或 None
        """
        with self._lock:
            if symbol not in self._ticker_history:
                return None

            history = self._ticker_history[symbol]
            if len(history) < 2:
                return None

            now = datetime.now()
            cutoff = now - timedelta(seconds=seconds)

            # 找到时间范围内的第一个价格
            old_price = None
            for ts, price in history:
                if ts >= cutoff:
                    old_price = price
                    break

            if old_price is None or old_price == 0:
                return None

            current_price = history[-1][1]
            return (current_price - old_price) / old_price * 100

    def get_volume_ma(
        self,
        symbol: str,
        interval: str,
        period: int = 20,
    ) -> Optional[Decimal]:
        """
        计算成交量均值

        Args:
            symbol: 交易对
            interval: 时间周期
            period: 均值周期

        Returns:
            成交量均值或 None
        """
        klines = self.get_klines(symbol, interval, limit=period)
        if len(klines) < period:
            return None

        total_volume = sum(k.volume for k in klines)
        return total_volume / period

    # ==================== 管理方法 ====================

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        清除缓存

        Args:
            symbol: 指定交易对，None 清除全部
        """
        with self._lock:
            if symbol:
                self._tickers.pop(symbol, None)
                self._ticker_history.pop(symbol, None)
                self._klines.pop(symbol, None)
                self._mark_prices.pop(symbol, None)
                self._mark_price_history.pop(symbol, None)
            else:
                self._tickers.clear()
                self._ticker_history.clear()
                self._klines.clear()
                self._mark_prices.clear()
                self._mark_price_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            kline_count = sum(
                sum(len(intervals) for intervals in symbols.values())
                for symbols in self._klines.values()
            )

            return {
                "ticker_count": len(self._tickers),
                "kline_count": kline_count,
                "mark_price_count": len(self._mark_prices),
                "symbols": list(self._tickers.keys()),
                "updates": dict(self._stats),
            }

    def get_cache_size(self) -> Dict[str, int]:
        """获取缓存大小"""
        with self._lock:
            sizes = {}
            for symbol in self._klines:
                for interval, klines in self._klines[symbol].items():
                    key = f"{symbol}_{interval}"
                    sizes[key] = len(klines)
            return sizes

    # ==================== WebSocket 回调适配 ====================

    async def on_ticker(self, ticker: TickerData) -> None:
        """WebSocket Ticker 回调"""
        self.update_ticker(ticker)

    async def on_kline(self, kline: KlineData) -> None:
        """WebSocket K线回调"""
        self.update_kline(kline)

    async def on_mark_price(self, mark_price: MarkPriceData) -> None:
        """WebSocket 标记价格回调"""
        self.update_mark_price(mark_price)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<MarketDataCache "
            f"tickers={stats['ticker_count']} "
            f"klines={stats['kline_count']} "
            f"mark_prices={stats['mark_price_count']}>"
        )


class CachedBinanceWebSocketManager:
    """
    带缓存的 Binance WebSocket 管理器

    封装 BinanceWebSocketManager 并自动缓存数据
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ):
        from .binance_websocket import BinanceWebSocketManager

        self.ws_manager = BinanceWebSocketManager(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        self.cache = MarketDataCache(cache_config)

        # 用户自定义回调
        self._user_on_ticker: Optional[Callable] = None
        self._user_on_kline: Optional[Callable] = None
        self._user_on_mark_price: Optional[Callable] = None

    async def start(self) -> bool:
        """启动 WebSocket"""
        return await self.ws_manager.start()

    async def stop(self) -> None:
        """停止 WebSocket"""
        await self.ws_manager.stop()

    async def subscribe_market_streams(
        self,
        symbols: List[str],
        streams: List[str] = None,
        on_ticker: Optional[Callable[[TickerData], Awaitable[None]]] = None,
        on_kline: Optional[Callable[[KlineData], Awaitable[None]]] = None,
        on_mark_price: Optional[Callable[[MarkPriceData], Awaitable[None]]] = None,
    ) -> bool:
        """
        订阅行情流（数据自动缓存）

        Args:
            symbols: 交易对列表
            streams: 数据流类型
            on_ticker: 用户 Ticker 回调（可选）
            on_kline: 用户 K线回调（可选）
            on_mark_price: 用户标记价格回调（可选）
        """
        self._user_on_ticker = on_ticker
        self._user_on_kline = on_kline
        self._user_on_mark_price = on_mark_price

        return await self.ws_manager.subscribe_market_streams(
            symbols=symbols,
            streams=streams,
            on_ticker=self._on_ticker_wrapper,
            on_kline=self._on_kline_wrapper,
            on_mark_price=self._on_mark_price_wrapper,
        )

    async def subscribe_user_stream(self, **kwargs) -> bool:
        """订阅用户数据流（透传）"""
        return await self.ws_manager.subscribe_user_stream(**kwargs)

    async def _on_ticker_wrapper(self, ticker: TickerData) -> None:
        """Ticker 回调包装器"""
        # 先缓存
        await self.cache.on_ticker(ticker)
        # 再调用用户回调
        if self._user_on_ticker:
            await self._user_on_ticker(ticker)

    async def _on_kline_wrapper(self, kline: KlineData) -> None:
        """K线回调包装器"""
        await self.cache.on_kline(kline)
        if self._user_on_kline:
            await self._user_on_kline(kline)

    async def _on_mark_price_wrapper(self, mark_price: MarkPriceData) -> None:
        """标记价格回调包装器"""
        await self.cache.on_mark_price(mark_price)
        if self._user_on_mark_price:
            await self._user_on_mark_price(mark_price)

    # ==================== 便捷查询方法 ====================

    def get_latest_price(self, symbol: str) -> Optional[Decimal]:
        """获取最新价格"""
        return self.cache.get_latest_price(symbol)

    def get_ticker(self, symbol: str) -> Optional[CachedTicker]:
        """获取 Ticker"""
        return self.cache.get_ticker(symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = None) -> List[CachedKline]:
        """获取 K线"""
        return self.cache.get_klines(symbol, interval, limit)

    def get_ohlcv(self, symbol: str, interval: str, limit: int = None) -> Dict[str, List]:
        """获取 OHLCV 数据"""
        return self.cache.get_ohlcv(symbol, interval, limit)

    def get_mark_price(self, symbol: str) -> Optional[MarkPriceData]:
        """获取标记价格"""
        return self.cache.get_mark_price(symbol)

    def get_funding_rate(self, symbol: str) -> Optional[Decimal]:
        """获取资金费率"""
        return self.cache.get_funding_rate(symbol)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()

    def __repr__(self) -> str:
        return f"<CachedBinanceWebSocketManager cache={self.cache}>"