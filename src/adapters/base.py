"""
交易所适配器基类

定义所有交易所适配器必须实现的接口，确保不同交易所的统一调用方式
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Callable, Awaitable
from decimal import Decimal

from ..models import (
    Position,
    ExecutionResult,
    Balance,
    TradingIntent,
    PositionSide,
    OrderType,
    Candle,
    Ticker24h,
    FundingRate,
    LatestPrice,
    OrderBook,
    Order,
    Trade,
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeUpdateEvent,
    AccountUpdateEvent,
)


class BaseExchangeAdapter(ABC):
    """
    交易所适配器基类

    所有交易所适配器都必须继承此类并实现所有抽象方法
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        """
        初始化适配器

        Args:
            api_key: API 密钥
            api_secret: API 私钥
            **kwargs: 其他配置参数（如 testnet, passphrase 等）
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = kwargs
        self._exchange = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化交易所连接

        在此方法中创建 CCXT 交易所实例并进行必要的配置
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭交易所连接

        清理资源，关闭连接
        """
        pass

    # ========== 交易操作 ==========

    @abstractmethod
    async def open_position(
        self,
        symbol: str,
        side: PositionSide,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        leverage: int = 1,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        开仓

        Args:
            symbol: 交易对（如 "BTC/USDT"）
            side: 持仓方向（long/short）
            amount: 开仓数量
            order_type: 订单类型（market/limit）
            price: 限价单价格（仅限价单需要）
            leverage: 杠杆倍数
            stop_loss: 止损价格
            take_profit: 止盈价格
            **params: 其他交易所特定参数

        Returns:
            ExecutionResult: 执行结果
        """
        pass

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        amount: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        平仓

        Args:
            symbol: 交易对
            position_id: 持仓ID（如果交易所支持）
            amount: 平仓数量（None 表示全部平仓）
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        pass

    @abstractmethod
    async def modify_stop_loss_take_profit(
        self,
        position: Union[Position, Dict[str, Any]],  # 接受Position对象或字典
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        修改止损止盈

        Args:
            position: 持仓对象或持仓信息字典
            stop_loss: 新的止损价格（None 表示不修改）
            take_profit: 新的止盈价格（None 表示不修改）
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        **params
    ) -> ExecutionResult:
        """
        取消订单

        Args:
            order_id: 订单ID
            symbol: 交易对
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        pass

    # ========== 查询操作 ==========

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None,
        **params
    ) -> List[Position]:
        """
        获取持仓列表

        Args:
            symbol: 交易对（None 表示获取所有持仓）
            **params: 其他参数

        Returns:
            List[Position]: 持仓列表
        """
        pass

    @abstractmethod
    async def get_position(
        self,
        symbol: str,
        **params
    ) -> Optional[Position]:
        """
        获取指定交易对的持仓

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            Optional[Position]: 持仓信息（如果存在）
        """
        pass

    @abstractmethod
    async def get_balance(
        self,
        currency: Optional[str] = None,
        **params
    ) -> Balance:
        """
        获取账户余额

        Args:
            currency: 币种（None 表示默认币种，通常是 USDT）
            **params: 其他参数

        Returns:
            Balance: 余额信息
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        **params
    ) -> List[Order]:
        """
        获取当前委托订单（未完成的订单）

        Args:
            symbol: 交易对（None 表示获取所有交易对的订单）
            **params: 其他参数

        Returns:
            List[Order]: 委托订单列表
        """
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Order]:
        """
        获取历史订单（包括已完成、已取消的订单）

        Args:
            symbol: 交易对（None 表示获取所有交易对的订单）
            since: 开始时间戳（毫秒），None 表示不限制
            limit: 返回数据条数
            **params: 其他参数

        Returns:
            List[Order]: 历史订单列表
        """
        pass

    @abstractmethod
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Trade]:
        """
        获取历史成交记录

        Args:
            symbol: 交易对（None 表示获取所有交易对的成交）
            since: 开始时间戳（毫秒），None 表示不限制
            limit: 返回数据条数
            **params: 其他参数

        Returns:
            List[Trade]: 成交记录列表
        """
        pass

    @abstractmethod
    async def get_ticker(
        self,
        symbol: str,
        **params
    ) -> Dict[str, Any]:
        """
        获取行情信息

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            Dict[str, Any]: 行情信息
        """
        pass

    # ========== 辅助方法 ==========

    @abstractmethod
    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        **params
    ) -> bool:
        """
        设置杠杆

        Args:
            symbol: 交易对
            leverage: 杠杆倍数
            **params: 其他参数

        Returns:
            bool: 是否设置成功
        """
        pass

    # ========== 行情数据 ==========

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Candle]:
        """
        获取K线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期（如 '1m', '5m', '15m', '1h', '4h', '1d'）
            since: 开始时间戳（毫秒），None 表示最近的数据
            limit: 返回数据条数
            **params: 其他参数

        Returns:
            List[Kline]: K线数据列表
        """
        pass

    @abstractmethod
    async def get_ticker_24h(
        self,
        symbol: str,
        **params
    ) -> Ticker24h:
        """
        获取24小时行情统计

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            Ticker24h: 24小时行情数据
        """
        pass

    @abstractmethod
    async def get_funding_rate(
        self,
        symbol: str,
        **params
    ) -> FundingRate:
        """
        获取资金费率

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            FundingRate: 资金费率信息
        """
        pass

    @abstractmethod
    async def get_open_interest(
        self,
        symbol: str,
        **params
    ) -> Optional[Decimal]:
        """
        获取持仓量（未平仓合约）

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            Optional[Decimal]: 持仓量（以USD计价），如果无法获取则返回None
        """
        pass

    @abstractmethod
    async def get_latest_price(
        self,
        symbol: str,
        **params
    ) -> LatestPrice:
        """
        获取最新价格信息

        包括最新成交价、标记价格、指数价格等

        Args:
            symbol: 交易对
            **params: 其他参数

        Returns:
            LatestPrice: 最新价格信息
        """
        pass

    @abstractmethod
    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
        **params
    ) -> OrderBook:
        """
        获取订单簿（盘口数据）

        Args:
            symbol: 交易对
            limit: 深度档位数量
            **params: 其他参数

        Returns:
            OrderBook: 订单簿数据
        """
        pass

    # ========== WebSocket 订阅 (可选) ==========

    async def subscribe_user_updates(
        self,
        on_order_update: Optional[Callable[[OrderUpdateEvent], Awaitable[None]]] = None,
        on_position_update: Optional[Callable[[PositionUpdateEvent], Awaitable[None]]] = None,
        on_trade_update: Optional[Callable[[TradeUpdateEvent], Awaitable[None]]] = None,
        on_account_update: Optional[Callable[[AccountUpdateEvent], Awaitable[None]]] = None,
        **params
    ) -> bool:
        """
        订阅用户数据更新（WebSocket）

        订阅用户的订单、仓位、成交、账户等实时更新。
        这是一个可选方法，不是所有交易所都需要实现。

        Args:
            on_order_update: 订单更新回调
            on_position_update: 仓位更新回调
            on_trade_update: 成交更新回调
            on_account_update: 账户更新回调
            **params: 其他参数

        Returns:
            bool: 是否订阅成功

        Note:
            - 默认实现返回 False（不支持）
            - 支持 WebSocket 的交易所应重写此方法
            - 回调函数应该是异步的 (async def)
        """
        return False

    async def unsubscribe_user_updates(self, **params) -> bool:
        """
        取消订阅用户数据更新

        Returns:
            bool: 是否取消成功
        """
        return False

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"