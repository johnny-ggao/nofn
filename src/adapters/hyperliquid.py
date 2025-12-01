"""
Hyperliquid 交易所适配器 - 优化重构版
"""
from typing import List, Optional, Dict, Any, Callable, Awaitable
from decimal import Decimal
import decimal
from datetime import datetime
import ccxt.async_support as ccxt
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
import eth_account
from termcolor import cprint

from .base import BaseExchangeAdapter
from .hyperliquid_websocket import HyperliquidWebSocketManager
from ..models import (
    Position,
    ExecutionResult,
    Balance,
    PositionSide,
    OrderType,
    ExecutionStatus,
    TradingAction,
    Candle,
    Ticker24h,
    FundingRate,
    LatestPrice,
    OrderBook,
    Order,
    Trade,
    OrderStatus,
    OrderUpdateEvent,
    PositionUpdateEvent,
    TradeUpdateEvent,
    AccountUpdateEvent,
)


class HyperliquidAdapter(BaseExchangeAdapter):
    """
    Hyperliquid 交易所适配器

    代码组织:
    - 第1部分: 初始化和连接管理
    - 第2部分: 交易操作 (开仓、平仓、止盈止损、取消订单)
    - 第3部分: 查询操作 (持仓、余额、订单、成交)
    - 第4部分: 行情数据 (K线、ticker、资金费率等)
    - 第5部分: 辅助方法 (私有方法)
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        super().__init__(api_key, api_secret, **kwargs)
        self._hl_exchange = None
        self._hl_info = None
        self._ws_manager: Optional[HyperliquidWebSocketManager] = None

    # ==================== 第1部分: 初始化和连接管理 ====================

    async def initialize(self) -> None:
        """
        初始化 Hyperliquid 交易所连接

        同时初始化:
        1. CCXT 客户端 - 用于查询操作
        2. Hyperliquid SDK 客户端 - 用于订单操作
        3. Hyperliquid Info API - 用于市场数据
        """
        try:
            # 1. 初始化 CCXT 客户端
            ccxt_config = {
                'walletAddress': self.api_key,
                'privateKey': self.api_secret,
                'enableRateLimit': True,
            }

            for key, value in self.config.items():
                if key not in ['testnet']:
                    ccxt_config[key] = value

            self._exchange = ccxt.hyperliquid(ccxt_config)
            await self._exchange.load_markets()

            # 2. 初始化 Hyperliquid SDK 客户端
            base_url = self.config.get('base_url', constants.MAINNET_API_URL)
            wallet = eth_account.Account.from_key(self.api_secret)

            self._hl_exchange = Exchange(
                wallet=wallet,
                base_url=base_url,
                account_address=self.api_key
            )

            # 3. 初始化 Hyperliquid Info API
            self._hl_info = Info(base_url=base_url, skip_ws=False)  # 允许 WebSocket

            # 4. 创建 WebSocket 管理器（但不自动订阅）
            self._ws_manager = HyperliquidWebSocketManager(
                hl_info=self._hl_info,
                user_address=self.api_key,
            )

        except Exception as e:
            raise ConnectionError(f"Hyperliquid 初始化失败: {str(e)}")

    async def close(self) -> None:
        """关闭交易所连接"""
        # 取消 WebSocket 订阅
        if self._ws_manager:
            await self._ws_manager.unsubscribe()

        if self._exchange:
            await self._exchange.close()

    # ==================== 第2部分: 交易操作 ====================

    async def open_position(
        self,
        symbol: str,
        side: PositionSide,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        leverage: int = 1,
        **params
    ) -> ExecutionResult:
        """
        开仓 (纯粹的开仓，不含止盈止损)

        Args:
            symbol: 交易对
            side: 持仓方向
            amount: 开仓数量
            order_type: 订单类型
            price: 限价单价格
            leverage: 杠杆倍数
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            await self.set_leverage(symbol, leverage)
            coin = self._symbol_to_coin(symbol)
            is_buy = side == PositionSide.LONG

            # 下单
            result = self._place_order(coin, is_buy, amount, order_type, price, params)

            # 检查结果
            if result.get('status') != 'ok':
                raise Exception(self._extract_error_message(result))

            # 解析订单结果
            order_id, executed_amount, executed_price = self._parse_order_response(result)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                order_id=order_id or '',
                symbol=symbol,
                executed_amount=executed_amount,
                executed_price=executed_price,
                fee=Decimal("0"),
                message=f"成功开{'多' if side == PositionSide.LONG else '空'}仓",
                raw_response=result,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return self._build_error_result(
                TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                symbol,
                "开仓失败",
                str(e)
            )

    async def open_position_with_sl_tp(
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
        开仓并设置止盈止损 (组合方法)

        这是一个便捷方法，组合了开仓和设置止盈止损两个操作
        """
        # 先开仓
        result = await self.open_position(symbol, side, amount, order_type, price, leverage, **params)

        if result.status != ExecutionStatus.SUCCESS:
            return result

        # 如果需要设置止盈止损
        if stop_loss or take_profit:
            try:
                rounded_sl = self._round_price(stop_loss)
                rounded_tp = self._round_price(take_profit)

                cprint(f"📝 开仓后设置止盈止损: SL={rounded_sl}, TP={rounded_tp}", "cyan")

                position = await self.get_position(symbol)
                if position:
                    sl_tp_result = await self.modify_stop_loss_take_profit(
                        position=position.model_dump(),
                        stop_loss=rounded_sl,
                        take_profit=rounded_tp
                    )

                    if sl_tp_result.status == ExecutionStatus.SUCCESS:
                        cprint(f"✓ 止盈止损设置成功", "green")
                    else:
                        cprint(f"⚠️ 止盈止损设置失败: {sl_tp_result.message}", "yellow")
                else:
                    cprint(f"⚠️ 未找到持仓，跳过止盈止损设置", "yellow")

            except Exception as e:
                cprint(f"❌ 设置止盈止损时出错: {str(e)}", "red")
                cprint(f"   开仓已成功，但止盈止损未设置", "yellow")

        return result

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
            position_id: 持仓 ID (未使用)
            amount: 平仓数量 (None 表示全部平仓)
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 获取当前持仓
            position = await self.get_position(symbol)
            if not position:
                return self._build_error_result(
                    TradingAction.CLOSE_POSITION,
                    symbol,
                    "未找到持仓",
                    "No position found"
                )

            # 确定平仓数量和方向
            close_amount = float(amount) if amount else float(position.amount)
            side = 'sell' if position.side == PositionSide.LONG else 'buy'

            # 获取当前市场价格
            ticker = await self._exchange.fetch_ticker(symbol)
            current_price = float(ticker.get('last', 0))

            # 创建市价平仓单
            result = await self._exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=close_amount,
                price=current_price,
                params={
                    'reduceOnly': True,
                    'user': self._exchange.walletAddress
                }
            )

            # 解析结果
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.CLOSE_POSITION,
                order_id=str(result.get('id', '')),
                symbol=symbol,
                executed_amount=self._safe_decimal(result.get('filled', 0)),
                executed_price=self._safe_decimal_optional(result.get('average')),
                fee=self._extract_fee(result),
                message="成功平仓",
                raw_response=result,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return self._build_error_result(
                TradingAction.CLOSE_POSITION,
                symbol,
                "平仓失败",
                str(e)
            )

    async def modify_stop_loss_take_profit(
        self,
        position: Dict[str, Any],
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        修改止损止盈

        Args:
            position: 持仓信息字典
            stop_loss: 止损价格
            take_profit: 止盈价格
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            symbol = position['symbol']
            position_amount = float(position['amount'])

            # 验证持仓数量
            if position_amount == 0:
                return self._build_error_result(
                    TradingAction.MODIFY_SL_TP,
                    symbol,
                    "持仓数量为0",
                    "Position size is 0"
                )

            # 四舍五入价格
            new_sl = self._round_price(stop_loss)
            new_tp = self._round_price(take_profit)

            # 检查是否需要修改
            if not self._should_modify_sl_tp(position, new_sl, new_tp):
                cprint(f"✓ SL/TP unchanged for {symbol}, skipping modification", "yellow")
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    action=TradingAction.MODIFY_SL_TP,
                    symbol=symbol,
                    message="止盈止损未改变，跳过修改",
                    timestamp=datetime.now(),
                )

            # 取消旧的止盈止损订单
            cancelled_orders = await self._cancel_sl_tp_orders(symbol)

            # 创建新的止盈止损订单
            new_orders = []
            coin = self._symbol_to_coin(symbol)
            is_buy = self._get_close_direction(position['side'])

            if new_sl is not None:
                sl_order = self._create_sl_order(coin, is_buy, position_amount, new_sl)
                if sl_order:
                    new_orders.append(sl_order)

            if new_tp is not None:
                tp_order = self._create_tp_order(coin, is_buy, position_amount, new_tp)
                if tp_order:
                    new_orders.append(tp_order)

            # 构建结果
            message = "成功设置止盈止损"
            if cancelled_orders:
                message += f"（已取消 {len(cancelled_orders)} 个旧订单）"

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.MODIFY_SL_TP,
                symbol=symbol,
                message=message,
                raw_response={'cancelled_orders': cancelled_orders, 'new_orders': new_orders},
                timestamp=datetime.now(),
            )

        except Exception as e:
            return self._build_error_result(
                TradingAction.MODIFY_SL_TP,
                position['symbol'],
                "修改止损止盈失败",
                str(e)
            )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        **params
    ) -> ExecutionResult:
        """
        取消订单

        Args:
            order_id: 订单 ID
            symbol: 交易对
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            coin = self._symbol_to_coin(symbol)
            result = self._hl_exchange.cancel(coin, int(order_id))

            if result.get('status') != 'ok':
                raise Exception(self._extract_error_message(result))

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.CANCEL_ORDER,
                order_id=order_id,
                symbol=symbol,
                message="成功取消订单",
                raw_response=result,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=TradingAction.CANCEL_ORDER,
                order_id=order_id,
                symbol=symbol,
                message="取消订单失败",
                error=str(e),
                timestamp=datetime.now(),
            )

    # ==================== 第3部分: 查询操作 ====================

    async def get_positions(
        self,
        symbol: Optional[str] = None,
        **params
    ) -> List[Position]:
        """获取持仓列表 (包含止盈止损信息)"""
        try:
            symbols = [symbol] if symbol else []
            positions_data = await self._exchange.fetch_positions(
                symbols=symbols,
                params={'user': self._exchange.walletAddress, **params}
            )

            positions = []
            for pos in positions_data:
                if float(pos.get('contracts', 0)) == 0:
                    continue

                # 查询止盈止损
                stop_loss, take_profit = await self._fetch_sl_tp_for_position(pos['symbol'])

                position = Position(
                    position_id=str(pos.get('id', '')),
                    symbol=pos['symbol'],
                    side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                    amount=Decimal(str(pos.get('contracts', 0))),
                    entry_price=Decimal(str(pos.get('entryPrice', 0))),
                    mark_price=Decimal(str(pos.get('markPrice', 0))) if pos.get('markPrice') else None,
                    liquidation_price=Decimal(str(pos.get('liquidationPrice', 0))) if pos.get('liquidationPrice') else None,
                    unrealized_pnl=Decimal(str(pos.get('unrealizedPnl', 0))) if pos.get('unrealizedPnl') is not None else None,
                    leverage=int(pos.get('leverage', 1)),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    opened_at=datetime.fromtimestamp(pos['timestamp'] / 1000) if pos.get('timestamp') else datetime.now(),
                    raw_data=pos,
                )
                positions.append(position)

            return positions

        except Exception as e:
            return []

    async def get_position(
        self,
        symbol: str,
        **params
    ) -> Optional[Position]:
        """获取指定交易对的持仓"""
        positions = await self.get_positions(symbol=symbol)
        return positions[0] if positions else None

    async def get_balance(
        self,
        currency: Optional[str] = None,
        **params
    ) -> Balance:
        """获取账户余额"""
        try:
            balance_data = await self._exchange.fetch_balance(
                params={'user': self._exchange.walletAddress, **params}
            )

            currency = currency or 'USDC'
            currency_balance = balance_data.get(currency, {})

            return Balance(
                currency=currency,
                total=Decimal(str(currency_balance.get('total', 0))),
                available=Decimal(str(currency_balance.get('free', 0))),
                frozen=Decimal(str(currency_balance.get('used', 0))),
                timestamp=datetime.now(),
                raw_data=balance_data,
            )

        except Exception as e:
            raise Exception(f"获取余额失败: {str(e)}")

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        **params
    ) -> List[Order]:
        """获取当前委托订单"""
        try:
            orders_data = await self._exchange.fetch_open_orders(
                symbol,
                params={'user': self._exchange.walletAddress, **params}
            )

            return [self._parse_order(order_dict) for order_dict in orders_data]

        except Exception as e:
            raise Exception(f"获取委托订单失败: {str(e)}")

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Order]:
        """获取历史订单"""
        try:
            if self._exchange.has.get('fetchOrders'):
                orders_data = await self._exchange.fetch_orders(
                    symbol=symbol,
                    since=since,
                    limit=limit,
                    params={'user': self._exchange.walletAddress, **params}
                )
            elif self._exchange.has.get('fetchClosedOrders'):
                orders_data = await self._exchange.fetch_closed_orders(
                    symbol=symbol,
                    since=since,
                    limit=limit,
                    params=params
                )
            else:
                raise NotImplementedError("交易所不支持获取历史订单")

            return [self._parse_order(order_dict) for order_dict in orders_data]

        except Exception as e:
            raise Exception(f"获取历史订单失败: {str(e)}")

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Trade]:
        """获取历史成交记录"""
        try:
            # Hyperliquid 的 fetch_my_trades 需要 symbol 参数
            # user 参数应该使用 walletAddress
            trades_data = await self._exchange.fetch_my_trades(
                symbol=symbol,
                since=since,
                limit=limit,
                params={'user': self.api_key, **params}  # api_key 就是 walletAddress
            )

            return [self._parse_trade(trade_dict) for trade_dict in trades_data]

        except Exception as e:
            raise Exception(f"获取成交记录失败: {str(e)}")

    async def get_ticker(
        self,
        symbol: str,
        **params
    ) -> Dict[str, Any]:
        """获取行情信息"""
        try:
            return await self._exchange.fetch_ticker(symbol, params)
        except Exception as e:
            raise Exception(f"获取行情失败: {str(e)}")

    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        **params
    ) -> bool:
        """设置杠杆"""
        try:
            await self._exchange.set_leverage(leverage, symbol, params)
            return True
        except Exception as e:
            return False

    # ==================== 第4部分: 行情数据 ====================

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List[Candle]:
        """获取K线数据"""
        try:
            if since is None:
                since = self._calculate_since(timeframe, limit)

            ohlcv_data = await self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
                params=params
            )

            return [self._parse_candle(candle) for candle in ohlcv_data]

        except Exception as e:
            raise Exception(f"获取K线数据失败: {str(e)}")

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        **params
    ) -> List[Candle]:
        """获取 K 线数据 (别名方法)"""
        return await self.get_candles(symbol, timeframe, None, limit, **params)

    async def get_ticker_24h(
        self,
        symbol: str,
        **params
    ) -> Ticker24h:
        """获取24小时行情统计"""
        try:
            ticker = await self._exchange.fetch_ticker(symbol, params)

            return Ticker24h(
                symbol=symbol,
                last_price=self._safe_decimal(ticker.get('last')),
                high_price=self._safe_decimal(ticker.get('high')),
                low_price=self._safe_decimal(ticker.get('low')),
                volume=self._safe_decimal(ticker.get('baseVolume')),
                quote_volume=self._safe_decimal(ticker.get('quoteVolume')),
                price_change=self._safe_decimal_optional(ticker.get('change')),
                price_change_percent=self._safe_decimal_optional(ticker.get('percentage')),
                bid_price=self._safe_decimal_optional(ticker.get('bid')),
                bid_qty=self._safe_decimal_optional(ticker.get('bidVolume')),
                ask_price=self._safe_decimal_optional(ticker.get('ask')),
                ask_qty=self._safe_decimal_optional(ticker.get('askVolume')),
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
                raw_data=ticker,
            )

        except Exception as e:
            raise Exception(f"获取24小时行情失败: {str(e)}")

    async def get_funding_rate(
        self,
        symbol: str,
        **params
    ) -> FundingRate:
        """获取资金费率"""
        try:
            funding_rate_data = await self._exchange.fetch_funding_rate(symbol, params)

            return FundingRate(
                symbol=symbol,
                funding_rate=Decimal(str(funding_rate_data.get('fundingRate', 0))),
                next_funding_rate=Decimal(str(funding_rate_data.get('nextFundingRate', 0))) if funding_rate_data.get('nextFundingRate') is not None else None,
                funding_time=datetime.fromtimestamp(funding_rate_data['fundingTimestamp'] / 1000) if funding_rate_data.get('fundingTimestamp') else datetime.now(),
                next_funding_time=datetime.fromtimestamp(funding_rate_data['nextFundingTimestamp'] / 1000) if funding_rate_data.get('nextFundingTimestamp') else None,
                mark_price=Decimal(str(funding_rate_data.get('markPrice', 0))) if funding_rate_data.get('markPrice') else None,
                index_price=Decimal(str(funding_rate_data.get('indexPrice', 0))) if funding_rate_data.get('indexPrice') else None,
                raw_data=funding_rate_data,
            )

        except Exception as e:
            raise Exception(f"获取资金费率失败: {str(e)}")

    async def get_open_interest(
        self,
        symbol: str,
        **params
    ) -> Optional[Decimal]:
        """获取持仓量"""
        try:
            coin = self._symbol_to_coin(symbol)
            data = self._hl_info.meta_and_asset_ctxs()

            if not isinstance(data, list) or len(data) < 2:
                return None

            meta_dict = data[0]
            asset_ctxs = data[1]

            if not isinstance(meta_dict, dict) or 'universe' not in meta_dict:
                return None

            universe = meta_dict['universe']

            # 查找币种索引
            coin_index = None
            for i, asset_info in enumerate(universe):
                if asset_info.get('name') == coin:
                    coin_index = i
                    break

            if coin_index is None or coin_index >= len(asset_ctxs):
                return None

            ctx = asset_ctxs[coin_index]
            open_interest = ctx.get('openInterest')

            if open_interest is not None:
                return Decimal(str(open_interest))

            return None

        except Exception as e:
            import traceback
            cprint(f"⚠️  获取持仓量失败 ({symbol}): {str(e)}", "yellow")
            cprint(f"   详细: {traceback.format_exc()}", "yellow")
            return None

    async def get_latest_price(
        self,
        symbol: str,
        **params
    ) -> LatestPrice:
        """获取最新价格信息"""
        try:
            ticker = await self._exchange.fetch_ticker(symbol, params)

            # 尝试获取标记价格和指数价格
            mark_price = None
            index_price = None
            try:
                funding_data = await self._exchange.fetch_funding_rate(symbol, params)
                mark_price = Decimal(str(funding_data.get('markPrice', 0))) if funding_data.get('markPrice') else None
                index_price = Decimal(str(funding_data.get('indexPrice', 0))) if funding_data.get('indexPrice') else None
            except:
                pass

            return LatestPrice(
                symbol=symbol,
                last_price=Decimal(str(ticker.get('last', 0))),
                mark_price=mark_price,
                index_price=index_price,
                bid_price=Decimal(str(ticker.get('bid', 0))) if ticker.get('bid') else None,
                ask_price=Decimal(str(ticker.get('ask', 0))) if ticker.get('ask') else None,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000) if ticker.get('timestamp') else datetime.now(),
                raw_data=ticker,
            )

        except Exception as e:
            raise Exception(f"获取最新价格失败: {str(e)}")

    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
        **params
    ) -> OrderBook:
        """获取订单簿"""
        try:
            order_book_data = await self._exchange.fetch_order_book(symbol, limit, params)

            bids = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book_data.get('bids', [])]
            asks = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book_data.get('asks', [])]

            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(order_book_data['timestamp'] / 1000) if order_book_data.get('timestamp') else datetime.now(),
                raw_data=order_book_data,
            )

        except Exception as e:
            raise Exception(f"获取订单簿失败: {str(e)}")

    async def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取资金费率历史数据"""
        try:
            coin = self._symbol_to_coin(symbol)

            if end_time is None:
                end_time = int(datetime.now().timestamp() * 1000)
            if start_time is None:
                start_time = end_time - (24 * 60 * 60 * 1000)

            result = self._hl_info.funding_history(coin, startTime=start_time, endTime=end_time)

            if not result:
                return []

            funding_history = []
            for item in result:
                funding_history.append({
                    'time': item.get('time'),
                    'funding_rate': float(item.get('fundingRate', 0)),
                    'premium': float(item.get('premium', 0)) if 'premium' in item else None
                })

            return funding_history

        except Exception as e:
            cprint(f"⚠️  获取资金费率历史失败: {e}", "yellow")
            return []

    # ==================== 第5部分: 辅助方法 (私有方法) ====================

    def _place_order(
        self,
        coin: str,
        is_buy: bool,
        amount: Decimal,
        order_type: OrderType,
        price: Optional[Decimal],
        params: Dict
    ) -> Dict:
        """下单的核心逻辑"""
        if order_type == OrderType.MARKET:
            slippage = params.get('slippage', 0.01)
            return self._hl_exchange.market_open(
                coin, is_buy, float(amount), None, slippage
            )
        elif order_type == OrderType.LIMIT:
            if not price:
                raise ValueError("限价单必须提供价格")
            tif = params.get('tif', 'Gtc')
            return self._hl_exchange.order(
                coin, is_buy, float(amount), float(price), {"limit": {"tif": tif}}
            )
        else:
            raise ValueError(f"不支持的订单类型: {order_type}")

    def _parse_order_response(self, result: Dict):
        """解析订单响应"""
        statuses = result.get('response', {}).get('data', {}).get('statuses', [])
        if not statuses:
            raise Exception("未获取到订单状态")

        status_data = statuses[0]
        order_id = None
        executed_amount = Decimal("0")
        executed_price = None

        if 'filled' in status_data:
            filled_data = status_data['filled']
            order_id = str(filled_data.get('oid', ''))
            executed_amount = Decimal(str(filled_data.get('totalSz', 0)))
            executed_price = Decimal(str(filled_data.get('avgPx', 0)))
        elif 'resting' in status_data:
            resting_data = status_data['resting']
            order_id = str(resting_data.get('oid', ''))

        return order_id, executed_amount, executed_price

    async def _cancel_sl_tp_orders(self, symbol: str) -> List[str]:
        """取消止盈止损订单"""
        cprint(f"🔍 查询 {symbol} 的未完成订单...", "cyan")

        open_orders = await self._exchange.fetch_open_orders(
            symbol=symbol,
            params={'user': self._exchange.walletAddress}
        )

        cancelled_orders = []
        cprint(f"📋 发现 {len(open_orders)} 个未完成订单", "cyan")

        for order in open_orders:
            if self._is_sl_tp_order(order):
                order_id = order.get('id')
                cprint(f"  🎯 识别到止盈止损订单: ID={order_id}, Type={order.get('type', '').lower()}", "yellow")

                try:
                    coin = self._symbol_to_coin(symbol)
                    cancel_result = self._hl_exchange.cancel(coin, int(order_id))

                    if cancel_result.get('status') == 'ok':
                        cancelled_orders.append(order_id)
                        cprint(f"  ✓ 成功取消订单: {order_id}", "green")
                    else:
                        cprint(f"  ❌ 取消失败: {cancel_result}", "red")

                except Exception as e:
                    cprint(f"  ⚠️ 取消订单 {order_id} 时出错: {str(e)} (可能已触发)", "yellow")

        if cancelled_orders:
            cprint(f"✓ 成功取消 {len(cancelled_orders)} 个旧的止盈止损订单", "green")

        return cancelled_orders

    def _create_sl_order(
        self,
        coin: str,
        is_buy: bool,
        amount: float,
        price: float
    ) -> Optional[Dict[str, Any]]:
        """创建止损订单"""
        try:
            order_type = {
                "trigger": {
                    "triggerPx": price,
                    "isMarket": True,
                    "tpsl": "sl"
                }
            }

            result = self._hl_exchange.order(
                coin, is_buy, amount, price, order_type, reduce_only=True
            )

            if result.get('status') != 'ok':
                raise Exception("Order API returned non-ok status")

            statuses = result.get('response', {}).get('data', {}).get('statuses', [])
            if statuses and 'error' in statuses[0]:
                error_msg = statuses[0].get('error', 'Unknown error')
                raise Exception(f"Stop loss order failed: {error_msg}")

            if statuses and 'resting' in statuses[0]:
                order_id = str(statuses[0]['resting'].get('oid', ''))
                return {'type': 'stop_loss', 'order_id': order_id, 'price': price}

        except Exception as e:
            cprint(f"❌ 创建止损订单失败: {e}", 'red')
            raise

        return None

    def _create_tp_order(
        self,
        coin: str,
        is_buy: bool,
        amount: float,
        price: float
    ) -> Optional[Dict[str, Any]]:
        """创建止盈订单"""
        try:
            order_type = {
                "trigger": {
                    "triggerPx": price,
                    "isMarket": True,
                    "tpsl": "tp"
                }
            }

            result = self._hl_exchange.order(
                coin, is_buy, amount, price, order_type, reduce_only=True
            )

            if result.get('status') != 'ok':
                raise Exception("Order API returned non-ok status")

            statuses = result.get('response', {}).get('data', {}).get('statuses', [])
            if statuses and 'error' in statuses[0]:
                error_msg = statuses[0].get('error', 'Unknown error')
                raise Exception(f"Take profit order failed: {error_msg}")

            if statuses and 'resting' in statuses[0]:
                order_id = str(statuses[0]['resting'].get('oid', ''))
                return {'type': 'take_profit', 'order_id': order_id, 'price': price}

        except Exception as e:
            cprint(f"❌ 创建止盈订单失败: {e}", 'red')
            raise

        return None

    async def _fetch_sl_tp_for_position(self, symbol: str):
        """查询持仓的止盈止损"""
        try:
            open_orders = await self._exchange.fetch_open_orders(
                symbol=symbol,
                params={'user': self._exchange.walletAddress}
            )

            stop_loss = None
            take_profit = None

            for order in open_orders:
                is_sl_tp = order['info'].get('isTrigger')

                if is_sl_tp:
                    trigger_price = order['info'].get('triggerPx')
                    if trigger_price:
                        trigger_price_decimal = Decimal(str(trigger_price))
                        order_type = order['info'].get('orderType', '').lower()

                        if order_type.startswith('stop'):
                            stop_loss = trigger_price_decimal
                        elif order_type.startswith('take profit'):
                            take_profit = trigger_price_decimal

            return stop_loss, take_profit

        except Exception as e:
            return None, None

    def _parse_order(self, order_dict: Dict[str, Any]) -> Order:
        """将 CCXT 订单数据转换为 Order 模型"""
        status = self._parse_order_status(order_dict)
        order_type = self._parse_order_type(order_dict)
        side = PositionSide.LONG if order_dict.get('side') == 'buy' else PositionSide.SHORT
        fee_cost, fee_currency = self._parse_fee(order_dict.get('fee'))

        return Order(
            order_id=str(order_dict.get('id', '')),
            symbol=order_dict.get('symbol', ''),
            order_type=order_type,
            side=side,
            amount=self._safe_decimal(order_dict.get('amount')),
            price=self._safe_decimal_optional(order_dict.get('price')),
            average_price=self._safe_decimal_optional(order_dict.get('average')),
            filled=self._safe_decimal(order_dict.get('filled', 0)),
            remaining=self._safe_decimal_optional(order_dict.get('remaining')),
            status=status,
            fee=fee_cost,
            fee_currency=fee_currency,
            reduce_only=order_dict.get('reduceOnly', False),
            post_only=order_dict.get('postOnly', False),
            created_at=datetime.fromtimestamp(order_dict['timestamp'] / 1000) if order_dict.get('timestamp') else datetime.now(),
            updated_at=datetime.fromtimestamp(order_dict['lastTradeTimestamp'] / 1000) if order_dict.get('lastTradeTimestamp') else None,
            raw_data=order_dict,
        )

    def _parse_trade(self, trade_dict: Dict[str, Any]) -> Trade:
        """将 CCXT 成交数据转换为 Trade 模型"""
        side = PositionSide.LONG if trade_dict.get('side') == 'buy' else PositionSide.SHORT
        fee_cost, fee_currency = self._parse_fee(trade_dict.get('fee'))

        # 从 info 中提取开平仓信息 (Hyperliquid 特有)
        info = trade_dict.get('info', {})
        start_position = self._safe_decimal_optional(info.get('startPosition'))
        closed_pnl = self._safe_decimal_optional(info.get('closedPnl'))

        # 判断交易类型
        trade_type = self._determine_trade_type(start_position, closed_pnl, side)

        return Trade(
            trade_id=str(trade_dict.get('id', '')),
            order_id=str(trade_dict.get('order', '')),
            symbol=trade_dict.get('symbol', ''),
            side=side,
            amount=self._safe_decimal(trade_dict.get('amount')),
            price=self._safe_decimal(trade_dict.get('price')),
            trade_type=trade_type,
            closed_pnl=closed_pnl,
            start_position=start_position,
            fee=fee_cost,
            fee_currency=fee_currency,
            is_maker=trade_dict.get('takerOrMaker') == 'maker' if trade_dict.get('takerOrMaker') else None,
            timestamp=datetime.fromtimestamp(trade_dict['timestamp'] / 1000) if trade_dict.get('timestamp') else datetime.now(),
            raw_data=trade_dict,
        )

    @staticmethod
    def _determine_trade_type(
        start_position: Optional[Decimal],
        closed_pnl: Optional[Decimal],
        side: PositionSide
    ) -> str:
        """
        判断交易类型：开仓、平仓、加仓、减仓

        逻辑：
        - startPosition 为 0 或 None -> 开仓
        - closedPnl 不为 0 -> 平仓（有盈亏说明是平仓）
        - startPosition 不为 0 且方向相同 -> 加仓
        - startPosition 不为 0 且方向相反 -> 减仓/平仓
        """
        if start_position is None or start_position == 0:
            return "open"

        if closed_pnl is not None and closed_pnl != 0:
            return "close"

        # 有持仓的情况下，根据方向判断
        # startPosition > 0 表示多仓，< 0 表示空仓
        if start_position > 0:
            # 原来是多仓
            return "add" if side == PositionSide.LONG else "reduce"
        else:
            # 原来是空仓
            return "add" if side == PositionSide.SHORT else "reduce"

    @staticmethod
    def _parse_order_status(order_dict: Dict[str, Any]) -> OrderStatus:
        """解析订单状态"""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.CLOSED,
            'canceled': OrderStatus.CANCELED,
            'cancelled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
        }

        ccxt_status = order_dict.get('status', 'open')
        filled = float(order_dict.get('filled', 0))
        amount = float(order_dict.get('amount', 0))

        if ccxt_status == 'open' and filled > 0 and filled < amount:
            return OrderStatus.PARTIALLY_FILLED

        return status_map.get(ccxt_status, OrderStatus.OPEN)

    @staticmethod
    def _parse_order_type(order_dict: Dict[str, Any]) -> OrderType:
        """解析订单类型"""
        order_type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop_market': OrderType.STOP_MARKET,
            'stop_limit': OrderType.STOP_LIMIT,
        }
        return order_type_map.get(order_dict.get('type', 'market'), OrderType.MARKET)

    @staticmethod
    def _parse_candle(candle: List) -> Candle:
        """解析K线数据"""
        return Candle(
            timestamp=datetime.fromtimestamp(candle[0] / 1000),
            open=Decimal(str(candle[1])),
            high=Decimal(str(candle[2])),
            low=Decimal(str(candle[3])),
            close=Decimal(str(candle[4])),
            volume=Decimal(str(candle[5])),
            raw_data={
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            }
        )

    @staticmethod
    def _is_sl_tp_order(order: Dict[str, Any]) -> bool:
        """判断是否为止盈止损订单"""
        order_type = order.get('type', '').lower()
        order_info = order.get('info', {})

        return (
            ('stop' in order_type or 'take_profit' in order_type or 'trigger' in order_type) or
            ('trigger' in order_info) or
            (order.get('reduceOnly') is True)
        )

    @staticmethod
    def _should_modify_sl_tp(position: Dict[str, Any], new_sl: Optional[float], new_tp: Optional[float]) -> bool:
        """检查是否需要修改止盈止损"""
        current_sl = position.get('stop_loss')
        current_tp = position.get('take_profit')

        if current_sl is not None:
            current_sl = float(current_sl)
        if current_tp is not None:
            current_tp = float(current_tp)

        sl_unchanged = (new_sl is None or new_sl == current_sl)
        tp_unchanged = (new_tp is None or new_tp == current_tp)

        return not (sl_unchanged and tp_unchanged)

    @staticmethod
    def _get_close_direction(position_side) -> bool:
        """获取平仓方向"""
        if isinstance(position_side, str):
            return position_side.lower() == "short"
        else:
            return position_side == PositionSide.SHORT

    @staticmethod
    def _round_price(price: Optional[Decimal]) -> Optional[float]:
        """四舍五入价格到1位小数 (Hyperliquid要求)"""
        return round(float(price), 1) if price is not None else None

    @staticmethod
    def _calculate_since(timeframe: str, limit: int) -> int:
        """计算起始时间戳"""
        import time
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200,
        }
        minutes = timeframe_minutes.get(timeframe, 60)
        return int((time.time() - limit * minutes * 60) * 1000)

    @staticmethod
    def _symbol_to_coin(symbol: str) -> str:
        """将交易对转换为币种"""
        if '/' in symbol:
            return symbol.split('/')[0]
        return symbol

    @staticmethod
    def _safe_decimal(value, default="0") -> Decimal:
        """安全地转换为 Decimal"""
        if value is None:
            return Decimal(default)
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, decimal.InvalidOperation):
            return Decimal(default)

    @staticmethod
    def _safe_decimal_optional(value) -> Optional[Decimal]:
        """安全地转换为可选的 Decimal"""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, decimal.InvalidOperation):
            return None

    @staticmethod
    def _parse_fee(fee_info: Optional[Dict[str, Any]]):
        """解析手续费信息"""
        fee_cost = None
        fee_currency = None

        if fee_info and isinstance(fee_info, dict):
            if fee_info.get('cost') is not None:
                try:
                    fee_cost = Decimal(str(fee_info['cost']))
                except:
                    pass
            fee_currency = fee_info.get('currency')

        return fee_cost, fee_currency

    @staticmethod
    def _extract_fee(result: Dict[str, Any]) -> Decimal:
        """从订单结果中提取手续费"""
        fee = Decimal("0")
        if result.get('fee') and result['fee'].get('cost'):
            try:
                fee = Decimal(str(result['fee']['cost']))
            except:
                pass
        return fee

    @staticmethod
    def _extract_error_message(result: Dict) -> str:
        """从结果中提取错误信息"""
        return result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('error', 'Unknown error')

    @staticmethod
    def _build_error_result(action: TradingAction, symbol: str, message: str, error: str) -> ExecutionResult:
        """构建错误结果"""
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            action=action,
            symbol=symbol,
            message=message,
            error=error,
            timestamp=datetime.now(),
        )