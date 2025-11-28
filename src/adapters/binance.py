"""
Binance 永续合约交易所适配器

支持:
- USDT 永续合约 (binance futures)
- USDC 永续合约 (binance futures)
"""
from typing import List, Optional, Dict, Any
from decimal import Decimal
import decimal
from datetime import datetime
import ccxt.async_support as ccxt
from termcolor import cprint

from .base import BaseExchangeAdapter
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
)


class BinanceAdapter(BaseExchangeAdapter):
    """
    Binance 永续合约适配器

    支持 USDT-M 和 USDC-M 永续合约

    代码组织:
    - 第1部分: 初始化和连接管理
    - 第2部分: 交易操作 (开仓、平仓、止盈止损、取消订单)
    - 第3部分: 查询操作 (持仓、余额、订单、成交)
    - 第4部分: 行情数据 (K线、ticker、资金费率等)
    - 第5部分: 辅助方法 (私有方法)
    """

    # 支持的保证金类型
    MARGIN_TYPES = ['USDT', 'USDC']

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        margin_type: str = 'USDT',
        **kwargs
    ):
        """
        初始化 Binance 适配器

        Args:
            api_key: API 密钥
            api_secret: API 私钥
            margin_type: 保证金类型 ('USDT' 或 'USDC')
            **kwargs: 其他配置参数
                - testnet: bool, 是否使用测试网
                - default_leverage: int, 默认杠杆倍数
                - position_mode: str, 持仓模式 ('one_way' 或 'hedge')
        """
        super().__init__(api_key, api_secret, **kwargs)

        if margin_type.upper() not in self.MARGIN_TYPES:
            raise ValueError(f"不支持的保证金类型: {margin_type}，支持: {self.MARGIN_TYPES}")

        self.margin_type = margin_type.upper()
        self.testnet = kwargs.get('testnet', False)
        self.default_leverage = kwargs.get('default_leverage', 1)
        self.position_mode = kwargs.get('position_mode', 'one_way')  # 'one_way' 或 'hedge'

        # 缓存市场信息
        self._markets_cache: Dict[str, Any] = {}
        self._leverage_cache: Dict[str, int] = {}

    # ==================== 第1部分: 初始化和连接管理 ====================

    async def initialize(self) -> None:
        """
        初始化 Binance 交易所连接

        根据 margin_type 选择不同的交易所实例:
        - USDT: binanceusdm (USDT-M 永续)
        - USDC: binanceusdm (同样使用 usdm，通过交易对区分)
        """
        try:
            ccxt_config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
            }

            # 设置测试网
            if self.testnet:
                ccxt_config['options']['sandboxMode'] = True
                cprint("⚠️  使用 Binance 测试网", "yellow")

            # 添加其他配置
            for key, value in self.config.items():
                if key not in ['testnet', 'default_leverage', 'position_mode', 'margin_type']:
                    ccxt_config[key] = value

            # 创建交易所实例 (USDT-M 和 USDC-M 都使用 binanceusdm)
            self._exchange = ccxt.binanceusdm(ccxt_config)

            # 加载市场
            await self._exchange.load_markets()

            # 设置持仓模式
            await self._set_position_mode()

            cprint(f"✅ Binance {self.margin_type}-M 永续合约连接成功", "green")

        except Exception as e:
            raise ConnectionError(f"Binance 初始化失败: {str(e)}")

    async def _set_position_mode(self) -> None:
        """设置持仓模式 (单向/双向)"""
        try:
            # 获取当前持仓模式
            result = await self._exchange.fapiPrivateGetPositionSideDual()
            current_mode = result.get('dualSidePosition', False)

            target_hedge = self.position_mode == 'hedge'

            if current_mode != target_hedge:
                await self._exchange.fapiPrivatePostPositionSideDual({
                    'dualSidePosition': 'true' if target_hedge else 'false'
                })
                mode_str = '双向持仓' if target_hedge else '单向持仓'
                cprint(f"✅ 已设置为 {mode_str} 模式", "green")

        except Exception as e:
            # 如果设置失败（可能因为有持仓），只记录警告
            cprint(f"⚠️  持仓模式设置失败 (可能存在持仓): {str(e)}", "yellow")

    async def close(self) -> None:
        """关闭交易所连接"""
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
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        开仓

        Args:
            symbol: 交易对 (如 "BTC/USDT:USDT" 或 "BTC/USDC:USDC")
            side: 持仓方向
            amount: 开仓数量
            order_type: 订单类型
            price: 限价单价格
            leverage: 杠杆倍数
            stop_loss: 止损价格
            take_profit: 止盈价格
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 标准化交易对
            symbol = self._normalize_symbol(symbol)

            # 设置杠杆
            await self.set_leverage(symbol, leverage)

            # 确定订单方向
            ccxt_side = 'buy' if side == PositionSide.LONG else 'sell'
            ccxt_type = 'market' if order_type == OrderType.MARKET else 'limit'

            # 构建订单参数
            order_params = {}

            # 双向持仓模式下需要指定持仓方向
            if self.position_mode == 'hedge':
                order_params['positionSide'] = 'LONG' if side == PositionSide.LONG else 'SHORT'

            # 下单
            result = await self._exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=float(amount),
                price=float(price) if price else None,
                params=order_params
            )

            # 解析结果
            order_id = str(result.get('id', ''))
            executed_amount = self._safe_decimal(result.get('filled', 0))
            executed_price = self._safe_decimal_optional(result.get('average'))
            fee = self._extract_fee(result)

            exec_result = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                order_id=order_id,
                symbol=symbol,
                executed_amount=executed_amount,
                executed_price=executed_price,
                fee=fee,
                message=f"成功开{'多' if side == PositionSide.LONG else '空'}仓",
                raw_response=result,
                timestamp=datetime.now(),
            )

            # 设置止盈止损
            if stop_loss or take_profit:
                await self._set_sl_tp_after_open(symbol, side, amount, stop_loss, take_profit)

            return exec_result

        except Exception as e:
            return self._build_error_result(
                TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                symbol,
                "开仓失败",
                str(e)
            )

    async def _set_sl_tp_after_open(
        self,
        symbol: str,
        side: PositionSide,
        amount: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal]
    ) -> None:
        """开仓后设置止盈止损"""
        try:
            position = await self.get_position(symbol)
            if position:
                await self.modify_stop_loss_take_profit(
                    position=position.model_dump(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                cprint("✅ 止盈止损设置成功", "green")
        except Exception as e:
            cprint(f"⚠️  止盈止损设置失败: {str(e)}", "yellow")

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
            symbol = self._normalize_symbol(symbol)

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
            ccxt_side = 'sell' if position.side == PositionSide.LONG else 'buy'

            # 构建订单参数
            order_params = {'reduceOnly': True}

            # 双向持仓模式
            if self.position_mode == 'hedge':
                order_params['positionSide'] = 'LONG' if position.side == PositionSide.LONG else 'SHORT'

            # 市价平仓
            result = await self._exchange.create_order(
                symbol=symbol,
                type='market',
                side=ccxt_side,
                amount=close_amount,
                params=order_params
            )

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

        Binance 使用条件订单实现止盈止损

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
            symbol = self._normalize_symbol(symbol)
            position_amount = abs(float(position['amount']))
            position_side = position['side']

            if isinstance(position_side, str):
                position_side = PositionSide.LONG if position_side.lower() == 'long' else PositionSide.SHORT

            if position_amount == 0:
                return self._build_error_result(
                    TradingAction.MODIFY_SL_TP,
                    symbol,
                    "持仓数量为0",
                    "Position size is 0"
                )

            # 先取消现有的止盈止损订单
            await self._cancel_sl_tp_orders(symbol)

            new_orders = []

            # 平仓方向
            close_side = 'sell' if position_side == PositionSide.LONG else 'buy'

            # 创建止损订单 (STOP_MARKET)
            if stop_loss:
                sl_order = await self._create_stop_order(
                    symbol=symbol,
                    side=close_side,
                    amount=position_amount,
                    stop_price=float(stop_loss),
                    position_side=position_side,
                    order_type='STOP_MARKET'
                )
                if sl_order:
                    new_orders.append(sl_order)

            # 创建止盈订单 (TAKE_PROFIT_MARKET)
            if take_profit:
                tp_order = await self._create_stop_order(
                    symbol=symbol,
                    side=close_side,
                    amount=position_amount,
                    stop_price=float(take_profit),
                    position_side=position_side,
                    order_type='TAKE_PROFIT_MARKET'
                )
                if tp_order:
                    new_orders.append(tp_order)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.MODIFY_SL_TP,
                symbol=symbol,
                message=f"成功设置止盈止损 (创建 {len(new_orders)} 个订单)",
                raw_response={'new_orders': new_orders},
                timestamp=datetime.now(),
            )

        except Exception as e:
            return self._build_error_result(
                TradingAction.MODIFY_SL_TP,
                position.get('symbol', ''),
                "修改止损止盈失败",
                str(e)
            )

    async def _create_stop_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        position_side: PositionSide,
        order_type: str  # 'STOP_MARKET' 或 'TAKE_PROFIT_MARKET'
    ) -> Optional[Dict[str, Any]]:
        """创建条件订单 (止损/止盈)"""
        try:
            params = {
                'stopPrice': stop_price,
                'reduceOnly': True,
                'type': order_type,
            }

            # 双向持仓模式
            if self.position_mode == 'hedge':
                params['positionSide'] = 'LONG' if position_side == PositionSide.LONG else 'SHORT'

            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side,
                amount=amount,
                params=params
            )

            order_type_name = '止损' if 'STOP' in order_type else '止盈'
            cprint(f"✅ {order_type_name}订单创建成功: {result.get('id')}", "green")

            return {
                'order_id': result.get('id'),
                'type': order_type,
                'stop_price': stop_price
            }

        except Exception as e:
            cprint(f"❌ 创建条件订单失败: {str(e)}", "red")
            return None

    async def _cancel_sl_tp_orders(self, symbol: str) -> List[str]:
        """取消所有止盈止损订单"""
        cancelled = []
        try:
            open_orders = await self._exchange.fetch_open_orders(symbol)

            for order in open_orders:
                order_type = order.get('type', '').upper()
                if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT']:
                    try:
                        await self._exchange.cancel_order(order['id'], symbol)
                        cancelled.append(order['id'])
                        cprint(f"✅ 取消订单: {order['id']}", "green")
                    except Exception as e:
                        cprint(f"⚠️  取消订单失败: {str(e)}", "yellow")

        except Exception as e:
            cprint(f"⚠️  查询订单失败: {str(e)}", "yellow")

        return cancelled

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
            symbol = self._normalize_symbol(symbol)
            result = await self._exchange.cancel_order(order_id, symbol, params)

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
        """获取持仓列表"""
        try:
            symbols = [self._normalize_symbol(symbol)] if symbol else None
            positions_data = await self._exchange.fetch_positions(symbols, params)

            positions = []
            for pos in positions_data:
                # 跳过空仓
                contracts = float(pos.get('contracts', 0))
                if contracts == 0:
                    continue

                # 查询止盈止损
                stop_loss, take_profit = await self._fetch_sl_tp_for_position(pos['symbol'])

                # 安全获取 leverage，处理 None 值
                leverage_val = pos.get('leverage')
                leverage = int(leverage_val) if leverage_val is not None else 1

                position = Position(
                    position_id=str(pos.get('id', '')),
                    symbol=pos['symbol'],
                    side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                    amount=Decimal(str(abs(contracts))),
                    entry_price=Decimal(str(pos.get('entryPrice', 0))),
                    mark_price=self._safe_decimal_optional(pos.get('markPrice')),
                    liquidation_price=self._safe_decimal_optional(pos.get('liquidationPrice')),
                    unrealized_pnl=self._safe_decimal_optional(pos.get('unrealizedPnl')),
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    opened_at=datetime.fromtimestamp(pos['timestamp'] / 1000) if pos.get('timestamp') else datetime.now(),
                    raw_data=pos,
                )
                positions.append(position)

            return positions

        except Exception as e:
            cprint(f"⚠️  获取持仓失败: {str(e)}", "yellow")
            return []

    async def get_position(
        self,
        symbol: str,
        **params
    ) -> Optional[Position]:
        """获取指定交易对的持仓"""
        positions = await self.get_positions(symbol=symbol, **params)
        return positions[0] if positions else None

    async def get_balance(
        self,
        currency: Optional[str] = None,
        **params
    ) -> Balance:
        """获取账户余额"""
        try:
            balance_data = await self._exchange.fetch_balance(params)

            # 默认使用配置的保证金类型
            currency = currency or self.margin_type
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
            symbol = self._normalize_symbol(symbol) if symbol else None
            orders_data = await self._exchange.fetch_open_orders(symbol, params=params)
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
            symbol = self._normalize_symbol(symbol) if symbol else None
            orders_data = await self._exchange.fetch_closed_orders(
                symbol=symbol,
                since=since,
                limit=limit,
                params=params
            )
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
            symbol = self._normalize_symbol(symbol) if symbol else None
            trades_data = await self._exchange.fetch_my_trades(
                symbol=symbol,
                since=since,
                limit=limit,
                params=params
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
            symbol = self._normalize_symbol(symbol)
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
            symbol = self._normalize_symbol(symbol)

            # 检查缓存，避免重复设置
            if self._leverage_cache.get(symbol) == leverage:
                return True

            await self._exchange.set_leverage(leverage, symbol, params)
            self._leverage_cache[symbol] = leverage
            return True

        except Exception as e:
            cprint(f"⚠️  设置杠杆失败: {str(e)}", "yellow")
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
            symbol = self._normalize_symbol(symbol)

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

    async def get_ticker_24h(
        self,
        symbol: str,
        **params
    ) -> Ticker24h:
        """获取24小时行情统计"""
        try:
            symbol = self._normalize_symbol(symbol)
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
            symbol = self._normalize_symbol(symbol)
            funding_rate_data = await self._exchange.fetch_funding_rate(symbol, params)

            return FundingRate(
                symbol=symbol,
                funding_rate=Decimal(str(funding_rate_data.get('fundingRate', 0))),
                next_funding_rate=self._safe_decimal_optional(funding_rate_data.get('nextFundingRate')),
                funding_time=datetime.fromtimestamp(funding_rate_data['fundingTimestamp'] / 1000) if funding_rate_data.get('fundingTimestamp') else datetime.now(),
                next_funding_time=datetime.fromtimestamp(funding_rate_data['nextFundingTimestamp'] / 1000) if funding_rate_data.get('nextFundingTimestamp') else None,
                mark_price=self._safe_decimal_optional(funding_rate_data.get('markPrice')),
                index_price=self._safe_decimal_optional(funding_rate_data.get('indexPrice')),
                raw_data=funding_rate_data,
            )

        except Exception as e:
            raise Exception(f"获取资金费率失败: {str(e)}")

    async def get_open_interest(
        self,
        symbol: str,
        **params
    ) -> Optional[Decimal]:
        """获取持仓量 (未平仓合约)"""
        try:
            symbol = self._normalize_symbol(symbol)

            # 使用 CCXT 的 fetchOpenInterest 方法
            oi_data = await self._exchange.fetch_open_interest(symbol, params)

            if oi_data and 'openInterestAmount' in oi_data:
                return Decimal(str(oi_data['openInterestAmount']))
            elif oi_data and 'openInterestValue' in oi_data:
                return Decimal(str(oi_data['openInterestValue']))

            return None

        except Exception as e:
            cprint(f"⚠️  获取持仓量失败 ({symbol}): {str(e)}", "yellow")
            return None

    async def get_open_interest_history(
        self,
        symbol: str,
        period: str = "4h",
        limit: int = 30,
        **params
    ) -> List[Dict[str, Any]]:
        """
        获取持仓量历史数据

        Args:
            symbol: 交易对 (如 "BTC/USDT:USDT" 或 "BTCUSDT")
            period: 时间周期 ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
            limit: 返回数据条数 (默认30，最大500)

        Returns:
            List[Dict]: 持仓量历史数据列表
            [
                {
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "12345.67",  # 持仓量 (合约数量)
                    "sumOpenInterestValue": "1234567890.12",  # 持仓量价值 (USD)
                    "timestamp": 1699876800000
                },
                ...
            ]
        """
        try:
            # 将 symbol 转换为 Binance 原生格式 (如 BTCUSDT)
            symbol = self._normalize_symbol(symbol)
            # 从 "BTC/USDT:USDT" 提取 "BTCUSDT"
            base_symbol = symbol.replace("/", "").replace(":USDT", "").replace(":USDC", "")

            # 直接调用 Binance Futures API
            response = await self._exchange.fapiDataGetOpenInterestHist({
                'symbol': base_symbol,
                'period': period,
                'limit': limit,
            })

            # 解析返回数据
            result = []
            for item in response:
                result.append({
                    'symbol': item.get('symbol'),
                    'sum_open_interest': float(item.get('sumOpenInterest', 0)),
                    'sum_open_interest_value': float(item.get('sumOpenInterestValue', 0)),
                    'timestamp': int(item.get('timestamp', 0)),
                })

            return result

        except Exception as e:
            cprint(f"⚠️  获取持仓量历史失败 ({symbol}): {str(e)}", "yellow")
            return []

    async def get_latest_price(
        self,
        symbol: str,
        **params
    ) -> LatestPrice:
        """获取最新价格信息"""
        try:
            symbol = self._normalize_symbol(symbol)
            ticker = await self._exchange.fetch_ticker(symbol, params)

            # 尝试获取标记价格和指数价格
            mark_price = None
            index_price = None
            try:
                funding_data = await self._exchange.fetch_funding_rate(symbol, params)
                mark_price = self._safe_decimal_optional(funding_data.get('markPrice'))
                index_price = self._safe_decimal_optional(funding_data.get('indexPrice'))
            except:
                pass

            return LatestPrice(
                symbol=symbol,
                last_price=Decimal(str(ticker.get('last', 0))),
                mark_price=mark_price,
                index_price=index_price,
                bid_price=self._safe_decimal_optional(ticker.get('bid')),
                ask_price=self._safe_decimal_optional(ticker.get('ask')),
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
            symbol = self._normalize_symbol(symbol)
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

    # ==================== 第5部分: 辅助方法 ====================

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化交易对格式

        支持多种输入格式:
        - "BTC" -> "BTC/USDT:USDT" 或 "BTC/USDC:USDC"
        - "BTCUSDT" -> "BTC/USDT:USDT"
        - "BTC/USDT" -> "BTC/USDT:USDT"
        - "BTC/USDT:USDT" -> "BTC/USDT:USDT" (保持不变)
        """
        if not symbol:
            raise ValueError("交易对不能为空")

        symbol = symbol.upper()

        # 已经是完整格式
        if ':' in symbol:
            return symbol

        # 处理 BTC/USDT 格式
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}/{quote}:{quote}"

        # 处理 BTCUSDT 或 BTCUSDC 格式
        for margin in ['USDT', 'USDC']:
            if symbol.endswith(margin):
                base = symbol[:-len(margin)]
                return f"{base}/{margin}:{margin}"

        # 只有币种名称，使用配置的保证金类型
        return f"{symbol}/{self.margin_type}:{self.margin_type}"

    async def _fetch_sl_tp_for_position(self, symbol: str):
        """查询持仓的止盈止损"""
        try:
            open_orders = await self._exchange.fetch_open_orders(symbol)

            stop_loss = None
            take_profit = None

            for order in open_orders:
                order_type = order.get('type', '').upper()
                stop_price = order.get('stopPrice') or order.get('info', {}).get('stopPrice')

                if stop_price:
                    price_decimal = Decimal(str(stop_price))
                    if order_type in ['STOP_MARKET', 'STOP']:
                        stop_loss = price_decimal
                    elif order_type in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                        take_profit = price_decimal

            return stop_loss, take_profit

        except Exception as e:
            return None, None

    def _parse_order(self, order_dict: Dict[str, Any]) -> Order:
        """将 CCXT 订单数据转换为 Order 模型"""
        status = self._parse_order_status(order_dict)
        order_type = self._parse_order_type(order_dict)
        side = PositionSide.LONG if order_dict.get('side') == 'buy' else PositionSide.SHORT
        fee_cost, fee_currency = self._parse_fee_info(order_dict.get('fee'))

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
        fee_cost, fee_currency = self._parse_fee_info(trade_dict.get('fee'))

        return Trade(
            trade_id=str(trade_dict.get('id', '')),
            order_id=str(trade_dict.get('order', '')),
            symbol=trade_dict.get('symbol', ''),
            side=side,
            amount=self._safe_decimal(trade_dict.get('amount')),
            price=self._safe_decimal(trade_dict.get('price')),
            fee=fee_cost,
            fee_currency=fee_currency,
            is_maker=trade_dict.get('takerOrMaker') == 'maker' if trade_dict.get('takerOrMaker') else None,
            timestamp=datetime.fromtimestamp(trade_dict['timestamp'] / 1000) if trade_dict.get('timestamp') else datetime.now(),
            raw_data=trade_dict,
        )

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
            'stop': OrderType.STOP_MARKET,
            'stop_limit': OrderType.STOP_LIMIT,
            'take_profit_market': OrderType.STOP_MARKET,
            'take_profit': OrderType.STOP_LIMIT,
        }
        return order_type_map.get(order_dict.get('type', 'market').lower(), OrderType.MARKET)

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
    def _parse_fee_info(fee_info: Optional[Dict[str, Any]]):
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

    def __repr__(self) -> str:
        return f"<BinanceAdapter margin_type={self.margin_type} testnet={self.testnet}>"
