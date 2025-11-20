"""
Hyperliquid 交易所适配器

实现 Hyperliquid 交易所的具体功能

注意：
- 使用 Hyperliquid SDK 进行订单操作（市价单、限价单、止盈止损单）
- 使用 CCXT 进行查询操作（持仓、余额、K线等）
"""
import json
from typing import List, Optional, Dict, Any
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


class HyperliquidAdapter(BaseExchangeAdapter):
    """
    Hyperliquid 交易所适配器

    基于 CCXT 实现 Hyperliquid 的交易功能
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        super().__init__(api_key, api_secret, **kwargs)
        self._hl_exchange = None

    @staticmethod
    def _safe_decimal(value, default="0") -> Decimal:
        """
        安全地将值转换为 Decimal

        Args:
            value: 要转换的值
            default: 默认值（当 value 为 None 时）

        Returns:
            Decimal: 转换后的值
        """
        if value is None:
            return Decimal(default)
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, decimal.InvalidOperation):
            return Decimal(default)

    @staticmethod
    def _safe_decimal_optional(value) -> Optional[Decimal]:
        """
        安全地将值转换为可选的 Decimal

        Args:
            value: 要转换的值

        Returns:
            Optional[Decimal]: 转换后的值，失败返回 None
        """
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, decimal.InvalidOperation):
            return None

    @staticmethod
    def _symbol_to_coin(symbol: str) -> str:
        """
        将 CCXT 格式的交易对转换为 Hyperliquid SDK 的币种格式

        Args:
            symbol: CCXT 交易对格式（如 "BTC/USDC:USDC" 或 "ETH/USDC:USDC"）

        Returns:
            str: Hyperliquid 币种格式（如 "BTC" 或 "ETH"）
        """
        # Hyperliquid SDK 使用币种名称而不是交易对
        # 例如: "BTC/USDC:USDC" -> "BTC", "ETH/USDC:USDC" -> "ETH"
        # 处理两种格式: "BTC/USDC:USDC" 和 "BTC/USDC"
        if '/' in symbol:
            return symbol.split('/')[0]
        return symbol

    async def initialize(self) -> None:
        """
        初始化 Hyperliquid 交易所连接

        同时初始化：
        1. CCXT 客户端 (self._exchange) - 用于查询操作
        2. Hyperliquid SDK 客户端 (self._hl_exchange) - 用于订单操作
        """
        try:
            # 1. 初始化 CCXT 客户端（用于查询）
            # 注意：不要传入 testnet 参数给 CCXT，它会导致缓存问题
            ccxt_config = {
                'walletAddress': self.api_key,
                'privateKey': self.api_secret,
                'enableRateLimit': True,
            }

            # 只传入 CCXT 支持的配置参数（排除 testnet）
            for key, value in self.config.items():
                if key not in ['testnet']:  # testnet 由 SDK 的 base_url 控制
                    ccxt_config[key] = value

            self._exchange = ccxt.hyperliquid(ccxt_config)
            await self._exchange.load_markets()

            # 2. 初始化 Hyperliquid SDK 客户端（用于下单）
            # 判断是否使用测试网
            base_url = self.config.get('base_url', constants.MAINNET_API_URL)

            # 创建 wallet 对象
            wallet = eth_account.Account.from_key(self.api_secret)

            # 创建 Exchange 实例
            self._hl_exchange = Exchange(
                wallet=wallet,                 # Wallet 对象
                base_url=base_url,             # API URL
                account_address=self.api_key   # 钱包地址
            )

        except Exception as e:
            raise ConnectionError(f"Hyperliquid 初始化失败: {str(e)}")

    async def close(self) -> None:
        """关闭交易所连接"""
        if self._exchange:
            await self._exchange.close()

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
        开仓 - 使用 Hyperliquid SDK

        Args:
            symbol: 交易对（如 "BTC/USDT"）
            side: 持仓方向
            amount: 开仓数量
            order_type: 订单类型（市价/限价）
            price: 限价单价格
            leverage: 杠杆倍数
            stop_loss: 止损价格
            take_profit: 止盈价格
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 设置杠杆
            if leverage > 1:
                await self.set_leverage(symbol, leverage)

            # 转换交易对格式
            coin = self._symbol_to_coin(symbol)

            # 确定订单方向
            is_buy = side == PositionSide.LONG

            # 下单
            if order_type == OrderType.MARKET:
                slippage = params.get('slippage', 0.01)
                result = self._hl_exchange.market_open(
                    coin,
                    is_buy,
                    float(amount),
                    None,  # px 参数，市价单为 None
                    slippage
                )

            elif order_type == OrderType.LIMIT:
                if not price:
                    raise ValueError("限价单必须提供价格")

                tif = params.get('tif', 'Gtc')  # Good til cancel
                result = self._hl_exchange.order(
                    coin,
                    is_buy,
                    float(amount),
                    float(price),
                    {"limit": {"tif": tif}}
                )
            else:
                raise ValueError(f"不支持的订单类型: {order_type}")

            # 检查结果
            if result.get('status') != 'ok':
                error_msg = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('error', 'Unknown error')
                raise Exception(error_msg)

            # 解析订单结果
            statuses = result.get('response', {}).get('data', {}).get('statuses', [])
            if not statuses:
                raise Exception("未获取到订单状态")

            status_data = statuses[0]

            # 获取订单 ID 和成交信息
            order_id = None
            executed_amount = Decimal("0")
            executed_price = None

            if 'filled' in status_data:
                # 订单已成交
                filled_data = status_data['filled']
                order_id = str(filled_data.get('oid', ''))
                executed_amount = Decimal(str(filled_data.get('totalSz', 0)))
                executed_price = Decimal(str(filled_data.get('avgPx', 0)))
            elif 'resting' in status_data:
                # 订单挂单中（限价单）
                resting_data = status_data['resting']
                order_id = str(resting_data.get('oid', ''))

            # 如果需要设置止损止盈，在开仓后设置
            sl_tp_result = None
            if (stop_loss or take_profit) and order_id:
                # 等待订单成交后设置止损止盈
                # 注意：这里可能需要等待一段时间让订单成交
                try:
                    # 🔧 修复：Hyperliquid 只接受最多1位小数的价格
                    rounded_sl = Decimal(str(round(float(stop_loss), 1))) if stop_loss else None
                    rounded_tp = Decimal(str(round(float(take_profit), 1))) if take_profit else None

                    cprint(f"📝 开仓后设置止盈止损: SL={rounded_sl}, TP={rounded_tp}", "cyan")
                    # 获取当前持仓
                    position = await self.get_position(symbol)
                    if position:
                        sl_tp_result = await self.modify_stop_loss_take_profit(
                            position=position.model_dump(),  # 转换为字典
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
                    # 止损止盈设置失败不影响开仓结果，但需要记录错误
                    cprint(f"❌ 设置止盈止损时出错: {str(e)}", "red")
                    cprint(f"   开仓已成功，但止盈止损未设置", "yellow")

            # 构造执行结果
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                order_id=order_id or '',
                symbol=symbol,
                executed_amount=executed_amount,
                executed_price=executed_price,
                fee=Decimal("0"),  # SDK 返回中没有直接的手续费字段
                message=f"成功开{'多' if side == PositionSide.LONG else '空'}仓",
                raw_response=result,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=TradingAction.OPEN_LONG if side == PositionSide.LONG else TradingAction.OPEN_SHORT,
                symbol=symbol,
                message="开仓失败",
                error=str(e),
                timestamp=datetime.now(),
            )

    async def close_position(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        amount: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        平仓 - 使用 Hyperliquid SDK

        Args:
            symbol: 交易对
            position_id: 持仓 ID（未使用）
            amount: 平仓数量（None 表示全部平仓）
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 转换交易对格式
            coin = self._symbol_to_coin(symbol)

            # 如果指定了平仓数量（部分平仓）
            if amount:
                # 获取当前持仓以确定方向
                position = await self.get_position(symbol)
                if not position:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        action=TradingAction.CLOSE_POSITION,
                        symbol=symbol,
                        message="未找到持仓",
                        error="No position found",
                        timestamp=datetime.now(),
                    )

                # 反向下单来平仓（使用 reduce_only）
                is_buy = position.side == PositionSide.SHORT  # 平多单需要卖，平空单需要买
                slippage = params.get('slippage', 0.01)

                # 使用 market_open 方法，但带 reduce_only 参数
                result = self._hl_exchange.order(
                    coin,
                    is_buy,
                    float(amount),
                    None,  # 市价单价格为 None
                    {"market": {}},
                    reduce_only=True
                )
            else:
                # 全部平仓 - 使用 SDK 的 market_close 方法
                # market_close(coin, px=None, slippage=0.05)
                slippage = params.get('slippage', 0.05)
                result = self._hl_exchange.market_close(coin, None, slippage)

            # 检查结果
            if result.get('status') != 'ok':
                error_msg = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('error', 'Unknown error')
                raise Exception(error_msg)

            # 解析订单结果
            statuses = result.get('response', {}).get('data', {}).get('statuses', [])
            if not statuses:
                raise Exception("未获取到订单状态")

            status_data = statuses[0]

            # 获取订单 ID 和成交信息
            order_id = None
            executed_amount = Decimal("0")
            executed_price = None

            if 'filled' in status_data:
                # 订单已成交
                filled_data = status_data['filled']
                order_id = str(filled_data.get('oid', ''))
                executed_amount = Decimal(str(filled_data.get('totalSz', 0)))
                executed_price = Decimal(str(filled_data.get('avgPx', 0)))

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.CLOSE_POSITION,
                order_id=order_id or '',
                symbol=symbol,
                executed_amount=executed_amount,
                executed_price=executed_price,
                fee=Decimal("0"),  # SDK 返回中没有直接的手续费字段
                message="成功平仓",
                raw_response=result,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=TradingAction.CLOSE_POSITION,
                symbol=symbol,
                message="平仓失败",
                error=str(e),
                timestamp=datetime.now(),
            )

    async def modify_stop_loss_take_profit(
        self,
        position: Dict[str, Any],  # 修改类型提示：接受字典而不是Position对象
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        **params
    ) -> ExecutionResult:
        """
        修改止损止盈 - 使用 Hyperliquid SDK

        Hyperliquid 的止损止盈是独立的订单，需要：
        1. 获取持仓信息
        2. 取消旧的止损/止盈订单（如果有）
        3. 创建新的止损/止盈订单

        Args:
            position: 持仓信息字典
            stop_loss: 止损价格
            take_profit: 止盈价格
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 转换交易对格式为币种名称
            coin = self._symbol_to_coin(position['symbol'])

            # 转换数量为 float（确保类型正确）
            position_amount = float(position['amount'])
            if position_amount == 0:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    action=TradingAction.MODIFY_SL_TP,
                    symbol=position['symbol'],
                    message="持仓数量为0",
                    error="Position size is 0",
                    timestamp=datetime.now(),
                )

            # 检查是否需要修改（优化：如果新旧值相同则跳过）
            current_sl = position.get('stop_loss')
            current_tp = position.get('take_profit')

            # 转换为 float 以便比较
            if current_sl is not None:
                current_sl = float(current_sl)
            if current_tp is not None:
                current_tp = float(current_tp)

            # 🔧 修复：Hyperliquid 只接受最多1位小数的价格
            # 将止损止盈价格四舍五入到1位小数
            new_sl = round(float(stop_loss), 1) if stop_loss is not None else None
            new_tp = round(float(take_profit), 1) if take_profit is not None else None

            # 如果止损和止盈都没有变化，直接返回成功
            sl_unchanged = (new_sl is None or new_sl == current_sl)
            tp_unchanged = (new_tp is None or new_tp == current_tp)

            if sl_unchanged and tp_unchanged:
                cprint(f"✓ SL/TP unchanged for {position['symbol']}, skipping modification", "yellow")
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    action=TradingAction.MODIFY_SL_TP,
                    symbol=position['symbol'],
                    message="止盈止损未改变，跳过修改",
                    timestamp=datetime.now(),
                )

            # 2. 获取并取消旧的止损/止盈订单（使用 CCXT 查询）
            cprint(f"🔍 查询 {position['symbol']} 的未完成订单...", "cyan")
            open_orders = await self._exchange.fetch_open_orders(
                symbol=position['symbol'],
                params={'user': self._exchange.walletAddress, **params}
            )
            cancelled_orders = []
            cancel_errors = []

            cprint(f"📋 发现 {len(open_orders)} 个未完成订单", "cyan")
            for order in open_orders:
                order_id = order.get('id')
                order_type = order.get('type', '').lower()
                order_info = order.get('info', {})

                # 更精确的止损止盈订单识别
                # Hyperliquid 的止损止盈订单特征：
                # 1. order_type 包含 'stop' 或 'take_profit'
                # 2. info 中有 'trigger' 字段
                # 3. reduceOnly = True
                is_sl_tp_order = (
                    ('stop' in order_type or 'take_profit' in order_type or 'trigger' in order_type) or
                    ('trigger' in order_info) or
                    (order.get('reduceOnly') is True)
                )

                if is_sl_tp_order:
                    cprint(f"  🎯 识别到止盈止损订单: ID={order_id}, Type={order_type}", "yellow")
                    try:
                        # 使用 SDK 的 cancel 方法取消订单（使用 coin 而不是 symbol）
                        cancel_result = self._hl_exchange.cancel(coin, int(order_id))

                        if cancel_result.get('status') == 'ok':
                            cancelled_orders.append(order_id)
                            cprint(f"  ✓ 成功取消订单: {order_id}", "green")
                        else:
                            error_msg = f"取消失败: {cancel_result}"
                            cancel_errors.append(error_msg)
                            cprint(f"  ❌ {error_msg}", "red")

                    except Exception as e:
                        # 订单可能已经被触发或已取消
                        error_msg = f"取消订单 {order_id} 时出错: {str(e)}"
                        cancel_errors.append(error_msg)
                        cprint(f"  ⚠️ {error_msg} (可能已触发)", "yellow")

            if cancelled_orders:
                cprint(f"✓ 成功取消 {len(cancelled_orders)} 个旧的止盈止损订单", "green")
            if cancel_errors and not cancelled_orders:
                cprint(f"⚠️ 取消旧订单时遇到 {len(cancel_errors)} 个错误，但将继续设置新订单", "yellow")

            # 3. 创建新的止损/止盈订单
            result_data = {
                'cancelled_orders': cancelled_orders,
                'new_orders': []
            }

            # 确定平仓方向（与持仓相反）
            # position['side'] 可能是字符串 "long" 或 "short"
            position_side = position['side']
            if isinstance(position_side, str):
                is_buy = position_side.lower() == "short"  # 平多单需要卖（False），平空单需要买（True）
            else:
                is_buy = position_side == PositionSide.SHORT

            # 创建止损订单（使用 SDK 的 order 方法）
            if new_sl is not None:
                try:
                    order_type = {
                        "trigger": {
                            "triggerPx": new_sl,
                            "isMarket": True,
                            "tpsl": "sl"
                        }
                    }
                    sl_result = self._hl_exchange.order(
                        coin,  # 使用 coin 而不是 symbol
                        is_buy,
                        position_amount,
                        new_sl,  # 触发价格
                        order_type,
                        reduce_only=True
                    )

                    # 检查结果 - 修复：需要检查statuses中的error字段
                    if sl_result.get('status') != 'ok':
                        raise Exception("Order API returned non-ok status")

                    # 检查statuses中是否有错误
                    statuses = sl_result.get('response', {}).get('data', {}).get('statuses', [])
                    if statuses and 'error' in statuses[0]:
                        error_msg = statuses[0].get('error', 'Unknown error')
                        raise Exception(f"Stop loss order failed: {error_msg}")

                    # 解析订单 ID
                    statuses = sl_result.get('response', {}).get('data', {}).get('statuses', [])
                    if statuses and 'resting' in statuses[0]:
                        order_id = str(statuses[0]['resting'].get('oid', ''))
                        result_data['new_orders'].append({
                            'type': 'stop_loss',
                            'order_id': order_id,
                            'price': new_sl
                        })

                except Exception as e:
                    cprint(f'{e}', 'red')
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        action=TradingAction.MODIFY_SL_TP,
                        symbol=position['symbol'],
                        message="创建止损订单失败",
                        error=f"Failed to create stop loss: {str(e)}",
                        timestamp=datetime.now(),
                    )

            # 创建止盈订单（使用 SDK 的 order 方法）
            if new_tp is not None:
                try:
                    order_type = {
                        "trigger": {
                            "triggerPx": new_tp,
                            "isMarket": True,
                            "tpsl": "tp"
                        }
                    }
                    tp_result = self._hl_exchange.order(
                        coin,  # 使用 coin 而不是 symbol
                        is_buy,
                        position_amount,
                        new_tp,  # 触发价格
                        order_type,
                        reduce_only=True
                    )

                    # 检查结果 - 修复：需要检查statuses中的error字段
                    if tp_result.get('status') != 'ok':
                        raise Exception("Order API returned non-ok status")

                    # 检查statuses中是否有错误
                    statuses = tp_result.get('response', {}).get('data', {}).get('statuses', [])
                    if statuses and 'error' in statuses[0]:
                        error_msg = statuses[0].get('error', 'Unknown error')
                        raise Exception(f"Take profit order failed: {error_msg}")

                    # 解析订单 ID
                    statuses = tp_result.get('response', {}).get('data', {}).get('statuses', [])
                    if statuses and 'resting' in statuses[0]:
                        order_id = str(statuses[0]['resting'].get('oid', ''))
                        result_data['new_orders'].append({
                            'type': 'take_profit',
                            'order_id': order_id,
                            'price': new_tp
                        })

                except Exception as e:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        action=TradingAction.MODIFY_SL_TP,
                        symbol=position['symbol'],
                        message="创建止盈订单失败",
                        error=f"Failed to create take profit: {str(e)}",
                        timestamp=datetime.now(),
                    )

            message = f"成功设置止盈止损"
            if cancelled_orders:
                message += f"（已取消 {len(cancelled_orders)} 个旧订单）"

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                action=TradingAction.MODIFY_SL_TP,
                symbol=position['symbol'],
                message=message,
                raw_response=result_data,
                timestamp=datetime.now(),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                action=TradingAction.MODIFY_SL_TP,
                symbol=position['symbol'],
                message="修改止损止盈失败",
                error=str(e),
                timestamp=datetime.now(),
            )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        **params
    ) -> ExecutionResult:
        """
        取消订单 - 使用 Hyperliquid SDK

        Args:
            order_id: 订单 ID
            symbol: 交易对
            **params: 其他参数

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 转换交易对格式
            coin = self._symbol_to_coin(symbol)

            # 使用 SDK 的 cancel 方法
            # cancel(coin, oid)
            result = self._hl_exchange.cancel(coin, int(order_id))

            # 检查结果
            if result.get('status') != 'ok':
                error_msg = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('error', 'Unknown error')
                raise Exception(error_msg)

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

    async def get_positions(
        self,
        symbol: Optional[str] = None,
        **params
    ) -> List[Position]:
        """获取持仓列表（包含止盈止损信息）"""
        try:
            symbols = []
            if symbol:
                symbols.append(symbol)

            positions_data = await self._exchange.fetch_positions(
                symbols=symbols,
                params={
                    'user': self._exchange.walletAddress, **params
                }
            )

            positions = []
            for pos in positions_data:
                if float(pos.get('contracts', 0)) == 0:
                    continue  # 跳过空持仓

                # 初始化止盈止损为 None
                stop_loss = None
                take_profit = None

                # 查询该交易对的未完成订单，查找止盈止损订单
                try:
                    open_orders = await self._exchange.fetch_open_orders(
                        symbol=pos['symbol'],
                        params={'user': self._exchange.walletAddress}
                    )

                    for order in open_orders:
                        # 识别止损止盈订单
                        is_sl_tp = order['info'].get('isTrigger')

                        if is_sl_tp:
                            # 获取触发价格
                            trigger_price = order['info'].get('triggerPx')
                            if trigger_price:
                                trigger_price_decimal = Decimal(str(trigger_price))

                                order_type = order['info'].get('orderType', '').lower() 

                                if order_type.startswith('stop'):
                                    # 止损单
                                    stop_loss = trigger_price_decimal
                                elif order_type.startswith('take profit'):
                                    # 止盈单
                                    take_profit = trigger_price_decimal
                                else:
                                    raise ValueError(f"未知的止盈止损订单类型: {order_type}")

                except Exception as e:
                    # 查询订单失败不影响持仓信息返回
                    pass

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
                    stop_loss=stop_loss,  # 添加止损
                    take_profit=take_profit,  # 添加止盈
                    opened_at=datetime.fromtimestamp(pos['timestamp'] / 1000) if pos.get('timestamp') else datetime.now(),
                    raw_data=pos,
                )
                positions.append(position)

            return positions

        except Exception as e:
            # 发生错误时返回空列表
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
            balance_data = await self._exchange.fetch_balance(params={'user': self._exchange.walletAddress, **params})

            # 默认获取 USDC 余额
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
    ) -> List["Order"]:
        """获取当前委托订单（未完成的订单）"""
        try:
            orders_data = await self._exchange.fetch_open_orders(
                symbol, 
                params={
                    'user': self._exchange.walletAddress,
                    **params
                }
            )

            orders = []
            for order_dict in orders_data:
                order = self._parse_order(order_dict)
                orders.append(order)

            return orders

        except Exception as e:
            raise Exception(f"获取委托订单失败: {str(e)}")

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List["Order"]:
        """获取历史订单（包括已完成、已取消的订单）"""
        try:
            if self._exchange.has.get('fetchOrders'):
                orders_data = await self._exchange.fetch_orders(
                    symbol=symbol,
                    since=since,
                    limit=limit,
                    params={
                        'user': self._exchange.walletAddress,
                        **params
                    }
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
            
            orders = []
            for order_dict in orders_data:
                order = self._parse_order(order_dict)
                orders.append(order)

            return orders

        except Exception as e:
            raise Exception(f"获取历史订单失败: {str(e)}")

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 100,
        **params
    ) -> List["Trade"]:
        """获取历史成交记录"""
        try:
            # 使用 CCXT 的 fetch_my_trades
            trades_data = await self._exchange.fetch_my_trades(
                symbol=symbol,
                since=since,
                limit=limit,
                params={
                    'user': self._exchange.apiKey,
                    **params
                }
            )

            trades = []
            for trade_dict in trades_data:
                trade = self._parse_trade(trade_dict)
                trades.append(trade)

            return trades

        except Exception as e:
            raise Exception(f"获取成交记录失败: {str(e)}")

    async def get_ticker(
        self,
        symbol: str,
        **params
    ) -> Dict[str, Any]:
        """获取行情信息"""
        try:
            ticker = await self._exchange.fetch_ticker(symbol, params)
            return ticker
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
            # 某些交易所可能不支持通过 API 设置杠杆
            return False

    # ========== 行情数据接口 ==========

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
            # 如果没有提供 since，计算最近的时间戳
            # Hyperliquid 需要 since 参数才能获取最新数据，否则返回缓存
            if since is None:
                import time
                # 时间周期映射（分钟数）
                timeframe_minutes = {
                    '1m': 1,
                    '15m': 15,
                    '1h': 60,
                    '4h': 240,
                    '1d': 1440,
                }
                minutes = timeframe_minutes.get(timeframe, 60)
                # 计算 since：当前时间 - (limit * 时间周期)
                since = int((time.time() - limit * minutes * 60) * 1000)

            # 使用 CCXT 的 fetch_ohlcv 方法获取K线
            ohlcv_data = await self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
                params=params
            )

            klines = []
            for candle in ohlcv_data:
                # CCXT OHLCV 格式: [timestamp, open, high, low, close, volume]
                kline = Candle(
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
                klines.append(kline)

            return klines

        except Exception as e:
            raise Exception(f"获取K线数据失败: {str(e)}")

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
            # CCXT 的资金费率接口
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

    async def get_latest_price(
        self,
        symbol: str,
        **params
    ) -> LatestPrice:
        """获取最新价格信息"""
        try:
            # 获取ticker数据来获取最新价格
            ticker = await self._exchange.fetch_ticker(symbol, params)

            # 尝试获取资金费率数据以获取标记价格和指数价格
            mark_price = None
            index_price = None
            try:
                funding_data = await self._exchange.fetch_funding_rate(symbol, params)
                mark_price = Decimal(str(funding_data.get('markPrice', 0))) if funding_data.get('markPrice') else None
                index_price = Decimal(str(funding_data.get('indexPrice', 0))) if funding_data.get('indexPrice') else None
            except:
                # 如果获取资金费率失败，使用 ticker 中的标记价格（如果有）
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
        """获取订单簿（盘口数据）"""
        try:
            order_book_data = await self._exchange.fetch_order_book(symbol, limit, params)

            # 转换买卖盘数据为 Decimal 类型
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

    # ========== 辅助解析方法 ==========

    def _parse_order(self, order_dict: Dict[str, Any]) -> Order:
        """
        将 CCXT 订单数据转换为 Order 模型

        Args:
            order_dict: CCXT 订单数据字典

        Returns:
            Order: 订单模型
        """
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.CLOSED,
            'canceled': OrderStatus.CANCELED,
            'cancelled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
        }

        ccxt_status = order_dict.get('status', 'open')
        # 检查是否部分成交
        filled = float(order_dict.get('filled', 0))
        amount = float(order_dict.get('amount', 0))
        if ccxt_status == 'open' and filled > 0 and filled < amount:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = status_map.get(ccxt_status, OrderStatus.OPEN)

        # 映射订单类型
        order_type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop_market': OrderType.STOP_MARKET,
            'stop_limit': OrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(order_dict.get('type', 'market'), OrderType.MARKET)

        # 映射订单方向
        side = PositionSide.LONG if order_dict.get('side') == 'buy' else PositionSide.SHORT

        # 安全地获取 fee 信息
        fee_info = order_dict.get('fee')
        fee_cost = None
        fee_currency = None
        if fee_info and isinstance(fee_info, dict):
            fee_cost = self._safe_decimal_optional(fee_info.get('cost'))
            fee_currency = fee_info.get('currency')

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
        """
        将 CCXT 成交数据转换为 Trade 模型

        Args:
            trade_dict: CCXT 成交数据字典

        Returns:
            Trade: 成交模型
        """
        # 映射订单方向
        side = PositionSide.LONG if trade_dict.get('side') == 'buy' else PositionSide.SHORT

        # 安全地获取 fee 信息
        fee_info = trade_dict.get('fee')
        fee_cost = None
        fee_currency = None
        if fee_info and isinstance(fee_info, dict):
            fee_cost = self._safe_decimal_optional(fee_info.get('cost'))
            fee_currency = fee_info.get('currency')

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

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        **params
    ) -> List[Candle]:
        """
        获取 K 线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期 (1m, 5m, 15m, 1h, 4h, 1d等)
            limit: 返回数量
            **params: 其他参数

        Returns:
            List[Kline]: K 线数据列表
        """
        try:
            # 使用 CCXT 获取 OHLCV 数据
            ohlcv_data = await self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                params=params
            )

            # 转换为 Kline 模型
            klines = []
            for candle in ohlcv_data:
                # CCXT OHLCV 格式: [timestamp, open, high, low, close, volume]
                kline = Candle(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=self._safe_decimal(candle[1]),
                    high=self._safe_decimal(candle[2]),
                    low=self._safe_decimal(candle[3]),
                    close=self._safe_decimal(candle[4]),
                    volume=self._safe_decimal(candle[5]),
                    raw_data={'ohlcv': candle}
                )
                klines.append(kline)

            return klines

        except Exception as e:
            raise Exception(f"获取 K 线数据失败 ({symbol}, {timeframe}): {str(e)}")