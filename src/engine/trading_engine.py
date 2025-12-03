"""
交易执行引擎

确定性交易引擎 - 不涉及LLM推理
- 批量获取市场数据
- 批量计算技术指标
- 执行交易订单

设计原则:
- 快速: 毫秒级响应
- 可靠: 确定性输出
- 高效: 批量操作，减少API调用
"""
import asyncio
from typing import List, Dict, Optional, TYPE_CHECKING

from decimal import Decimal
from datetime import datetime

from termcolor import cprint

from ..adapters.base import BaseExchangeAdapter
from ..models import PositionSide, OrderType, TradeHistoryManager
from .market_snapshot import MarketSnapshot, AssetData

if TYPE_CHECKING:
    from ..strategies import BaseStrategy


class TradingEngine:
    """
    交易引擎

    职责:
    1. 批量获取市场数据（价格、K线、持仓等）
    2. 批量计算技术指标（numpy操作，毫秒级）
    3. 执行交易订单（直接调用adapter）
    4. 管理数据缓存

    策略支持:
    - 根据策略配置动态选择时间框架和K线数量
    - 使用策略对应的指标计算器
    """

    def __init__(
        self,
        adapter: BaseExchangeAdapter,
        trade_history: Optional[TradeHistoryManager] = None,
        strategy: Optional["BaseStrategy"] = None,
    ):
        self.adapter = adapter
        self.trade_history = trade_history
        self.strategy = strategy
        self._cache: Dict[str, any] = {}
        self._cache_ttl = 10  # 缓存10秒

        # 策略相关配置
        if strategy:
            self._timeframes = strategy.get_timeframe_list()
            self._candle_limits = strategy.get_candle_limits()
            self._indicator_calculator = strategy.get_indicator_calculator()
            cprint(f"📊 TradingEngine 使用策略: {strategy.name}", "cyan")
        else:
            # 默认配置 (兼容无策略模式)
            self._timeframes = ["1h", "15m", "5m"]
            self._candle_limits = {"1h": 200, "15m": 100, "5m": 100}
            self._indicator_calculator = None

    async def get_market_snapshot(self, symbols: List[str]) -> MarketSnapshot:
        """
        批量获取市场快照

        一次性获取所有需要的数据，避免多次API调用

        Args:
            symbols: 交易对列表

        Returns:
            MarketSnapshot: 完整的市场快照
        """
        try:
            cprint(f"📊 获取市场快照: {', '.join(symbols)}", "cyan")

            tasks = []
            for symbol in symbols:
                tasks.append(self._get_asset_data(symbol))

            # 同时获取账户信息
            tasks.append(self._get_account_data())

            results = await asyncio.gather(*tasks, return_exceptions=True)

            asset_results = results[:-1]
            account_data = results[-1]

            assets = {}
            for result in asset_results:
                if isinstance(result, Exception):
                    cprint(f"⚠️  获取资产数据失败: {result}", "yellow")
                    continue
                if result:
                    assets[result.symbol] = result

            if isinstance(account_data, Exception):
                cprint(f"⚠️  获取账户数据失败: {account_data}", "yellow")
                account_data = {
                    'balance': Decimal('0'),
                    'available': Decimal('0'),
                }

            # 计算总持仓市值和未实现盈亏
            total_position_value = Decimal('0')
            total_unrealized_pnl = Decimal('0')
            for asset in assets.values():
                if asset.has_position():
                    total_position_value += asset.position_size * asset.current_price
                    if asset.unrealized_pnl:
                        total_unrealized_pnl += asset.unrealized_pnl

            snapshot = MarketSnapshot(
                assets=assets,
                account_balance=account_data.get('balance', Decimal('0')),
                account_available=account_data.get('available', Decimal('0')),
                total_position_value=total_position_value,
                total_unrealized_pnl=total_unrealized_pnl,
                timestamp=datetime.now()
            )

            cprint(f"✅ 市场快照获取完成: {len(assets)} 个资产", "green")
            return snapshot

        except Exception as e:
            cprint(f"❌ 获取市场快照失败: {e}", "red")
            # 返回空快照
            return MarketSnapshot(assets={})

    async def _get_asset_data(self, symbol: str) -> Optional[AssetData]:
        """
        获取单个资产的完整数据

        使用策略模块获取 K 线数据和计算指标
        """
        try:
            # 并发获取：价格、持仓、资金费率、持仓量、OI历史
            price_task = self.adapter.get_latest_price(symbol)
            position_task = self.adapter.get_position(symbol)
            funding_rate_task = self.adapter.get_funding_rate(symbol)
            open_interest_task = self.adapter.get_open_interest(symbol)

            # OI 历史（可选）
            oi_history_task = None
            if hasattr(self.adapter, 'get_open_interest_history'):
                oi_history_task = self.adapter.get_open_interest_history(symbol, period='4h', limit=10)

            tasks = [price_task, position_task, funding_rate_task, open_interest_task]
            if oi_history_task:
                tasks.append(oi_history_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            price = results[0]
            position = results[1]
            funding_rate_data = results[2]
            open_interest = results[3]
            oi_history = results[4] if len(results) > 4 else []

            # 处理价格异常
            if isinstance(price, Exception):
                cprint(f"⚠️  {symbol} 价格获取失败: {price}", "yellow")
                return None

            # 使用策略获取市场数据和计算指标
            tf_1h = None
            tf_15m = None
            tf_5m = None

            if self.strategy:
                try:
                    tf_indicators = await self.strategy.fetch_market_data(self.adapter, symbol)
                    tf_1h = tf_indicators.get("1h")
                    tf_15m = tf_indicators.get("15m")
                    tf_5m = tf_indicators.get("5m")

                    # 添加 OI 指标到 1H 时间框架
                    if tf_1h and not isinstance(oi_history, Exception) and oi_history:
                        self._add_oi_indicators(tf_1h, oi_history)
                except Exception as e:
                    cprint(f"⚠️  {symbol} 策略指标计算失败: {e}", "yellow")

            # 提取市场情绪指标
            funding_rate_val = None
            if not isinstance(funding_rate_data, Exception) and funding_rate_data:
                funding_rate_val = float(funding_rate_data.funding_rate)

            open_interest_val = None
            if not isinstance(open_interest, Exception) and open_interest:
                open_interest_val = float(open_interest)

            # 构建持仓信息
            position_size = Decimal('0')
            position_side = None
            entry_price = None
            unrealized_pnl = None
            stop_loss = None
            take_profit = None

            if position and not isinstance(position, Exception):
                position_size = position.amount
                position_side = position.side.value if position.side else None
                entry_price = position.entry_price
                unrealized_pnl = position.unrealized_pnl
                stop_loss = position.stop_loss
                take_profit = position.take_profit

            # 获取 24 小时涨跌幅
            change_24h_percent = None
            if price.price_change_percent is not None:
                change_24h_percent = float(price.price_change_percent)

            # 构建 AssetData
            asset_data = AssetData(
                symbol=symbol,
                current_price=price.last_price,
                mark_price=price.mark_price,
                bid=price.bid_price,
                ask=price.ask_price,
                change_24h_percent=change_24h_percent,
                tf_1h=tf_1h,
                tf_15m=tf_15m,
                tf_5m=tf_5m,
                funding_rate=funding_rate_val,
                open_interest=open_interest_val,
                position_size=position_size,
                position_side=position_side,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )

            return asset_data

        except Exception as e:
            cprint(f"❌ 获取 {symbol} 数据失败: {e}", "red")
            return None

    @staticmethod
    def _add_oi_indicators(tf_4h, oi_history: List[Dict]) -> None:
        """
        将持仓量指标添加到4小时时间框架

        计算:
        - oi_current: 当前持仓量 (USD)
        - oi_change_4h: 4小时持仓量变化率 (%)
        - oi_change_24h: 24小时持仓量变化率 (%)
        - oi_series: 持仓量序列

        Args:
            tf_4h: TimeframeIndicators 对象
            oi_history: 持仓量历史数据列表 (按时间排序，最新在后)
        """
        if not oi_history or len(oi_history) < 2:
            return

        try:
            # 提取 OI 值序列 (sum_open_interest_value 是 USD 价值)
            oi_values = [item.get('sum_open_interest_value', 0) for item in oi_history]

            # 当前持仓量 (最新值)
            tf_4h.oi_current = oi_values[-1] if oi_values else None

            # 4小时变化率 (最新 vs 上一个4小时)
            if len(oi_values) >= 2 and oi_values[-2] > 0:
                change_4h = ((oi_values[-1] - oi_values[-2]) / oi_values[-2]) * 100
                tf_4h.oi_change_4h = round(change_4h, 2)

            # 24小时变化率 (需要6个4小时数据点)
            if len(oi_values) >= 7 and oi_values[-7] > 0:
                change_24h = ((oi_values[-1] - oi_values[-7]) / oi_values[-7]) * 100
                tf_4h.oi_change_24h = round(change_24h, 2)

            # 保存序列数据
            tf_4h.oi_series = oi_values[-10:] if len(oi_values) >= 10 else oi_values

        except Exception as e:
            cprint(f"⚠️  OI指标计算失败: {e}", "yellow")

    async def _get_account_data(self) -> dict:
        """获取账户数据"""
        try:
            # 使用 adapter 的保证金类型（如果有），否则默认 USDC
            margin_type = getattr(self.adapter, 'margin_type', 'USDC')
            balance = await self.adapter.get_balance(margin_type)
            return {
                'balance': balance.total,
                'available': balance.available,
            }
        except Exception as e:
            cprint(f"⚠️  获取账户数据失败: {e}", "yellow")
            return {
                'balance': Decimal('0'),
                'available': Decimal('0'),
            }

    async def execute_signal(self, signal: dict) -> dict:
        """
        执行交易信号

        直接调用adapter，不涉及LLM推理

        Args:
            signal: 交易信号字典
                {
                    'action': 'open_long' | 'open_short' | 'close_position',
                    'symbol': 'BTC/USDC:USDC',
                    'amount': 0.001,
                    'leverage': 3,
                    'stop_loss': 88000.0,
                    'take_profit': 96000.0,
                }

        Returns:
            执行结果字典
        """
        try:
            action = signal.get('action')
            symbol = signal.get('symbol')

            if action == 'open_long':
                result = await self.adapter.open_position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    amount=Decimal(str(signal['amount'])),
                    order_type=OrderType.MARKET,
                    leverage=signal.get('leverage', 1),
                    stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                    take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None,
                )

                # 记录交易
                if result.status.value == 'success' and self.trade_history:
                    self.trade_history.record_trade(
                        symbol=symbol,
                        side='long',
                        action='open',
                        price=result.executed_price or Decimal(str(signal.get('price', 0))),
                        amount=Decimal(str(signal['amount'])),
                        leverage=signal.get('leverage', 1),
                        fee=result.fee or Decimal('0'),
                        note=f"Open long position"
                    )
                    # 更新持仓的止损止盈
                    if signal.get('stop_loss') or signal.get('take_profit'):
                        self.trade_history.update_position_sl_tp(
                            symbol=symbol,
                            stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                            take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None,
                        )

                return {'success': result.status.value == 'success', 'result': result}

            elif action == 'open_short':
                result = await self.adapter.open_position(
                    symbol=symbol,
                    side=PositionSide.SHORT,
                    amount=Decimal(str(signal['amount'])),
                    order_type=OrderType.MARKET,
                    leverage=signal.get('leverage', 1),
                    stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                    take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None,
                )

                # 记录交易
                if result.status.value == 'success' and self.trade_history:
                    self.trade_history.record_trade(
                        symbol=symbol,
                        side='short',
                        action='open',
                        price=result.executed_price or Decimal(str(signal.get('price', 0))),
                        amount=Decimal(str(signal['amount'])),
                        leverage=signal.get('leverage', 1),
                        fee=result.fee or Decimal('0'),
                        note=f"Open short position"
                    )
                    # 更新持仓的止损止盈
                    if signal.get('stop_loss') or signal.get('take_profit'):
                        self.trade_history.update_position_sl_tp(
                            symbol=symbol,
                            stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                            take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None,
                        )

                return {'success': result.status.value == 'success', 'result': result}

            elif action == 'close_position':
                # 先获取持仓信息以计算盈亏
                position = await self.adapter.get_position(symbol)

                result = await self.adapter.close_position(symbol=symbol)

                # 记录交易
                if result.status.value == 'success' and self.trade_history and position:
                    self.trade_history.record_trade(
                        symbol=symbol,
                        side=position.side.value if position.side else 'long',
                        action='close',
                        price=result.executed_price or Decimal('0'),
                        amount=position.amount,
                        leverage=None,
                        fee=result.fee or Decimal('0'),
                        note=f"Close position manually"
                    )

                return {'success': result.status.value == 'success', 'result': result}

            elif action == 'set_stop_loss':
                position = await self.adapter.get_position(symbol)
                if position:
                    result = await self.adapter.modify_stop_loss_take_profit(
                        position=position.model_dump(),
                        stop_loss=Decimal(str(signal['stop_loss'])),
                        take_profit=None
                    )
                    return {'success': result.status.value == 'success', 'result': result}
                else:
                    return {'success': False, 'error': 'Position not found'}

            elif action == 'set_take_profit':
                position = await self.adapter.get_position(symbol)
                if position:
                    result = await self.adapter.modify_stop_loss_take_profit(
                        position=position.model_dump(),
                        stop_loss=None,
                        take_profit=Decimal(str(signal['take_profit']))
                    )
                    return {'success': result.status.value == 'success', 'result': result}
                else:
                    return {'success': False, 'error': 'Position not found'}

            elif action == 'set_stop_loss_take_profit':
                position = await self.adapter.get_position(symbol)
                if position:
                    result = await self.adapter.modify_stop_loss_take_profit(
                        position=position.model_dump(),
                        stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                        take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None
                    )

                    # 更新持仓的止损止盈
                    if result.status.value == 'success' and self.trade_history:
                        self.trade_history.update_position_sl_tp(
                            symbol=symbol,
                            stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                            take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None,
                        )

                    return {'success': result.status.value == 'success', 'result': result}
                else:
                    return {'success': False, 'error': 'Position not found'}

            elif action == 'hold':
                # hold 操作：维持现有仓位，可选更新止损止盈
                position = await self.adapter.get_position(symbol)
                if position:
                    # 如果提供了止损止盈，检查是否需要更新
                    new_sl = Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None
                    new_tp = Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None

                    # 获取当前止损止盈（容差比较，避免浮点精度问题）
                    current_sl = position.stop_loss
                    current_tp = position.take_profit

                    def is_price_different(new_price: Decimal | None, current_price: Decimal | None, tolerance: float = 0.01) -> bool:
                        """检查价格是否有显著变化（超过容差百分比）"""
                        if new_price is None and current_price is None:
                            return False
                        if new_price is None or current_price is None:
                            return True
                        if current_price == 0:
                            return new_price != 0
                        diff_percent = abs(float(new_price - current_price) / float(current_price)) * 100
                        return diff_percent > tolerance

                    sl_changed = is_price_different(new_sl, current_sl)
                    tp_changed = is_price_different(new_tp, current_tp)

                    if sl_changed or tp_changed:
                        cprint(f"  📝 更新止损止盈: SL {current_sl} → {new_sl}, TP {current_tp} → {new_tp}", "cyan")
                        result = await self.adapter.modify_stop_loss_take_profit(
                            position=position.model_dump(),
                            stop_loss=new_sl,
                            take_profit=new_tp
                        )

                        if result.status.value == 'success' and self.trade_history:
                            self.trade_history.update_position_sl_tp(
                                symbol=symbol,
                                stop_loss=new_sl,
                                take_profit=new_tp,
                            )

                        return {'success': result.status.value == 'success', 'result': result, 'message': 'Position held, SL/TP updated'}
                    else:
                        cprint(f"  ℹ️ 止损止盈无变化，跳过更新", "yellow")
                        return {'success': True, 'message': 'Position held, SL/TP unchanged'}
                else:
                    return {'success': True, 'message': 'No position to hold'}

            elif action == 'wait':
                # wait 操作：不做任何交易，直接返回成功
                return {'success': True, 'message': 'Waiting, no action taken'}

            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            cprint(f"❌ 执行信号失败: {e}", "red")
            return {'success': False, 'error': str(e)}
