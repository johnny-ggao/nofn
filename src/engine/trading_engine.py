"""
交易执行引擎 (Layer 1)

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
from typing import List, Dict, Optional
from decimal import Decimal
from datetime import datetime

from termcolor import cprint

from ..adapters.base import BaseExchangeAdapter
from ..models import Position, PositionSide, OrderType, Candle, TradeHistoryManager
from ..utils.indicators import ema, rsi, macd, atr
from .market_snapshot import MarketSnapshot, AssetData, IndicatorData


class TradingEngine:
    """
    确定性交易引擎

    职责:
    1. 批量获取市场数据（价格、K线、持仓等）
    2. 批量计算技术指标（numpy操作，毫秒级）
    3. 执行交易订单（直接调用adapter）
    4. 管理数据缓存
    """

    def __init__(self, adapter: BaseExchangeAdapter, trade_history: Optional[TradeHistoryManager] = None):
        self.adapter = adapter
        self.trade_history = trade_history
        self._cache: Dict[str, any] = {}
        self._cache_ttl = 10  # 缓存10秒

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
        """获取单个资产的完整数据"""
        try:
            # 并发获取价格、K线（15分钟、1小时、4小时）、持仓、资金费率、持仓量、持仓量历史
            price_task = self.adapter.get_latest_price(symbol)
            candles_15m_task = self.adapter.get_candles(symbol, '15m', limit=100)  # 15分钟K线 - 精确入场
            candles_1h_task = self.adapter.get_candles(symbol, '1h', limit=100)    # 1小时K线 - 入场时机
            candles_4h_task = self.adapter.get_candles(symbol, '4h', limit=200)    # 4小时K线 - 趋势确认
            position_task = self.adapter.get_position(symbol)
            funding_rate_task = self.adapter.get_funding_rate(symbol)
            open_interest_task = self.adapter.get_open_interest(symbol)

            # 获取持仓量历史（4小时周期，用于计算变化率）
            # 需要检查 adapter 是否支持此方法
            oi_history_task = None
            if hasattr(self.adapter, 'get_open_interest_history'):
                oi_history_task = self.adapter.get_open_interest_history(symbol, period='4h', limit=10)

            tasks = [
                price_task, candles_15m_task, candles_1h_task, candles_4h_task,
                position_task, funding_rate_task, open_interest_task
            ]
            if oi_history_task:
                tasks.append(oi_history_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            price = results[0]
            candles_15m = results[1]
            candles_1h = results[2]
            candles_4h = results[3]
            position = results[4]
            funding_rate_data = results[5]
            open_interest = results[6]
            oi_history = results[7] if len(results) > 7 else []

            # 处理异常
            if isinstance(price, Exception):
                cprint(f"⚠️  {symbol} 价格获取失败: {price}", "yellow")
                return None
            if isinstance(candles_15m, Exception):
                cprint(f"⚠️  {symbol} 15分钟K线获取失败: {candles_15m}", "yellow")
                candles_15m = []
            if isinstance(candles_1h, Exception):
                cprint(f"⚠️  {symbol} 1小时K线获取失败: {candles_1h}", "yellow")
                candles_1h = []
            if isinstance(candles_4h, Exception):
                cprint(f"⚠️  {symbol} 4小时K线获取失败: {candles_4h}", "yellow")
                candles_4h = []

            current_price = float(price.last_price)

            # 延迟导入以避免循环依赖
            from ..utils.mtf_calculator import MTFCalculator

            # 计算多时间框架指标
            tf_4h = None
            tf_1h = None
            tf_15m = None

            if candles_4h and len(candles_4h) >= 50:
                ohlcv_4h = self._candles_to_ohlcv(candles_4h)
                tf_4h = MTFCalculator.calculate_4h(ohlcv_4h, current_price)

                # 添加 OI 指标到 4H 时间框架
                if tf_4h and not isinstance(oi_history, Exception) and oi_history:
                    self._add_oi_indicators(tf_4h, oi_history)

            if candles_1h and len(candles_1h) >= 50:
                ohlcv_1h = self._candles_to_ohlcv(candles_1h)
                tf_1h = MTFCalculator.calculate_1h(ohlcv_1h, current_price)

            if candles_15m and len(candles_15m) >= 20:
                ohlcv_15m = self._candles_to_ohlcv(candles_15m)
                tf_15m = MTFCalculator.calculate_15m(ohlcv_15m, current_price)

            # 旧版指标（向后兼容）- 使用1小时数据
            indicators_5m = self._calculate_indicators(candles_1h) if candles_1h else IndicatorData()
            indicators_4h = self._calculate_indicators(candles_4h) if candles_4h else None

            # 提取市场情绪指标（独立于时间框架）
            funding_rate_val = None
            if not isinstance(funding_rate_data, Exception) and funding_rate_data:
                funding_rate_val = float(funding_rate_data.funding_rate)

            open_interest_val = None
            if not isinstance(open_interest, Exception) and open_interest:
                open_interest_val = float(open_interest)

            # 计算成交量指标（从4小时K线）
            volume_current = None
            volume_avg = None
            if candles_4h and len(candles_4h) >= 20:
                volumes_4h = [float(c.volume) for c in candles_4h]
                volume_current = volumes_4h[-1]  # 最新一根K线的成交量
                volume_avg = sum(volumes_4h[-20:]) / 20  # 最近20根的平均

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

            # 构建AssetData
            asset_data = AssetData(
                symbol=symbol,
                current_price=price.last_price,
                mark_price=price.mark_price,
                bid=price.bid_price,
                ask=price.ask_price,
                # 多时间框架指标
                tf_4h=tf_4h,
                tf_1h=tf_1h,
                tf_15m=tf_15m,
                # 旧版指标（向后兼容）
                indicators=indicators_5m,
                indicators_4h=indicators_4h,
                funding_rate=funding_rate_val,
                open_interest=open_interest_val,
                volume_current=volume_current,
                volume_avg=volume_avg,
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
    def _candles_to_ohlcv(candles: List[Candle]):
        """将Candle列表转换为OHLCVData"""
        from ..utils.mtf_calculator import OHLCVData
        return OHLCVData(
            open=[float(c.open) for c in candles],
            high=[float(c.high) for c in candles],
            low=[float(c.low) for c in candles],
            close=[float(c.close) for c in candles],
            volume=[float(c.volume) for c in candles],
        )

    @staticmethod
    def _calculate_indicators(candles: List[Candle]) -> IndicatorData:
        """
        批量计算所有技术指标

        使用numpy操作，毫秒级完成
        """
        if not candles or len(candles) < 50:
            return IndicatorData()

        try:
            # 提取价格数据
            closes = [float(c.close) for c in candles]
            highs = [float(c.high) for c in candles]
            lows = [float(c.low) for c in candles]

            # 批量计算指标（返回数组，需要提取最后一个值）
            ema20_arr = ema(closes, 20)
            ema50_arr = ema(closes, 50)
            ema200_arr = ema(closes, 200) if len(closes) >= 200 else None

            # 提取最新值（数组的最后一个元素）
            ema20_val = float(ema20_arr[-1]) if len(ema20_arr) > 0 else None
            ema50_val = float(ema50_arr[-1]) if len(ema50_arr) > 0 else None
            ema200_val = float(ema200_arr[-1]) if ema200_arr is not None and len(ema200_arr) > 0 else None

            rsi14_arr = rsi(closes, 14)
            rsi14_val = float(rsi14_arr[-1]) if len(rsi14_arr) > 0 else 50.0

            # MACD 返回 (macd_line, signal_line, histogram) 三个数组
            macd_line, signal_line, histogram = macd(closes)
            macd_value = float(macd_line[-1]) if len(macd_line) > 0 else 0.0
            macd_signal = float(signal_line[-1]) if len(signal_line) > 0 else 0.0
            macd_hist = float(histogram[-1]) if len(histogram) > 0 else 0.0

            atr14_arr = atr(highs, lows, closes, 14)
            atr14_val = float(atr14_arr[-1]) if len(atr14_arr) > 0 else 0.0

            # 提取最近10个点的序列（从旧到新）
            n_points = 10
            prices_series = [float(closes[i]) for i in range(-min(n_points, len(closes)), 0)] if len(closes) > 0 else None
            ema20_series = [float(ema20_arr[i]) for i in range(-min(n_points, len(ema20_arr)), 0)] if len(ema20_arr) > 0 else None
            macd_series = [float(macd_line[i]) for i in range(-min(n_points, len(macd_line)), 0)] if len(macd_line) > 0 else None
            rsi14_series = [float(rsi14_arr[i]) for i in range(-min(n_points, len(rsi14_arr)), 0)] if len(rsi14_arr) > 0 else None

            return IndicatorData(
                ema20=ema20_val,
                ema50=ema50_val,
                ema200=ema200_val,
                rsi14=rsi14_val,
                macd_value=macd_value,
                macd_signal=macd_signal,
                macd_histogram=macd_hist,
                atr14=atr14_val,
                # 序列数据
                prices_series=prices_series,
                ema20_series=ema20_series,
                macd_series=macd_series,
                rsi14_series=rsi14_series,
            )

        except Exception as e:
            cprint(f"⚠️  指标计算失败: {e}", "yellow")
            return IndicatorData()

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
            balance = await self.adapter.get_balance('USDC')
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

            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            cprint(f"❌ 执行信号失败: {e}", "red")
            return {'success': False, 'error': str(e)}
