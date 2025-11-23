"""
Generator - ACE 生成者模块

职责：
1. 获取市场数据
2. 检索相关知识条目
3. 构建动态 Prompt
4. LLM 推理
5. 执行交易
6. 创建 ExecutionTrace
"""


import json
from typing import List
from decimal import Decimal
from pathlib import Path

from termcolor import cprint

from ..models import ExecutionTrace, TradeDecision
from ..storage import ContextStore
from ..utils import EmbeddingService, PromptBuilder


class Generator:
    """ACE 生成者"""

    def __init__(
        self,
        llm,  # LangChain LLM
        exchange_adapter,  # 现有的 HyperliquidAdapter
        context_store: ContextStore,
        embedding_service: EmbeddingService
    ):
        self.llm = llm
        self.exchange_adapter = exchange_adapter
        self.context_store = context_store
        self.embedding_service = embedding_service

        # 加载 ACE 专用 system prompt
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "ace_system.txt"
        try:
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
            cprint("✅ ACE System Prompt 加载成功", "green")
        except Exception as e:
            cprint(f"⚠️  加载 System Prompt 失败: {e}，使用默认", "yellow")
            self.system_prompt = "你是一个专业的加密货币交易智能体。"

    async def execute(self, symbols: List[str]) -> ExecutionTrace:
        """执行一次完整的生成-决策-执行流程"""
        trace = ExecutionTrace()

        try:
            # 1. 获取市场数据
            cprint("📊 获取市场数据...", "cyan")
            trace.market_data = await self._fetch_market_data(symbols)

            # 2. 获取账户状态
            trace.account_state = await self._fetch_account_state()

            # 3. 检索相关知识条目

            query_text = self._create_query_text(trace.market_data)
            query_embedding = await self.embedding_service.embed(query_text)

            entries_with_scores = self.context_store.retrieve_similar_entries(
                query_embedding=query_embedding,
                top_k=15,
                min_confidence=0.3
            )

            entries = [e for e, score in entries_with_scores]
            trace.retrieved_entries = [e.entry_id for e in entries]

            cprint(f"📚 检索到 {len(entries)} 个相关知识条目", "cyan")

            # 4. 构建 Prompt
            prompt = PromptBuilder.build_generator_prompt(
                market_data=trace.market_data,
                account_state=trace.account_state,
                entries=entries
            )

            # 5. LLM 推理
            cprint("🤔 LLM 正在分析市场...", "yellow")
            response = await self.llm.ainvoke([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ])

            trace.raw_llm_output = response.content

            # 6. 解析决策（多个币对）
            decisions = self._parse_decisions(response.content)
            for decision in decisions:
                decision.used_entry_ids = trace.retrieved_entries
            trace.decisions = decisions

            cprint(f"💡 解析到 {len(decisions)} 个决策", "green")
            for dec in decisions:
                cprint(f"   {dec.symbol}: {dec.action} (置信度: {dec.confidence:.2f})", "cyan")

            # 7. 执行交易（逐个处理）
            account_before = trace.account_state.get('balance', {}).get('total', 0)

            for decision in decisions:
                if decision.action != "hold":
                    cprint(f"\n🔄 执行 {decision.symbol} 的交易...", "yellow")
                    execution_result = await self._execute_trade(decision)
                    trace.execution_results.append(execution_result)

                    if not execution_result.get('success', False):
                        trace.execution_errors.append(f"{decision.symbol}: {execution_result.get('error', 'Unknown error')}")
                else:
                    cprint(f"⏸️  {decision.symbol}: 观望/持有", "white")
                    trace.execution_results.append({'symbol': decision.symbol, 'action': 'hold', 'success': True})

            # 8. 判断整体执行成功
            trace.execution_success = len(trace.execution_errors) == 0

            # 9. 获取账户整体变化
            account_after_state = await self._fetch_account_state()
            account_after = account_after_state.get('balance', {}).get('total', 0)
            pnl = account_after - account_before

            trace.account_change = {
                'balance_before': account_before,
                'balance_after': account_after,
                'pnl': pnl
            }

        except Exception as e:
            cprint(f"❌ Generator 执行失败: {e}", "red")
            trace.execution_errors.append(str(e))
            trace.execution_success = False
            import traceback
            traceback.print_exc()

        return trace

    async def _fetch_market_data(self, symbols: List[str]) -> dict:
        """获取市场数据（包含技术指标）"""
        market_data = {}

        def safe_float(value, default=0.0):
            """安全地转换为 float"""
            if value is None:
                return default
            # 如果是列表，取第一个元素（最新值）
            if isinstance(value, list):
                if len(value) > 0 and value[0] is not None:
                    return float(value[0])
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        for symbol in symbols:
            try:
                # 1. 获取行情数据
                ticker = await self.exchange_adapter.get_ticker(symbol)

                # 2. 计算技术指标
                indicators = await self._calculate_indicators(symbol)

                market_data[symbol] = {
                    'price': safe_float(ticker.get('last') or ticker.get('close')),
                    'change_24h': safe_float(ticker.get('percentage')),
                    'volume': safe_float(ticker.get('quoteVolume') or ticker.get('volume')),
                    # 优先从indicators获取high/low（更准确）
                    'high_24h': indicators.get('high_24h') or safe_float(ticker.get('high')),
                    'low_24h': indicators.get('low_24h') or safe_float(ticker.get('low')),
                    'indicators': indicators  # 技术指标
                }
            except Exception as e:
                cprint(f"⚠️  获取 {symbol} 数据失败: {e}", "yellow")
                market_data[symbol] = {
                    'price': 0,
                    'change_24h': 0,
                    'volume': 0,
                    'indicators': {}
                }

        return market_data

    async def _calculate_indicators(self, symbol: str, timeframe: str = "5m") -> dict:
        """计算技术指标"""
        try:
            # 延迟导入避免循环依赖
            from ...agents.hansen.market_analyzer import MarketAnalyzer
            from ...models.strategy import TimeFrame

            # 创建市场分析器
            analyzer = MarketAnalyzer(adapter=self.exchange_adapter)

            # 获取K线数据（需要足够数据计算EMA200）
            candles = await self.exchange_adapter.get_candles(symbol, timeframe, limit=250)

            # 转换时间周期
            tf_map = {
                "1m": TimeFrame.M1,
                "5m": TimeFrame.M5,
                "15m": TimeFrame.M15,
                "1h": TimeFrame.H1,
                "4h": TimeFrame.H4,
            }
            tf = tf_map.get(timeframe, TimeFrame.M5)

            # 计算指标
            indicators_map = await analyzer.calculate_indicators(
                symbol=symbol,
                candle_data={tf: candles}
            )

            indicators = indicators_map.get(tf, {})

            def safe_float(value):
                """安全提取指标值（单值）"""
                if value is None:
                    return None
                try:
                    return float(value)
                except:
                    return None

            def safe_float_array(arr, max_len=10):
                """安全提取数组指标值"""
                if arr is None:
                    return None
                # 如果是列表，转换每个元素
                if isinstance(arr, list):
                    result = []
                    for val in arr[:max_len]:  # 限制长度
                        if val is not None:
                            try:
                                result.append(float(val))
                            except:
                                result.append(None)
                        else:
                            result.append(None)
                    return result if result else None
                return None

            # 获取资金费率历史（最近24小时）
            funding_history = []
            try:
                funding_history_raw = await self.exchange_adapter.get_funding_rate_history(symbol)
                # 转换为简化格式，只保留最近10个数据点
                if funding_history_raw:
                    funding_history = [
                        {
                            'time': item['time'],
                            'rate': item['funding_rate']
                        }
                        for item in funding_history_raw[-10:]  # 只取最近10个
                    ]
            except Exception as e:
                cprint(f"⚠️  获取 {symbol} 资金费率历史失败: {e}", "yellow")

            return {
                # 价格数据
                "high_24h": safe_float(indicators.get("high_24h")),
                "low_24h": safe_float(indicators.get("low_24h")),

                # 趋势指标（数组：最近10个点，新到旧）
                "ema_20": safe_float_array(indicators.get("ema_20")),
                "ema_50": safe_float_array(indicators.get("ema_50")),
                "ema_200": safe_float_array(indicators.get("ema_200")),

                # 动量指标（数组：最近10个点）
                "rsi_7": safe_float_array(indicators.get("rsi_7")),
                "rsi_14": safe_float_array(indicators.get("rsi_14")),
                "macd_line": safe_float_array(indicators.get("macd_line")),
                "macd_signal": safe_float_array(indicators.get("macd_signal")),
                "macd_histogram": safe_float_array(indicators.get("macd_histogram")),

                # 波动率指标（单值）
                "atr": safe_float(indicators.get("atr")),
                "atr_percent": safe_float(indicators.get("atr_percent")),

                # 成交量指标（单值）
                "volume_24h": safe_float(indicators.get("volume_24h")),

                # 市场情绪指标（永续合约特有）
                "funding_rate": safe_float(indicators.get("funding_rate")),  # 当前资金费率
                "funding_rate_history": funding_history,  # 资金费率历史（最近10个）
                "open_interest": safe_float(indicators.get("open_interest")),
            }

        except Exception as e:
            cprint(f"⚠️  计算 {symbol} 技术指标失败: {e}", "yellow")
            return {}

    async def _fetch_account_state(self) -> dict:
        """获取账户状态（包含持仓的止盈止损信息）"""
        try:
            # 使用现有的 adapter 方法
            balance = await self.exchange_adapter.get_balance()
            positions = await self.exchange_adapter.get_positions()

            def safe_float(value):
                """安全转换为 float"""
                if value is None:
                    return None
                try:
                    return float(value)
                except:
                    return None

            # 格式化为字典
            return {
                'balance': {
                    'total': float(balance.total),
                    'available': float(balance.available),
                    'frozen': float(balance.frozen)
                },
                'positions': [
                    {
                        'symbol': p.symbol,
                        'side': p.side.value if hasattr(p.side, 'value') else str(p.side),
                        'amount': float(p.amount),
                        'entry_price': float(p.entry_price),
                        'mark_price': safe_float(p.mark_price),
                        'liquidation_price': safe_float(p.liquidation_price),
                        'unrealized_pnl': float(p.unrealized_pnl) if p.unrealized_pnl else 0,
                        'pnl_percentage': p.pnl_percentage if p.pnl_percentage else 0,
                        # 止盈止损信息
                        'stop_loss': safe_float(p.stop_loss),
                        'take_profit': safe_float(p.take_profit),
                        # 杠杆信息
                        'leverage': p.leverage if hasattr(p, 'leverage') else None,
                    }
                    for p in positions
                ],
                'statistics': {}  # 可以添加统计信息
            }
        except Exception as e:
            cprint(f"⚠️  获取账户状态失败: {e}", "yellow")
            return {'balance': {}, 'positions': [], 'statistics': {}}

    def _create_query_text(self, market_data: dict) -> str:
        """创建用于检索的查询文本"""
        parts = []
        for symbol, data in market_data.items():
            price = data.get('price', 0)
            change = data.get('change_24h', 0)

            if change > 5:
                parts.append(f"{symbol} 强势上涨 价格突破")
            elif change < -5:
                parts.append(f"{symbol} 快速下跌 价格回调")
            elif abs(change) < 1:
                parts.append(f"{symbol} 窄幅震荡 横盘整理")
            else:
                parts.append(f"{symbol} 震荡整理")

        return " ".join(parts)

    def _parse_decisions(self, llm_output: str) -> List[TradeDecision]:
        """解析 LLM 输出的多个决策"""
        try:
            # 尝试提取 JSON
            start = llm_output.find('{')
            end = llm_output.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = llm_output[start:end]
                data = json.loads(json_str)

                decisions_data = data.get('decisions', [])

                if not decisions_data:
                    # 兼容旧格式：单个decision
                    decision_data = data.get('decision', {})
                    if decision_data:
                        decisions_data = [decision_data]

                decisions = []
                for dec_data in decisions_data:
                    decision = TradeDecision(
                        action=dec_data.get('action', 'hold'),
                        symbol=dec_data.get('symbol', 'BTC/USDC:USDC'),
                        amount=Decimal(str(dec_data.get('amount', 0))) if dec_data.get('amount') else None,
                        leverage=dec_data.get('leverage'),
                        stop_loss=Decimal(str(dec_data.get('stop_loss'))) if dec_data.get('stop_loss') else None,
                        take_profit=Decimal(str(dec_data.get('take_profit'))) if dec_data.get('take_profit') else None,
                        reasoning=dec_data.get('reasoning', ''),
                        confidence=dec_data.get('confidence', 0.5)
                    )
                    decisions.append(decision)

                return decisions

        except Exception as e:
            cprint(f"⚠️  解析决策失败: {e}, 默认为观望", "yellow")
            import traceback
            traceback.print_exc()

        # 降级：返回观望决策
        return [TradeDecision(
            action='hold',
            symbol='BTC/USDC:USDC',
            reasoning='解析失败，默认观望'
        )]

    async def _execute_trade(self, decision: TradeDecision) -> dict:
        """执行交易"""
        def serialize_result(result):
            """序列化 ExecutionResult 对象为字典"""
            if hasattr(result, 'model_dump'):
                # Pydantic v2
                return result.model_dump(mode='json')
            elif hasattr(result, 'dict'):
                # Pydantic v1
                return result.dict()
            else:
                return str(result)

        try:
            if decision.action == 'open_long':
                cprint(f"📈 开多仓: {decision.symbol}", "green")
                result = await self.exchange_adapter.open_position(
                    symbol=decision.symbol,
                    side='long',
                    amount=float(decision.amount) if decision.amount else 0.01,
                    leverage=decision.leverage or 1,
                    stop_loss=float(decision.stop_loss) if decision.stop_loss else None,
                    take_profit=float(decision.take_profit) if decision.take_profit else None
                )
                return {'success': True, 'result': serialize_result(result), 'action': 'open_long'}

            elif decision.action == 'open_short':
                cprint(f"📉 开空仓: {decision.symbol}", "red")
                result = await self.exchange_adapter.open_position(
                    symbol=decision.symbol,
                    side='short',
                    amount=float(decision.amount) if decision.amount else 0.01,
                    leverage=decision.leverage or 1,
                    stop_loss=float(decision.stop_loss) if decision.stop_loss else None,
                    take_profit=float(decision.take_profit) if decision.take_profit else None
                )
                return {'success': True, 'result': serialize_result(result), 'action': 'open_short'}

            elif decision.action == 'close':
                # 先检查是否有持仓
                position = await self.exchange_adapter.get_position(decision.symbol)

                if not position:
                    cprint(f"⚠️  {decision.symbol}: 无持仓，跳过平仓", "yellow")
                    return {
                        'success': True,
                        'action': 'close',
                        'message': 'No position to close',
                        'skipped': True
                    }

                cprint(f" {decision.symbol}", "yellow")
                result = await self.exchange_adapter.close_position(
                    symbol=decision.symbol
                )
                return {'success': True, 'result': serialize_result(result), 'action': 'close'}

            elif decision.action == 'adjust':
                cprint(f"🔧 调整止损止盈: {decision.symbol}", "cyan")
                # 这里需要实现调整逻辑
                return {'success': True, 'action': 'adjust'}

            else:
                return {'success': False, 'error': f'Unknown action: {decision.action}'}

        except Exception as e:
            cprint(f"❌ 执行交易失败: {e}", "red")
            return {'success': False, 'error': str(e)}

    async def _calculate_account_change(self, before_state: dict, execution_result: dict) -> dict:
        """计算账户变化"""
        try:
            after_state = await self._fetch_account_state()

            balance_before = before_state.get('balance', {}).get('total', 0)
            balance_after = after_state.get('balance', {}).get('total', 0)

            pnl = balance_after - balance_before

            return {
                'balance_before': balance_before,
                'balance_after': balance_after,
                'pnl': pnl
            }
        except Exception as e:
            cprint(f"⚠️  计算账户变化失败: {e}", "yellow")
            return {}
