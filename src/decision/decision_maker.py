"""
决策引擎 (Layer 2)

LLM驱动的市场分析和交易决策
"""
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from termcolor import cprint

from ..engine.market_snapshot import MarketSnapshot


@dataclass
class TradeSignal:
    """交易信号"""
    action: str  # open_long, open_short, close_position, hold, wait
    symbol: str
    amount: Optional[float] = None
    leverage: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0  # 0-100
    reason: str = ""


@dataclass
class Decision:
    """决策结果"""
    decision_type: str  # trade, hold, wait
    signals: List[TradeSignal]
    analysis: str  # LLM的分析文本
    timestamp: datetime

    def has_trade_signals(self) -> bool:
        """是否有交易信号"""
        return len([s for s in self.signals if s.action not in ['hold', 'wait']]) > 0


class DecisionMaker:
    """
    决策引擎

    职责:
    1. 接收预处理的市场快照
    2. 结合历史记忆（来自Layer 3）
    3. 一次性LLM推理生成决策
    4. 解析并返回结构化决策

    不负责:
    - 数据获取（由Layer 1负责）
    - 记忆管理（由Layer 3负责）
    - 订单执行（由Layer 1负责）
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # 加载系统提示词
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "prompts" / "nofn_v2.txt"
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    async def analyze_and_decide(
        self,
        market_snapshot: MarketSnapshot,
        memory_context: Optional[str] = None
    ) -> Decision:
        """
        分析市场并做出决策

        核心改进：
        - 输入：完整的市场快照（已包含所有数据和指标）
        - 过程：只需一次LLM推理
        - 输出：结构化的决策

        Args:
            market_snapshot: 完整的市场快照（来自Layer 1）
            memory_context: 历史记忆上下文（来自Layer 3）

        Returns:
            Decision: 决策结果
        """
        try:
            cprint("=" * 70, "cyan")
            cprint("🧠 LLM 开始分析决策...", "cyan")
            cprint("=" * 70, "cyan")

            # 构建完整的上下文
            context = self._build_context(market_snapshot, memory_context)

            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context}
            ]

            # LLM推理
            response = await self.llm.ainvoke(messages)
            response_text = response.content

            cprint(f"\n{response_text}\n", "white")

            # 解析决策
            decision = self._parse_decision(response_text, market_snapshot)

            cprint("=" * 70, "green")
            cprint(f"✅ 决策完成: {decision.decision_type}", "green")
            if decision.has_trade_signals():
                cprint(f"   交易信号: {len(decision.signals)} 个", "green")
            cprint("=" * 70, "green")

            return decision

        except Exception as e:
            cprint(f"❌ 决策失败: {e}", "red")
            # 返回默认决策：观望
            return Decision(
                decision_type="wait",
                signals=[TradeSignal(action="wait", symbol="", reason=f"Error: {str(e)}")],
                analysis=f"Error occurred: {str(e)}",
                timestamp=datetime.now()
            )


    def _build_context(self, snapshot: MarketSnapshot, memory: Optional[str]) -> str:
        """
        构建完整的上下文

        包含：
        1. 市场快照（所有资产的价格、指标、持仓）
        2. 账户状态
        3. 历史记忆（如果有）
        """
        lines = [
            "请分析以下市场情况并做出交易决策：",
            "",
            "=" * 60,
            snapshot.to_text(),  # 完整的市场快照，文本格式
            "=" * 60,
        ]

        if memory:
            lines.append("")
            lines.append("## 历史记忆")
            lines.append(memory)
            lines.append("=" * 60)

        lines.append("")
        lines.append("## 决策要求")
        lines.append("")
        lines.append("请基于以上信息做出决策，并以以下格式输出：")
        lines.append("")
        lines.append("### 分析总结")
        lines.append("[你的市场分析]")
        lines.append("")
        lines.append("### 决策")
        lines.append("```json")
        lines.append("{")
        lines.append('  "decision_type": "trade | hold | wait",')
        lines.append('  "signals": [')
        lines.append('    {')
        lines.append('      "action": "open_long | open_short | close_position | set_stop_loss | set_take_profit | hold | wait",')
        lines.append('      "symbol": "BTC/USDC:USDC",')
        lines.append('      "amount": 0.001,')
        lines.append('      "leverage": 3,')
        lines.append('      "stop_loss": 88000.0,')
        lines.append('      "take_profit": 96000.0,')
        lines.append('      "confidence": 85,')
        lines.append('      "reason": "原因说明"')
        lines.append('    }')
        lines.append('  ]')
        lines.append("}")
        lines.append("```")
        lines.append("")
        lines.append("注意：")
        lines.append("- 如果决定交易（open_long/open_short），必须提供 stop_loss 和 take_profit")
        lines.append("- **如果action是hold（持有现有仓位）**：")
        lines.append("  * 必须检查当前持仓的止损止盈是否完整")
        lines.append("  * 如果止损缺失，必须在信号中提供合理的stop_loss")
        lines.append("  * 如果止盈缺失，必须在信号中提供合理的take_profit")
        lines.append("  * **如果止损止盈都已设置且合理，保持不变（直接使用当前值）**")
        lines.append("  * **只在显著变化时才调整**（如价格变化>1%，或风险明显增加）")
        lines.append("  * 避免频繁微调，稳定性比完美更重要")
        lines.append("- confidence 表示信心度 (0-100)")
        lines.append("- 如果决定观望，action 填 'wait'")

        return "\n".join(lines)

    def _parse_decision(self, response_text: str, snapshot: MarketSnapshot) -> Decision:
        """
        解析LLM响应，提取决策

        尝试从响应中提取JSON格式的决策
        如果失败，则进行文本分析
        """
        try:
            # 尝试提取JSON块
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 7)

            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start + 7:json_end].strip()
                decision_data = json.loads(json_text)

                # 解析signals
                signals = []
                for sig in decision_data.get('signals', []):
                    signals.append(TradeSignal(
                        action=sig.get('action', 'wait'),
                        symbol=sig.get('symbol', ''),
                        amount=sig.get('amount'),
                        leverage=sig.get('leverage'),
                        stop_loss=sig.get('stop_loss'),
                        take_profit=sig.get('take_profit'),
                        confidence=sig.get('confidence', 0),
                        reason=sig.get('reason', '')
                    ))

                return Decision(
                    decision_type=decision_data.get('decision_type', 'wait'),
                    signals=signals,
                    analysis=response_text[:json_start].strip() if json_start > 0 else response_text,
                    timestamp=datetime.now()
                )

        except Exception as e:
            cprint(f"⚠️  JSON解析失败，尝试文本解析: {e}", "yellow")

        # 文本解析备选方案
        return self._parse_text_decision(response_text, snapshot)

    def _parse_text_decision(self, text: str, snapshot: MarketSnapshot) -> Decision:
        """
        文本解析决策

        当JSON解析失败时的备选方案
        """
        text_lower = text.lower()

        # 检测决策类型
        if any(keyword in text_lower for keyword in ['开多', '开仓', 'open long', 'buy']):
            # 尝试提取交易对和参数
            signals = self._extract_trade_signals_from_text(text, 'open_long', snapshot)
            return Decision(
                decision_type="trade" if signals else "wait",
                signals=signals if signals else [TradeSignal(action="wait", symbol="", reason="未能解析交易参数")],
                analysis=text,
                timestamp=datetime.now()
            )

        elif any(keyword in text_lower for keyword in ['开空', 'open short', 'short']):
            signals = self._extract_trade_signals_from_text(text, 'open_short', snapshot)
            return Decision(
                decision_type="trade" if signals else "wait",
                signals=signals if signals else [TradeSignal(action="wait", symbol="", reason="未能解析交易参数")],
                analysis=text,
                timestamp=datetime.now()
            )

        elif any(keyword in text_lower for keyword in ['平仓', 'close', '止盈', '止损']):
            # 平仓或调整止损止盈
            signals = self._extract_close_signals_from_text(text, snapshot)
            return Decision(
                decision_type="trade" if signals else "wait",
                signals=signals if signals else [TradeSignal(action="wait", symbol="", reason="未能解析平仓参数")],
                analysis=text,
                timestamp=datetime.now()
            )

        else:
            # 默认：观望或持有
            return Decision(
                decision_type="hold",
                signals=[TradeSignal(action="hold", symbol="", reason="继续持有现有仓位或观望")],
                analysis=text,
                timestamp=datetime.now()
            )

    def _extract_trade_signals_from_text(self, text: str, action: str, snapshot: MarketSnapshot) -> List[TradeSignal]:
        """从文本中提取交易信号（简化版）"""
        # 这里可以实现更复杂的文本解析
        # 暂时返回空，强制LLM使用JSON格式
        return []

    def _extract_close_signals_from_text(self, text: str, snapshot: MarketSnapshot) -> List[TradeSignal]:
        """从文本中提取平仓信号（简化版）"""
        # 检查是否有持仓需要平仓或调整
        signals = []
        for asset in snapshot.get_positions():
            if asset.symbol.upper() in text.upper() or asset.symbol.split('/')[0] in text.upper():
                signals.append(TradeSignal(
                    action="close_position",
                    symbol=asset.symbol,
                    reason="根据分析决定平仓"
                ))
        return signals
