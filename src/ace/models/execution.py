"""
Execution Trace - Generator 的执行轨迹
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class TradeDecision:
    """
    交易决策（Generator 的输出）
    """
    action: str  # "open_long", "open_short", "close", "hold", "adjust"
    symbol: str
    amount: Optional[Decimal] = None
    price: Optional[Decimal] = None
    leverage: Optional[int] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # 推理信息
    reasoning: str = ""  # LLM 的推理过程
    confidence: float = 0.5  # 决策置信度
    used_entry_ids: List[str] = field(default_factory=list)  # 使用了哪些知识条目

    def to_dict(self) -> dict:
        return {
            'action': self.action,
            'symbol': self.symbol,
            'amount': str(self.amount) if self.amount else None,
            'price': str(self.price) if self.price else None,
            'leverage': self.leverage,
            'stop_loss': str(self.stop_loss) if self.stop_loss else None,
            'take_profit': str(self.take_profit) if self.take_profit else None,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'used_entry_ids': self.used_entry_ids,
        }


@dataclass
class ExecutionTrace:
    """
    执行轨迹（Generator 的完整记录）

    包含：市场输入 → LLM 推理 → 交易决策 → 执行结果
    """

    # 唯一标识
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.now)

    # 输入：市场数据
    market_data: Dict[str, Any] = field(default_factory=dict)  # 价格、指标、K线等
    account_state: Dict[str, Any] = field(default_factory=dict)  # 余额、持仓等

    # Generator 使用的知识条目
    retrieved_entries: List[str] = field(default_factory=list)  # entry_ids

    # 输出：决策（支持多个币对）
    decisions: List[TradeDecision] = field(default_factory=list)

    # 执行结果
    execution_success: bool = False
    execution_results: List[Dict[str, Any]] = field(default_factory=list)  # 多个订单的执行结果
    execution_errors: List[str] = field(default_factory=list)  # 多个错误信息

    # 账户变化（执行后）
    account_change: Dict[str, Any] = field(default_factory=dict)  # PnL, 余额变化等

    # 完整的 LLM 原始输出（调试用）
    raw_llm_output: str = ""

    def to_dict(self) -> dict:
        """序列化"""
        return {
            'trace_id': self.trace_id,
            'timestamp': self.timestamp.isoformat(),
            'market_data': self.market_data,
            'account_state': self.account_state,
            'retrieved_entries': self.retrieved_entries,
            'decisions': [d.to_dict() for d in self.decisions],
            'execution_success': self.execution_success,
            'execution_results': self.execution_results,
            'execution_errors': self.execution_errors,
            'account_change': self.account_change,
            'raw_llm_output': self.raw_llm_output,
        }

    @property
    def is_profitable(self) -> Optional[bool]:
        """判断是否盈利"""
        if 'pnl' in self.account_change:
            return float(self.account_change['pnl']) > 0
        return None

    @property
    def pnl(self) -> Optional[Decimal]:
        """获取盈亏"""
        if 'pnl' in self.account_change:
            return Decimal(str(self.account_change['pnl']))
        return None

    def __repr__(self) -> str:
        status = "✅" if self.execution_success else "❌"
        actions = [d.action for d in self.decisions] if self.decisions else ["no_decisions"]
        actions_str = ", ".join(actions)
        return f"ExecutionTrace({self.trace_id[:8]}... {status} [{actions_str}])"
