"""LangGraph 交易状态定义。

定义交易决策图的 State 结构，所有状态都会被 Checkpointer 自动持久化。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from operator import add


class DecisionMemory(TypedDict):
    """单条决策记忆。

    记录一次决策周期的关键信息，由 LangGraph 自动持久化。
    """
    cycle_index: int
    timestamp_ms: int
    action: Optional[str]  # TradeDecisionAction.value
    symbol: Optional[str]
    quantity: Optional[float]
    rationale: Optional[str]
    confidence: Optional[float]
    executed: bool
    exec_price: Optional[float]
    realized_pnl: Optional[float]


def _merge_memories(
    existing: List[DecisionMemory],
    new: List[DecisionMemory],
) -> List[DecisionMemory]:
    """合并决策记忆，保留最近 10 条。

    LangGraph reducer 函数，用于更新 State 中的 memories。
    """
    combined = existing + new
    # 保留最近 10 条
    return combined[-10:] if len(combined) > 10 else combined


def _merge_signals(
    existing: Dict[str, str],
    new: Dict[str, str],
) -> Dict[str, str]:
    """合并待观察信号。

    新的信号会覆盖旧的，value 为空字符串表示删除。
    """
    result = dict(existing)
    for k, v in new.items():
        if v:
            result[k] = v
        else:
            result.pop(k, None)
    return result


class TradingState(TypedDict):
    """交易决策图的状态。

    所有字段都会被 LangGraph Checkpointer 自动持久化。
    通过 thread_id（strategy_id）可以恢复策略的完整状态。

    状态分为三类：
    1. 周期数据：每次决策周期更新的临时数据
    2. 记忆数据：跨周期保留的短期记忆（由 reducer 管理）
    3. 配置数据：策略级别的配置（初始化时设置）
    """

    # =========================================================================
    # 周期数据（每次 invoke 时传入，不跨周期保留）
    # =========================================================================

    # 基本信息
    compose_id: str
    timestamp_ms: int

    # 市场数据（由外部传入）
    features: List[Dict[str, Any]]  # FeatureVector 的 dict 形式
    portfolio: Dict[str, Any]  # PortfolioView 的 dict 形式
    digest: Dict[str, Any]  # TradeDigest 的 dict 形式

    # 决策结果（由 decide 节点填充）
    instructions: List[Dict[str, Any]]  # TradeInstruction 的 dict 形式
    rationale: Optional[str]

    # 执行结果（由 execute 节点填充）
    trades: List[Dict[str, Any]]  # TradeHistoryEntry 的 dict 形式

    # =========================================================================
    # 记忆数据（跨周期保留，由 Checkpointer 持久化）
    # =========================================================================

    # 决策历史（最近 10 条）
    # 使用 Annotated + reducer 实现自动合并
    memories: Annotated[List[DecisionMemory], _merge_memories]

    # 待观察信号
    # 使用 Annotated + reducer 实现自动合并
    pending_signals: Annotated[Dict[str, str], _merge_signals]

    # 周期计数器
    cycle_index: int

    # =========================================================================
    # 配置数据（策略级别，不变）
    # =========================================================================

    strategy_id: str


def create_initial_state(
    strategy_id: str,
    compose_id: str,
    timestamp_ms: int,
    features: List[Dict[str, Any]],
    portfolio: Dict[str, Any],
    digest: Dict[str, Any],
) -> TradingState:
    """创建初始状态。

    用于首次运行或没有 checkpoint 时。
    """
    return TradingState(
        # 周期数据
        compose_id=compose_id,
        timestamp_ms=timestamp_ms,
        features=features,
        portfolio=portfolio,
        digest=digest,
        instructions=[],
        rationale=None,
        trades=[],
        # 记忆数据（初始为空，后续由 reducer 累加）
        memories=[],
        pending_signals={},
        cycle_index=0,
        # 配置
        strategy_id=strategy_id,
    )


def state_to_compose_context(state: TradingState) -> Dict[str, Any]:
    """从 State 提取 ComposeContext 所需的数据。

    用于构建 LLM 决策的上下文。
    """
    # 从 memories 提取最近 5 条用于 prompt
    recent_decisions = []
    for m in state["memories"][-5:]:
        recent_decisions.append({
            "cycle": m["cycle_index"],
            "action": m["action"],
            "symbol": m["symbol"],
            "executed": m["executed"],
            "exec_price": m.get("exec_price"),
            "realized_pnl": m.get("realized_pnl"),
            "reason": m.get("rationale", "")[:100] if m.get("rationale") else None,
        })

    return {
        "ts": state["timestamp_ms"],
        "compose_id": state["compose_id"],
        "strategy_id": state["strategy_id"],
        "features": state["features"],
        "portfolio": state["portfolio"],
        "digest": state["digest"],
        "recent_decisions": recent_decisions,
        "pending_signals": dict(state["pending_signals"]),
    }


def create_memory_from_result(
    cycle_index: int,
    timestamp_ms: int,
    instructions: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    rationale: Optional[str],
) -> List[DecisionMemory]:
    """从决策结果创建记忆条目。

    如果没有指令，创建一个 NOOP 记忆；否则为每个指令创建一条记忆。
    """
    memories: List[DecisionMemory] = []

    if not instructions:
        # NOOP
        memories.append(DecisionMemory(
            cycle_index=cycle_index,
            timestamp_ms=timestamp_ms,
            action="noop",
            symbol=None,
            quantity=None,
            rationale=rationale,
            confidence=None,
            executed=False,
            exec_price=None,
            realized_pnl=None,
        ))
        return memories

    # 构建 trade 映射
    trade_map = {}
    for t in trades:
        if t.get("instruction_id"):
            trade_map[t["instruction_id"]] = t

    for inst in instructions:
        inst_id = inst.get("instruction_id")
        trade = trade_map.get(inst_id) if inst_id else None

        meta = inst.get("meta") or {}

        memories.append(DecisionMemory(
            cycle_index=cycle_index,
            timestamp_ms=timestamp_ms,
            action=inst.get("action"),
            symbol=inst.get("instrument", {}).get("symbol"),
            quantity=inst.get("quantity"),
            rationale=meta.get("rationale"),
            confidence=meta.get("confidence"),
            executed=trade is not None,
            exec_price=trade.get("avg_exec_price") if trade else None,
            realized_pnl=trade.get("realized_pnl") if trade else None,
        ))

    return memories
