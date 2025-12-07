"""LangGraph 交易状态定义。

定义交易决策图的 State 结构，所有状态都会被 Checkpointer 自动持久化。
支持反思模式，将反思洞察作为状态的一部分进行管理。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
from operator import add


class DecisionSummary(TypedDict):
    """决策历史摘要。

    当 memories 累积到一定数量时，旧的记忆会被压缩成摘要。
    摘要由 LLM 生成，包含关键的决策模式和教训。
    """
    # 摘要覆盖的周期范围 [start_cycle, end_cycle]
    cycle_range: Tuple[int, int]
    # 摘要覆盖的时间范围 [start_ts, end_ts]
    time_range: Tuple[int, int]
    # 摘要内容（由 LLM 生成）
    content: str
    # 关键统计
    total_decisions: int
    executed_count: int
    total_pnl: float


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


def _merge_summaries(
    existing: List[DecisionSummary],
    new: List[DecisionSummary],
) -> List[DecisionSummary]:
    """合并决策摘要，保留最近 5 个摘要。

    每个摘要覆盖一段历史周期，多个摘要组成完整的历史视图。
    """
    combined = existing + new
    # 保留最近 5 个摘要（覆盖约 25-50 个周期的历史）
    return combined[-5:] if len(combined) > 5 else combined


class ReflectionState(TypedDict, total=False):
    """反思状态（可选字段）。

    当启用反思模式时，这些字段会被填充。
    使用 total=False 使所有字段可选。
    """
    # 当前反思结果（由 reflect 节点填充）
    reflection_insight: Optional[Dict[str, Any]]  # ReflectionInsight 的 dict 形式
    # 剩余冷静周期（每个周期递减）
    cooldown_remaining: int
    # 是否处于冷静期
    in_cooldown: bool
    # 最近一次反思的周期索引
    last_reflection_cycle: int


class TradingState(TypedDict):
    """交易决策图的状态。

    所有字段都会被 LangGraph Checkpointer 自动持久化。
    通过 thread_id（strategy_id）可以恢复策略的完整状态。

    状态分为四类：
    1. 周期数据：每次决策周期更新的临时数据
    2. 反思数据：反思节点产生的洞察和冷静期状态
    3. 记忆数据：跨周期保留的短期记忆（由 reducer 管理）
    4. 配置数据：策略级别的配置（初始化时设置）
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

    # 历史记录（由外部传入，用于反思分析）
    history_records: List[Dict[str, Any]]  # HistoryRecord 的 dict 形式

    # 决策结果（由 decide 节点填充）
    instructions: List[Dict[str, Any]]  # TradeInstruction 的 dict 形式
    rationale: Optional[str]

    # 执行结果（由 execute 节点填充）
    trades: List[Dict[str, Any]]  # TradeHistoryEntry 的 dict 形式

    # =========================================================================
    # 反思数据（由 reflect 节点填充，跨周期保留部分状态）
    # =========================================================================

    # 当前周期的反思洞察（由 reflect 节点生成）
    reflection_insight: Optional[Dict[str, Any]]  # ReflectionInsight 的 dict 形式

    # 冷静期状态（跨周期保留，每周期递减）
    cooldown_remaining: int

    # 最近一次反思的周期索引（用于判断是否需要新反思）
    last_reflection_cycle: int

    # =========================================================================
    # 记忆数据（跨周期保留，由 Checkpointer 持久化）
    # =========================================================================

    # 决策历史（最近 10 条详细记录）
    # 使用 Annotated + reducer 实现自动合并
    memories: Annotated[List[DecisionMemory], _merge_memories]

    # 历史摘要（压缩的旧决策，最多 5 个摘要）
    # 当 memories 满时，旧记忆被压缩成摘要
    summaries: Annotated[List[DecisionSummary], _merge_summaries]

    # 待观察信号
    # 使用 Annotated + reducer 实现自动合并
    pending_signals: Annotated[Dict[str, str], _merge_signals]

    # 周期计数器
    cycle_index: int

    # =========================================================================
    # 配置数据（策略级别，不变）
    # =========================================================================

    strategy_id: str

    # 是否启用反思模式
    reflection_enabled: bool


def create_initial_state(
    strategy_id: str,
    compose_id: str,
    timestamp_ms: int,
    features: List[Dict[str, Any]],
    portfolio: Dict[str, Any],
    digest: Dict[str, Any],
    history_records: Optional[List[Dict[str, Any]]] = None,
    reflection_enabled: bool = False,
) -> TradingState:
    """创建初始状态。

    用于首次运行或没有 checkpoint 时。

    Args:
        strategy_id: 策略 ID
        compose_id: 本次决策 ID
        timestamp_ms: 时间戳
        features: 特征向量列表
        portfolio: 投资组合视图
        digest: 交易摘要
        history_records: 历史记录（用于反思分析）
        reflection_enabled: 是否启用反思模式
    """
    return TradingState(
        # 周期数据
        compose_id=compose_id,
        timestamp_ms=timestamp_ms,
        features=features,
        portfolio=portfolio,
        digest=digest,
        history_records=history_records or [],
        instructions=[],
        rationale=None,
        trades=[],
        # 反思数据
        reflection_insight=None,
        cooldown_remaining=0,
        last_reflection_cycle=0,
        # 记忆数据（初始为空，后续由 reducer 累加）
        memories=[],
        summaries=[],
        pending_signals={},
        cycle_index=0,
        # 配置
        strategy_id=strategy_id,
        reflection_enabled=reflection_enabled,
    )


def state_to_compose_context(state: TradingState) -> Dict[str, Any]:
    """从 State 提取 ComposeContext 所需的数据。

    用于构建 LLM 决策的上下文。
    包含反思洞察（如果启用反思模式）。
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

    # 从 summaries 提取历史摘要
    history_summaries = []
    for s in state.get("summaries", []):
        history_summaries.append({
            "cycle_range": s["cycle_range"],
            "content": s["content"],
            "stats": {
                "decisions": s["total_decisions"],
                "executed": s["executed_count"],
                "pnl": s["total_pnl"],
            },
        })

    context = {
        "ts": state["timestamp_ms"],
        "compose_id": state["compose_id"],
        "strategy_id": state["strategy_id"],
        "features": state["features"],
        "portfolio": state["portfolio"],
        "digest": state["digest"],
        "recent_decisions": recent_decisions,
        "history_summaries": history_summaries,
        "pending_signals": dict(state["pending_signals"]),
    }

    # 添加反思上下文（如果有）
    if state.get("reflection_enabled"):
        context["reflection"] = {
            "insight": state.get("reflection_insight"),
            "cooldown_remaining": state.get("cooldown_remaining", 0),
            "in_cooldown": state.get("cooldown_remaining", 0) > 0,
        }

    return context


def should_generate_summary(memories: List[DecisionMemory]) -> bool:
    """判断是否需要生成摘要。

    当 memories 达到 10 条时，需要将前 5 条压缩成摘要。
    """
    return len(memories) >= 10


def prepare_memories_for_summary(
    memories: List[DecisionMemory],
) -> Tuple[List[DecisionMemory], List[DecisionMemory]]:
    """准备用于摘要的记忆。

    将 memories 分为两部分：
    - to_summarize: 需要压缩成摘要的旧记忆（前 5 条）
    - to_keep: 保留的新记忆（后 5 条）

    Returns:
        (to_summarize, to_keep)
    """
    if len(memories) < 10:
        return [], memories

    # 前 5 条压缩，后 5 条保留
    to_summarize = memories[:5]
    to_keep = memories[5:]
    return to_summarize, to_keep


def create_summary_from_memories(
    memories: List[DecisionMemory],
    summary_content: str,
) -> DecisionSummary:
    """从记忆列表创建摘要对象。

    Args:
        memories: 要压缩的记忆列表
        summary_content: LLM 生成的摘要内容

    Returns:
        DecisionSummary 对象
    """
    if not memories:
        raise ValueError("memories cannot be empty")

    # 计算统计信息
    cycle_range = (memories[0]["cycle_index"], memories[-1]["cycle_index"])
    time_range = (memories[0]["timestamp_ms"], memories[-1]["timestamp_ms"])
    total_decisions = len(memories)
    executed_count = sum(1 for m in memories if m["executed"])
    total_pnl = sum(m.get("realized_pnl") or 0.0 for m in memories)

    return DecisionSummary(
        cycle_range=cycle_range,
        time_range=time_range,
        content=summary_content,
        total_decisions=total_decisions,
        executed_count=executed_count,
        total_pnl=total_pnl,
    )


def format_memories_for_summarization(memories: List[DecisionMemory]) -> str:
    """将记忆格式化为文本，用于 LLM 摘要生成。

    Args:
        memories: 要格式化的记忆列表

    Returns:
        格式化的文本
    """
    lines = []
    for m in memories:
        action = m.get("action", "unknown")
        symbol = m.get("symbol", "N/A")
        executed = "已执行" if m.get("executed") else "未执行"
        price = m.get("exec_price")
        pnl = m.get("realized_pnl")
        reason = m.get("rationale", "")[:80] if m.get("rationale") else ""

        line = f"- 周期{m['cycle_index']}: {action} {symbol}, {executed}"
        if price:
            line += f", 价格={price:.2f}"
        if pnl:
            line += f", PnL={pnl:.4f}"
        if reason:
            line += f" | {reason}"
        lines.append(line)

    return "\n".join(lines)


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
