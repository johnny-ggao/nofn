"""LangGraph 交易决策工作流。

定义交易决策图的节点和边，以及 Checkpointer 和 Store 配置。

图结构（集成反思模式）：

    START -> reflect -> should_cooldown?
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
             cooldown_noop              decide
                  |                       |
                  v                       v
                 END               execute -> record -> should_summarize?
                                                              |
                                                  +-----------+-----------+
                                                  |                       |
                                                  v                       v
                                             summarize                   END
                                                  |
                                                  v
                                                 END

反思模式说明：
    - reflect 节点：分析近期交易表现，生成洞察和建议
    - 冷静期处理：当反思建议暂停交易时，直接返回 noop
    - 反思洞察注入到 decide 节点的上下文中

记忆层次：
    - 短期记忆 (memories): LangGraph State + Checkpointer
    - 中期记忆 (summaries): LLM 压缩的摘要
    - 长期记忆 (Store): 跨会话持久化
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from termcolor import cprint

from .state import (
    TradingState,
    create_memory_from_result,
    state_to_compose_context,
    should_generate_summary,
    prepare_memories_for_summary,
    create_summary_from_memories,
    format_memories_for_summarization,
)
from .long_term_memory import (
    LongTermMemoryManager,
    MemoryType,
    get_memory_namespace,
)


@dataclass
class TradingGraphConfig:
    """交易图配置。"""

    # 数据库路径
    db_path: str = "data/trading_graph.db"

    # 是否启用持久化（测试时可关闭）
    enable_persistence: bool = True

    # 触发摘要的 memories 数量阈值
    summary_threshold: int = 10

    # 是否启用反思模式
    enable_reflection: bool = False

    # 反思间隔（每隔多少周期触发一次反思）
    reflection_interval: int = 5


class TradingGraphNodes:
    """交易图节点实现。

    节点是纯函数，接收 State 并返回更新的字段。
    实际的业务逻辑（LLM 调用、交易执行）由外部注入的回调执行。

    图结构（集成反思模式）：
        START -> reflect -> [冷静期?] -> decide/cooldown_noop -> execute -> record -> [摘要?] -> summarize? -> END

    记忆集成：
        - reflect 节点：分析历史，生成反思洞察
        - decide 节点：从 Store 检索相关长期记忆，注入到上下文（包含反思洞察）
        - summarize 节点：当 summaries 满时，归档最旧的到 Store
    """

    # summaries 最大数量（与 state.py 中 _merge_summaries 保持一致）
    MAX_SUMMARIES = 5

    def __init__(
        self,
        *,
        decide_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
        execute_callback: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Dict[str, Any]]],
        summarize_callback: Optional[Callable[[str], str]] = None,
        reflect_callback: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]] = None,
        summary_threshold: int = 10,
        reflection_interval: int = 5,
    ):
        """初始化节点。

        Args:
            decide_callback: 决策回调，接收 compose_context，返回 {instructions, rationale}
            execute_callback: 执行回调，接收 (instructions, market_features)，返回 trades
            summarize_callback: 摘要回调，接收 memories_text，返回 summary_content
            reflect_callback: 反思回调，接收 (digest, history_records)，返回 ReflectionInsight.dict()
            summary_threshold: 触发摘要的 memories 数量阈值
            reflection_interval: 反思间隔（每隔多少周期触发一次反思）
        """
        self._decide_callback = decide_callback
        self._execute_callback = execute_callback
        self._summarize_callback = summarize_callback
        self._reflect_callback = reflect_callback
        self._summary_threshold = summary_threshold
        self._reflection_interval = reflection_interval

    async def reflect(self, state: TradingState) -> Dict[str, Any]:
        """反思节点：分析近期交易表现，生成洞察。

        输入：State 中的 digest 和 history_records
        输出：reflection_insight, cooldown_remaining（更新）

        反思逻辑：
            1. 检查是否需要触发反思（基于周期间隔或冷静期）
            2. 如果处于冷静期，递减 cooldown_remaining
            3. 如果需要反思，调用 reflect_callback 生成洞察
            4. 根据洞察设置新的冷静期
        """
        # 如果未启用反思模式，直接跳过
        if not state.get("reflection_enabled", False):
            return {}

        cycle_index = state.get("cycle_index", 0)
        cooldown_remaining = state.get("cooldown_remaining", 0)
        last_reflection_cycle = state.get("last_reflection_cycle", 0)

        # 如果处于冷静期，递减并返回
        if cooldown_remaining > 0:
            new_cooldown = cooldown_remaining - 1
            cprint(f"冷静期中：剩余 {new_cooldown} 个周期", "magenta")
            return {
                "cooldown_remaining": new_cooldown,
            }

        # 判断是否需要触发反思（每隔 N 个周期）
        cycles_since_last = cycle_index - last_reflection_cycle
        if cycles_since_last < self._reflection_interval:
            cprint(
                f"跳过反思：距上次反思仅 {cycles_since_last} 个周期，"
                f"间隔要求 {self._reflection_interval}"
            , "magenta")
            return {}

        # 无反思回调时跳过
        if self._reflect_callback is None:
            cprint("跳过反思：未配置 reflect_callback", "magenta")
            return {}

        # 执行反思
        try:
            digest = state.get("digest", {})
            history_records = state.get("history_records", [])

            insight = await self._reflect_callback(digest, history_records)

            if not insight:
                cprint("反思回调返回空结果", "magenta")
                return {"last_reflection_cycle": cycle_index}

            # 提取冷静期建议
            new_cooldown = insight.get("cooldown_cycles", 0)

            cprint(
                f"反思完成：周期 {cycle_index}, "
                f"警报数={len(insight.get('alerts', []))}, "
                f"冷静期={new_cooldown}",
                "cyan"
            )

            return {
                "reflection_insight": insight,
                "cooldown_remaining": new_cooldown,
                "last_reflection_cycle": cycle_index,
            }

        except Exception as e:
            cprint(f"反思失败: {e}", "yellow")
            return {}

    async def cooldown_noop(self, state: TradingState) -> Dict[str, Any]:
        """冷静期 NOOP 节点：在冷静期内跳过交易。

        输入：State
        输出：instructions=[], rationale（说明冷静期原因）

        当反思建议暂停交易时，不调用 LLM，直接返回 noop。
        """
        cooldown_remaining = state.get("cooldown_remaining", 0)
        insight = state.get("reflection_insight", {})

        # 生成冷静期说明
        summary = insight.get("summary", "") if insight else ""
        rationale = (
            f"[冷静期] 剩余 {cooldown_remaining} 个周期。"
            f"{summary[:100] if summary else '反思建议暂停交易。'}"
        )

        cprint(f"冷静期 NOOP: {rationale}", "white")

        return {
            "instructions": [],
            "rationale": rationale,
            "trades": [],
        }

    async def decide(
        self, state: TradingState, *, store: Optional[BaseStore] = None
    ) -> Dict[str, Any]:
        """决策节点：调用 LLM 生成交易指令。

        输入：State 中的市场数据、记忆和反思洞察
        输出：instructions, rationale

        上下文集成：
            - 从 Store 检索相关的历史经验
            - 注入到 context 的 long_term_memories 字段
            - 反思洞察自动通过 state_to_compose_context 注入
        """
        # 构建 ComposeContext（自动包含反思上下文）
        context = state_to_compose_context(state)

        # 从 Store 检索长期记忆
        if store is not None:
            try:
                strategy_id = state["strategy_id"]
                long_term_memories = await self._retrieve_long_term_memories(
                    store, strategy_id
                )
                if long_term_memories:
                    context["long_term_memories"] = long_term_memories
                    cprint(
                        f"注入 {len(long_term_memories)} 条长期记忆到决策上下文"
                    )
            except Exception as e:
                cprint(f"检索长期记忆失败: {e}", "yellow")

        # 调用决策回调
        result = await self._decide_callback(context)

        return {
            "instructions": result.get("instructions", []),
            "rationale": result.get("rationale"),
        }

    async def _retrieve_long_term_memories(
        self, store: BaseStore, strategy_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """从 Store 检索长期记忆。

        按重要性排序，返回最相关的记忆。
        """
        memories: List[Dict[str, Any]] = []

        # 遍历所有记忆类型
        for memory_type in MemoryType:
            namespace = get_memory_namespace(strategy_id, memory_type)
            try:
                results = await store.asearch(namespace, limit=limit)
                for item in results:
                    memory = dict(item.value)
                    memory["memory_id"] = item.key
                    memories.append(memory)
            except Exception as e:
                cprint(f"搜索 {memory_type.value} 记忆失败: {e}", "magenta")

        # 按重要性排序
        memories.sort(key=lambda m: m.get("importance", 0), reverse=True)
        return memories[:limit]

    async def execute(self, state: TradingState) -> Dict[str, Any]:
        """执行节点：执行交易指令。

        输入：State 中的 instructions
        输出：trades
        """
        instructions = state.get("instructions", [])
        if not instructions:
            return {"trades": []}

        # 提取市场特征用于定价
        market_features = [
            f for f in state.get("features", [])
            if (f.get("meta") or {}).get("source") == "market_snapshot"
        ]

        # 调用执行回调
        trades = await self._execute_callback(instructions, market_features)

        return {"trades": trades}

    async def record(self, state: TradingState) -> Dict[str, Any]:
        """记录节点：将决策结果记录到记忆。

        输入：State 中的 instructions, trades, rationale
        输出：memories（增量更新），cycle_index

        注意：不再在此节点处理摘要，摘要由独立的 summarize 节点处理。
        """
        new_memories = create_memory_from_result(
            cycle_index=state["cycle_index"] + 1,
            timestamp_ms=state["timestamp_ms"],
            instructions=state.get("instructions", []),
            trades=state.get("trades", []),
            rationale=state.get("rationale"),
        )

        return {
            "memories": new_memories,  # reducer 会自动合并
            "cycle_index": state["cycle_index"] + 1,
        }

    async def summarize(
        self, state: TradingState, *, store: Optional[BaseStore] = None
    ) -> Dict[str, Any]:
        """摘要节点：压缩旧的决策记忆。

        当 memories 数量达到阈值时，将前半部分压缩成摘要。

        输入：State 中的 memories
        输出：summaries（新增摘要）

        长期记忆集成：
            - 当 summaries 即将满时，归档最旧的摘要到 Store
            - 这样可以保证历史经验不丢失

        注意：memories 的裁剪由 reducer 自动处理（保留最近10条）。
        """
        memories = state.get("memories", [])
        summaries = state.get("summaries", [])

        # 取前半部分用于摘要
        to_summarize, _ = prepare_memories_for_summary(memories)

        if not to_summarize:
            cprint("无需摘要：没有足够的历史记忆", "magenta")
            return {}

        # 生成摘要内容
        memories_text = format_memories_for_summarization(to_summarize)

        if self._summarize_callback:
            try:
                summary_content = await self._summarize_callback(memories_text)
            except Exception as e:
                cprint(f"摘要回调失败，使用默认摘要: {e}", "yellow")
                summary_content = self._generate_simple_summary(to_summarize)
        else:
            summary_content = self._generate_simple_summary(to_summarize)

        # 创建摘要对象
        summary = create_summary_from_memories(to_summarize, summary_content)

        cprint(
            f"生成决策摘要: 周期{summary['cycle_range'][0]}-{summary['cycle_range'][1]}, "
            f"{summary['total_decisions']}次决策"
        , "cyan")

        # 归档最旧的摘要到长期记忆（当 summaries 即将满时）
        # 新摘要加入后如果会超过 MAX_SUMMARIES，需要先归档最旧的
        if store is not None and len(summaries) >= self.MAX_SUMMARIES:
            oldest_summary = summaries[0]
            try:
                await self._archive_summary_to_store(
                    store, state["strategy_id"], oldest_summary
                )
            except Exception as e:
                cprint(f"归档摘要到长期记忆失败: {e}", "yellow")

        return {"summaries": [summary]}

    async def _archive_summary_to_store(
        self, store: BaseStore, strategy_id: str, summary: Dict[str, Any]
    ) -> str:
        """将摘要归档到长期记忆 Store。

        Args:
            store: LangGraph Store
            strategy_id: 策略 ID
            summary: DecisionSummary 字典

        Returns:
            记忆 ID
        """
        import uuid

        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        namespace = get_memory_namespace(strategy_id, MemoryType.CASE)

        # 计算重要性
        total_pnl = summary.get("total_pnl", 0.0)
        pnl_factor = min(abs(total_pnl) / 100, 1.0)
        importance = 0.5 + 0.5 * pnl_factor

        cycle_range = summary.get("cycle_range", (0, 0))

        value = {
            "type": MemoryType.CASE.value,
            "content": summary.get("content", ""),
            "importance": importance,
            "source_cycles": list(range(cycle_range[0], cycle_range[1] + 1)),
            "source_summary": summary.get("content"),
            "total_pnl": total_pnl,
            "access_count": 0,
            "metadata": {
                "total_decisions": summary.get("total_decisions", 0),
                "executed_count": summary.get("executed_count", 0),
                "time_range": summary.get("time_range"),
            },
        }

        await store.aput(namespace, memory_id, value)

        cprint(
            f"归档摘要为长期记忆: {memory_id}, "
            f"cycles={cycle_range}, pnl={total_pnl:.4f}"
        , "cyan")

        return memory_id

    def _generate_simple_summary(self, memories: List[Dict[str, Any]]) -> str:
        """生成简单的统计摘要（不依赖 LLM）。"""
        if not memories:
            return "无历史决策记录。"

        total = len(memories)
        executed = sum(1 for m in memories if m.get("executed"))
        total_pnl = sum(m.get("realized_pnl") or 0.0 for m in memories)

        actions = {}
        for m in memories:
            action = m.get("action")
            actions[action] = actions.get(action, 0) + 1

        action_desc = ", ".join(f"{k}={v}次" for k, v in actions.items())
        pnl_desc = f"盈利{total_pnl:.4f}" if total_pnl >= 0 else f"亏损{abs(total_pnl):.4f}"

        cycle_start = memories[0]["cycle_index"]
        cycle_end = memories[-1]["cycle_index"]

        return (
            f"周期{cycle_start}-{cycle_end}共{total}次决策，执行{executed}次。"
            f"行为分布：{action_desc}。累计{pnl_desc}。"
        )


def _should_cooldown(state: TradingState) -> Literal["cooldown", "decide"]:
    """条件边：判断是否处于冷静期。

    当 cooldown_remaining > 0 时，路由到 cooldown_noop 节点。
    """
    cooldown_remaining = state.get("cooldown_remaining", 0)
    if cooldown_remaining > 0:
        cprint(f"冷静期中：剩余 {cooldown_remaining} 个周期，跳过决策", "magenta")
        return "cooldown"
    return "decide"


def _should_summarize(state: TradingState) -> Literal["summarize", "end"]:
    """条件边：判断是否需要生成摘要。

    当 memories 数量达到阈值时，路由到 summarize 节点。
    """
    memories = state.get("memories", [])
    if should_generate_summary(memories):
        cprint(f"触发摘要：memories 数量={len(memories)}")
        return "summarize"
    return "end"


def create_trading_graph(
    *,
    decide_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
    execute_callback: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Dict[str, Any]]],
    summarize_callback: Optional[Callable[[str], str]] = None,
    reflect_callback: Optional[Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    summary_threshold: int = 10,
    reflection_interval: int = 5,
    enable_reflection: bool = False,
):
    """创建交易决策图。

    图结构（集成反思模式）：

        START -> reflect -> [should_cooldown?]
                                  |
                      +-----------+-----------+
                      |                       |
                      v                       v
                 cooldown_noop              decide
                      |                       |
                      v                       v
                    record            execute -> record -> [should_summarize?]
                      |                                          |
                      v                              +-----------+-----------+
                     END                             |                       |
                                                     v                       v
                                                summarize                   END
                                                     |
                                                     v
                                                    END

    记忆层次：
        - 短期记忆 (memories): LangGraph State + Checkpointer
        - 中期记忆 (summaries): LLM 压缩的摘要
        - 长期记忆 (Store): 跨会话持久化

    反思模式：
        - reflect 节点：分析近期表现，生成洞察
        - 冷静期处理：当反思建议暂停交易时，跳过 LLM 决策

    Args:
        decide_callback: 决策回调（async）
        execute_callback: 执行回调（async）
        summarize_callback: 摘要回调（async），接收 memories_text 返回摘要
        reflect_callback: 反思回调（async），接收 (digest, history_records) 返回 ReflectionInsight.dict()
        checkpointer: 状态检查点保存器（短期/中期记忆持久化）
        store: 长期记忆存储（跨会话持久化）
        summary_threshold: 触发摘要的 memories 数量阈值（默认 10）
        reflection_interval: 反思间隔（每隔多少周期触发一次反思）
        enable_reflection: 是否启用反思模式

    Returns:
        编译后的 StateGraph
    """
    nodes = TradingGraphNodes(
        decide_callback=decide_callback,
        execute_callback=execute_callback,
        summarize_callback=summarize_callback,
        reflect_callback=reflect_callback,
        summary_threshold=summary_threshold,
        reflection_interval=reflection_interval,
    )

    # 创建图
    workflow = StateGraph(TradingState)

    # 添加节点
    workflow.add_node("reflect", nodes.reflect)
    workflow.add_node("cooldown_noop", nodes.cooldown_noop)
    workflow.add_node("decide", nodes.decide)
    workflow.add_node("execute", nodes.execute)
    workflow.add_node("record", nodes.record)
    workflow.add_node("summarize", nodes.summarize)

    # 添加边
    workflow.add_edge(START, "reflect")

    # 条件边：反思后判断是否处于冷静期
    workflow.add_conditional_edges(
        "reflect",
        _should_cooldown,
        {
            "cooldown": "cooldown_noop",
            "decide": "decide",
        }
    )

    # 冷静期路径：直接记录并结束
    workflow.add_edge("cooldown_noop", "record")

    # 正常路径
    workflow.add_edge("decide", "execute")
    workflow.add_edge("execute", "record")

    # 条件边：根据 memories 数量决定是否摘要
    workflow.add_conditional_edges(
        "record",
        _should_summarize,
        {
            "summarize": "summarize",
            "end": END,
        }
    )
    workflow.add_edge("summarize", END)

    # 编译（带 checkpointer 和 store）
    return workflow.compile(checkpointer=checkpointer, store=store)


def get_sqlite_checkpointer_context(
    db_path: str = "data/trading_graph.db",
):
    """获取 SQLite Checkpointer 上下文管理器。

    需要配合 async with 使用：
        async with get_sqlite_checkpointer_context(db_path) as checkpointer:
            graph = create_trading_graph(..., checkpointer=checkpointer)

    Args:
        db_path: 数据库文件路径

    Returns:
        AsyncContextManager[AsyncSqliteSaver]
    """
    # 确保目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    cprint(f"LangGraph Checkpointer 路径: {db_path}", "white")
    return AsyncSqliteSaver.from_conn_string(db_path)


def get_thread_config(strategy_id: str) -> Dict[str, Any]:
    """获取 LangGraph 线程配置。

    thread_id 使用 strategy_id，这样每个策略有独立的状态。
    """
    return {
        "configurable": {
            "thread_id": strategy_id,
        }
    }
