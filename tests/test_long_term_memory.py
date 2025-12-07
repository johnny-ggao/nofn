"""长期记忆集成测试。

测试 LangGraph Store 与交易决策图的集成：
1. 摘要归档到长期记忆
2. 决策时检索长期记忆
3. Store 持久化
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langgraph.store.memory import InMemoryStore

from src.trading.graph.long_term_memory import (
    MemoryType,
    LongTermMemoryConfig,
    LongTermMemoryManager,
    get_memory_namespace,
    create_in_memory_store,
)
from src.trading.graph.state import (
    TradingState,
    DecisionMemory,
    DecisionSummary,
    create_initial_state,
)
from src.trading.graph.workflow import (
    TradingGraphNodes,
    create_trading_graph,
    get_thread_config,
)


def create_test_state(
    strategy_id: str = "test-strategy",
    memories: list = None,
    summaries: list = None,
) -> TradingState:
    """创建测试用 State。"""
    state = create_initial_state(
        strategy_id=strategy_id,
        compose_id="test-compose",
        timestamp_ms=1700000000000,
        features=[],
        portfolio={},
        digest={},
    )
    if memories:
        state["memories"] = memories
    if summaries:
        state["summaries"] = summaries
    return state


def create_test_memory(cycle_index: int, action: str = "buy") -> DecisionMemory:
    """创建测试用记忆。"""
    return DecisionMemory(
        cycle_index=cycle_index,
        timestamp_ms=1700000000000 + cycle_index * 60000,
        action=action,
        symbol="BTCUSDT",
        quantity=0.1,
        rationale=f"Test rationale for cycle {cycle_index}",
        confidence=0.8,
        executed=True,
        exec_price=50000.0,
        realized_pnl=10.0 if action == "sell" else 0.0,
    )


def create_test_summary(
    cycle_start: int, cycle_end: int, pnl: float = 50.0
) -> DecisionSummary:
    """创建测试用摘要。"""
    return DecisionSummary(
        cycle_range=(cycle_start, cycle_end),
        time_range=(1700000000000, 1700000600000),
        content=f"周期{cycle_start}-{cycle_end}共5次决策，执行4次。累计盈利{pnl:.4f}。",
        total_decisions=5,
        executed_count=4,
        total_pnl=pnl,
    )


class TestMemoryNamespace:
    """测试命名空间生成。"""

    def test_base_namespace(self):
        """测试基础命名空间。"""
        ns = get_memory_namespace("my-strategy")
        assert ns == ("strategies", "my-strategy", "memories")

    def test_typed_namespace(self):
        """测试带类型的命名空间。"""
        ns = get_memory_namespace("my-strategy", MemoryType.PATTERN)
        assert ns == ("strategies", "my-strategy", "memories", "pattern")

        ns = get_memory_namespace("my-strategy", MemoryType.LESSON)
        assert ns == ("strategies", "my-strategy", "memories", "lesson")

        ns = get_memory_namespace("my-strategy", MemoryType.CASE)
        assert ns == ("strategies", "my-strategy", "memories", "case")


class TestLongTermMemoryManager:
    """测试长期记忆管理器。"""

    def test_save_and_get_memory(self):
        """测试保存和获取记忆。"""
        store = create_in_memory_store()
        manager = LongTermMemoryManager(store)

        memory_id = manager.save_memory(
            strategy_id="test-strategy",
            memory_type=MemoryType.PATTERN,
            content="当价格突破阻力位时买入",
            importance=0.8,
        )

        assert memory_id.startswith("mem-")

        # 获取记忆
        memory = manager.get_memory(
            "test-strategy", memory_id, MemoryType.PATTERN
        )
        assert memory is not None
        assert memory["content"] == "当价格突破阻力位时买入"
        assert memory["importance"] == 0.8
        assert memory["access_count"] == 1  # 访问后计数+1

    def test_search_memories(self):
        """测试搜索记忆。"""
        store = create_in_memory_store()
        manager = LongTermMemoryManager(store)

        # 保存多条记忆
        manager.save_memory(
            "test-strategy", MemoryType.PATTERN, "模式1", importance=0.9
        )
        manager.save_memory(
            "test-strategy", MemoryType.PATTERN, "模式2", importance=0.5
        )
        manager.save_memory(
            "test-strategy", MemoryType.PATTERN, "模式3", importance=0.7
        )

        # 搜索所有
        results = manager.search_memories(
            "test-strategy", memory_type=MemoryType.PATTERN, limit=10
        )
        assert len(results) == 3

        # 按重要性排序
        assert results[0]["importance"] == 0.9
        assert results[1]["importance"] == 0.7
        assert results[2]["importance"] == 0.5

        # 过滤低重要性
        results = manager.search_memories(
            "test-strategy",
            memory_type=MemoryType.PATTERN,
            min_importance=0.6,
        )
        assert len(results) == 2

    def test_archive_summary(self):
        """测试归档摘要。"""
        store = create_in_memory_store()
        manager = LongTermMemoryManager(store)

        summary = create_test_summary(1, 5, pnl=100.0)

        memory_ids = manager.archive_summary("test-strategy", summary)

        assert len(memory_ids) == 1
        assert memory_ids[0].startswith("mem-")

        # 验证归档内容
        memory = manager.get_memory(
            "test-strategy", memory_ids[0], MemoryType.CASE
        )
        assert memory is not None
        assert memory["type"] == "case"
        assert memory["total_pnl"] == 100.0
        assert memory["source_cycles"] == [1, 2, 3, 4, 5]

    def test_get_relevant_memories(self):
        """测试获取相关记忆。"""
        store = create_in_memory_store()
        manager = LongTermMemoryManager(store)

        # 保存不同类型的记忆
        manager.save_memory(
            "test-strategy", MemoryType.PATTERN, "模式", importance=0.9
        )
        manager.save_memory(
            "test-strategy", MemoryType.LESSON, "教训", importance=0.8
        )
        manager.save_memory(
            "test-strategy", MemoryType.CASE, "案例", importance=0.7
        )

        # 获取相关记忆
        memories = manager.get_relevant_memories("test-strategy", limit=5)

        assert len(memories) == 3
        # 按重要性排序
        assert memories[0]["content"] == "模式"
        assert memories[1]["content"] == "教训"
        assert memories[2]["content"] == "案例"

    def test_format_memories_for_prompt(self):
        """测试格式化记忆为提示词。"""
        store = create_in_memory_store()
        manager = LongTermMemoryManager(store)

        memories = [
            {
                "type": "pattern",
                "content": "突破阻力位时买入",
                "importance": 0.8,
                "total_pnl": 50.0,
            },
            {
                "type": "lesson",
                "content": "不要追涨",
                "importance": 0.7,
                "total_pnl": -30.0,
            },
        ]

        formatted = manager.format_memories_for_prompt(memories)

        assert "pattern" in formatted
        assert "突破阻力位时买入" in formatted
        assert "+50.0000" in formatted
        assert "lesson" in formatted
        assert "不要追涨" in formatted
        assert "-30.0000" in formatted


class TestGraphLongTermMemoryIntegration:
    """测试图与长期记忆的集成。"""

    @pytest.mark.asyncio
    async def test_decide_node_retrieves_long_term_memories(self):
        """测试决策节点检索长期记忆。"""
        store = create_in_memory_store()

        # 预存一些长期记忆
        manager = LongTermMemoryManager(store)
        manager.save_memory(
            "test-strategy",
            MemoryType.PATTERN,
            "重要的交易模式",
            importance=0.9,
        )

        # 创建节点
        decide_callback = AsyncMock(return_value={"instructions": [], "rationale": "Test"})
        nodes = TradingGraphNodes(
            decide_callback=decide_callback,
            execute_callback=AsyncMock(return_value=[]),
        )

        # 调用决策节点
        state = create_test_state(strategy_id="test-strategy")
        result = await nodes.decide(state, store=store)

        # 验证回调收到了长期记忆
        decide_callback.assert_called_once()
        context = decide_callback.call_args[0][0]
        assert "long_term_memories" in context
        assert len(context["long_term_memories"]) == 1
        assert context["long_term_memories"][0]["content"] == "重要的交易模式"

    @pytest.mark.asyncio
    async def test_decide_node_works_without_store(self):
        """测试决策节点在没有 Store 时正常工作。"""
        decide_callback = AsyncMock(return_value={"instructions": [], "rationale": "Test"})
        nodes = TradingGraphNodes(
            decide_callback=decide_callback,
            execute_callback=AsyncMock(return_value=[]),
        )

        state = create_test_state()
        result = await nodes.decide(state, store=None)

        # 验证正常返回
        assert "instructions" in result
        assert "rationale" in result

        # 验证上下文中没有长期记忆
        context = decide_callback.call_args[0][0]
        assert "long_term_memories" not in context

    @pytest.mark.asyncio
    async def test_summarize_node_archives_old_summaries(self):
        """测试摘要节点归档旧摘要。"""
        store = create_in_memory_store()

        # 创建已满的 summaries
        summaries = [create_test_summary(i * 5 + 1, i * 5 + 5) for i in range(5)]

        # 创建满的 memories 触发摘要
        memories = [create_test_memory(i) for i in range(1, 11)]

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(return_value={}),
            execute_callback=AsyncMock(return_value=[]),
            summarize_callback=None,  # 使用简单摘要
        )

        state = create_test_state(
            strategy_id="test-strategy",
            memories=memories,
            summaries=summaries,
        )

        # 调用摘要节点
        result = await nodes.summarize(state, store=store)

        # 验证新摘要生成
        assert "summaries" in result
        assert len(result["summaries"]) == 1

        # 验证旧摘要被归档到 Store
        manager = LongTermMemoryManager(store)
        archived = manager.search_memories(
            "test-strategy", memory_type=MemoryType.CASE
        )
        assert len(archived) == 1
        assert archived[0]["source_cycles"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_summarize_node_works_without_store(self):
        """测试摘要节点在没有 Store 时正常工作。"""
        memories = [create_test_memory(i) for i in range(1, 11)]
        summaries = [create_test_summary(i * 5 + 1, i * 5 + 5) for i in range(5)]

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(return_value={}),
            execute_callback=AsyncMock(return_value=[]),
        )

        state = create_test_state(memories=memories, summaries=summaries)
        result = await nodes.summarize(state, store=None)

        # 仍然生成摘要
        assert "summaries" in result
        assert len(result["summaries"]) == 1


class TestGraphCompileWithStore:
    """测试图编译时带 Store。"""

    @pytest.mark.asyncio
    async def test_create_graph_with_store(self):
        """测试创建带 Store 的图。"""
        store = create_in_memory_store()

        graph = create_trading_graph(
            decide_callback=AsyncMock(return_value={"instructions": [], "rationale": "Test"}),
            execute_callback=AsyncMock(return_value=[]),
            store=store,
        )

        assert graph is not None
        # 验证图可以被调用
        state = create_test_state()
        config = get_thread_config("test-strategy")

        # 不实际运行，只验证编译成功
        assert graph.nodes is not None


class TestStoreAsyncMethods:
    """测试 Store 的异步方法。"""

    @pytest.mark.asyncio
    async def test_async_put_and_search(self):
        """测试异步存储和搜索。"""
        store = create_in_memory_store()
        namespace = ("test", "memories")

        # 异步存储
        await store.aput(namespace, "key1", {"content": "value1"})
        await store.aput(namespace, "key2", {"content": "value2"})

        # 异步搜索
        results = await store.asearch(namespace, limit=10)
        assert len(list(results)) == 2

    @pytest.mark.asyncio
    async def test_async_get(self):
        """测试异步获取。"""
        store = create_in_memory_store()
        namespace = ("test", "memories")

        await store.aput(namespace, "key1", {"content": "test"})

        item = await store.aget(namespace, "key1")
        assert item is not None
        assert item.value["content"] == "test"
