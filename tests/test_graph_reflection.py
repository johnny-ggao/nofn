"""反思节点集成测试。

测试 LangGraph 工作流中反思节点的功能：
1. 反思节点执行和洞察生成
2. 冷静期处理逻辑
3. 反思上下文注入到决策节点
4. 完整工作流测试
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock

from src.trading.graph.state import (
    TradingState,
    create_initial_state,
    state_to_compose_context,
)
from src.trading.graph.workflow import (
    TradingGraphNodes,
    create_trading_graph,
    _should_cooldown,
)


# ============================================================================
# 测试工具
# ============================================================================


def create_test_state(
    cycle_index: int = 0,
    cooldown_remaining: int = 0,
    reflection_enabled: bool = True,
    reflection_insight: Dict[str, Any] = None,
) -> TradingState:
    """创建测试用的状态。"""
    state = create_initial_state(
        strategy_id="test-strategy",
        compose_id="test-compose-001",
        timestamp_ms=1700000000000,
        features=[{"symbol": "BTCUSDT", "price": 50000}],
        portfolio={"total_value": 10000, "positions": []},
        digest={"sharpe_ratio": 0.5, "by_instrument": {}},
        history_records=[],
        reflection_enabled=reflection_enabled,
    )
    # 覆盖需要测试的字段
    state["cycle_index"] = cycle_index
    state["cooldown_remaining"] = cooldown_remaining
    if reflection_insight is not None:
        state["reflection_insight"] = reflection_insight
    return state


def create_mock_insight(
    cooldown_cycles: int = 0,
    alerts: List[Dict] = None,
    summary: str = "测试反思摘要",
) -> Dict[str, Any]:
    """创建模拟的反思洞察。"""
    return {
        "ts": 1700000000000,
        "sharpe_ratio": 0.3,
        "sharpe_trend": "declining",
        "total_trades": 10,
        "win_rate": 0.4,
        "alerts": alerts or [],
        "lessons": [],
        "cooldown_cycles": cooldown_cycles,
        "symbols_to_avoid": [],
        "symbols_performing_well": [],
        "summary": summary,
    }


# ============================================================================
# 反思节点测试
# ============================================================================


class TestReflectNode:
    """测试反思节点。"""

    @pytest.mark.asyncio
    async def test_reflect_disabled(self):
        """测试禁用反思模式时跳过反思。"""
        state = create_test_state(reflection_enabled=False)

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
            reflect_callback=AsyncMock(),
        )

        result = await nodes.reflect(state)

        assert result == {}
        nodes._reflect_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflect_in_cooldown(self):
        """测试冷静期内递减 cooldown。"""
        state = create_test_state(cooldown_remaining=3)

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
            reflect_callback=AsyncMock(),
        )

        result = await nodes.reflect(state)

        assert result["cooldown_remaining"] == 2
        nodes._reflect_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflect_interval_not_reached(self):
        """测试未达到反思间隔时跳过。"""
        state = create_test_state(cycle_index=3, cooldown_remaining=0)
        state["last_reflection_cycle"] = 1  # 距上次仅 2 个周期

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
            reflect_callback=AsyncMock(),
            reflection_interval=5,  # 需要 5 个周期间隔
        )

        result = await nodes.reflect(state)

        assert result == {}
        nodes._reflect_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflect_executes_callback(self):
        """测试正常执行反思回调。"""
        state = create_test_state(cycle_index=10, cooldown_remaining=0)
        state["last_reflection_cycle"] = 0

        mock_insight = create_mock_insight(cooldown_cycles=2)
        reflect_callback = AsyncMock(return_value=mock_insight)

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
            reflect_callback=reflect_callback,
            reflection_interval=5,
        )

        result = await nodes.reflect(state)

        # 验证回调被调用
        reflect_callback.assert_called_once()

        # 验证结果
        assert result["reflection_insight"] == mock_insight
        assert result["cooldown_remaining"] == 2
        assert result["last_reflection_cycle"] == 10

    @pytest.mark.asyncio
    async def test_reflect_handles_callback_error(self):
        """测试反思回调失败时的处理。"""
        state = create_test_state(cycle_index=10, cooldown_remaining=0)
        state["last_reflection_cycle"] = 0

        reflect_callback = AsyncMock(side_effect=Exception("反思失败"))

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
            reflect_callback=reflect_callback,
            reflection_interval=5,
        )

        result = await nodes.reflect(state)

        # 失败时返回空
        assert result == {}


# ============================================================================
# 冷静期 NOOP 节点测试
# ============================================================================


class TestCooldownNoopNode:
    """测试冷静期 NOOP 节点。"""

    @pytest.mark.asyncio
    async def test_cooldown_noop_generates_rationale(self):
        """测试冷静期节点生成说明。"""
        insight = create_mock_insight(cooldown_cycles=3, summary="夏普比过低，建议暂停")
        state = create_test_state(
            cooldown_remaining=2,
            reflection_insight=insight,
        )

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
        )

        result = await nodes.cooldown_noop(state)

        assert result["instructions"] == []
        assert result["trades"] == []
        assert "[冷静期]" in result["rationale"]
        assert "2" in result["rationale"]  # 剩余周期数

    @pytest.mark.asyncio
    async def test_cooldown_noop_without_insight(self):
        """测试无反思洞察时的冷静期处理。"""
        state = create_test_state(cooldown_remaining=1)

        nodes = TradingGraphNodes(
            decide_callback=AsyncMock(),
            execute_callback=AsyncMock(),
        )

        result = await nodes.cooldown_noop(state)

        assert result["instructions"] == []
        assert "[冷静期]" in result["rationale"]


# ============================================================================
# 条件边测试
# ============================================================================


class TestConditionalEdges:
    """测试条件边函数。"""

    def test_should_cooldown_in_cooldown(self):
        """测试冷静期中的路由。"""
        state = create_test_state(cooldown_remaining=2)
        assert _should_cooldown(state) == "cooldown"

    def test_should_cooldown_not_in_cooldown(self):
        """测试非冷静期的路由。"""
        state = create_test_state(cooldown_remaining=0)
        assert _should_cooldown(state) == "decide"


# ============================================================================
# 反思上下文注入测试
# ============================================================================


class TestReflectionContextInjection:
    """测试反思上下文注入到决策节点。"""

    def test_context_includes_reflection_when_enabled(self):
        """测试启用反思时上下文包含反思信息。"""
        insight = create_mock_insight(cooldown_cycles=0)
        state = create_test_state(
            reflection_enabled=True,
            reflection_insight=insight,
            cooldown_remaining=1,
        )

        context = state_to_compose_context(state)

        assert "reflection" in context
        assert context["reflection"]["insight"] == insight
        assert context["reflection"]["cooldown_remaining"] == 1
        assert context["reflection"]["in_cooldown"] is True

    def test_context_excludes_reflection_when_disabled(self):
        """测试禁用反思时上下文不包含反思信息。"""
        state = create_test_state(reflection_enabled=False)

        context = state_to_compose_context(state)

        assert "reflection" not in context


# ============================================================================
# 完整工作流测试
# ============================================================================


class TestCompleteWorkflow:
    """测试完整的反思工作流。"""

    @pytest.mark.asyncio
    async def test_workflow_with_reflection_enabled(self):
        """测试启用反思的完整工作流。"""
        # 创建回调
        decide_callback = AsyncMock(return_value={
            "instructions": [{"action": "buy", "symbol": "BTCUSDT"}],
            "rationale": "测试决策",
        })
        execute_callback = AsyncMock(return_value=[])
        reflect_callback = AsyncMock(return_value=create_mock_insight(cooldown_cycles=0))

        # 创建图
        graph = create_trading_graph(
            decide_callback=decide_callback,
            execute_callback=execute_callback,
            reflect_callback=reflect_callback,
            enable_reflection=True,
            reflection_interval=5,  # 每 5 个周期反思一次
        )

        # 创建初始状态 - cycle_index=5 满足反思间隔
        state = create_initial_state(
            strategy_id="test-strategy",
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            history_records=[],
            reflection_enabled=True,
        )
        state["cycle_index"] = 5  # 距上次反思 5 个周期
        state["last_reflection_cycle"] = 0

        # 运行图
        result = await graph.ainvoke(state)

        # 验证反思被执行
        reflect_callback.assert_called_once()
        # 验证决策被执行（因为 cooldown=0）
        decide_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_cooldown_skips_decide(self):
        """测试冷静期时跳过决策。"""
        decide_callback = AsyncMock()
        execute_callback = AsyncMock()
        # 反思回调返回 cooldown
        reflect_callback = AsyncMock(return_value=create_mock_insight(cooldown_cycles=3))

        graph = create_trading_graph(
            decide_callback=decide_callback,
            execute_callback=execute_callback,
            reflect_callback=reflect_callback,
            enable_reflection=True,
            reflection_interval=5,
        )

        # 创建满足反思间隔的状态
        state = create_initial_state(
            strategy_id="test-strategy",
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            history_records=[],
            reflection_enabled=True,
        )
        state["cycle_index"] = 5
        state["last_reflection_cycle"] = 0

        result = await graph.ainvoke(state)

        # 反思被执行
        reflect_callback.assert_called_once()
        # 注意：reflect 节点设置 cooldown_remaining=3，但条件边是在 reflect 之后判断
        # 由于 reflect 节点返回的更新会合并到 state，所以此时 cooldown_remaining=3
        # 因此会进入 cooldown_noop 分支
        decide_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_continues_cooldown(self):
        """测试持续冷静期的行为。"""
        decide_callback = AsyncMock()
        execute_callback = AsyncMock()
        reflect_callback = AsyncMock()

        graph = create_trading_graph(
            decide_callback=decide_callback,
            execute_callback=execute_callback,
            reflect_callback=reflect_callback,
            enable_reflection=True,
        )

        # 已经处于冷静期的状态
        state = create_initial_state(
            strategy_id="test-strategy",
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            history_records=[],
            reflection_enabled=True,
        )
        state["cooldown_remaining"] = 2

        result = await graph.ainvoke(state)

        # 反思不会调用回调（只是递减 cooldown）
        reflect_callback.assert_not_called()
        # 决策不会被执行（跳过）
        decide_callback.assert_not_called()
        # cooldown 递减
        assert result["cooldown_remaining"] == 1


# ============================================================================
# 反思节点与记忆集成测试
# ============================================================================


class TestReflectionWithMemories:
    """测试反思与记忆系统的集成。"""

    @pytest.mark.asyncio
    async def test_cooldown_creates_noop_memory(self):
        """测试冷静期创建 NOOP 记忆。

        注意：cooldown_remaining=1 时，reflect 节点会递减为 0，
        然后走 decide 路径。要测试冷静期 NOOP，需要设置 cooldown_remaining >= 2。
        """
        # 正确配置 mock 返回值
        decide_callback = AsyncMock(return_value={
            "instructions": [],
            "rationale": "测试决策",
        })
        execute_callback = AsyncMock(return_value=[])

        graph = create_trading_graph(
            decide_callback=decide_callback,
            execute_callback=execute_callback,
            enable_reflection=True,
        )

        # cooldown_remaining=2：reflect 递减为 1，然后走 cooldown_noop 路径
        state = create_initial_state(
            strategy_id="test-strategy",
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            history_records=[],
            reflection_enabled=True,
        )
        state["cooldown_remaining"] = 2

        result = await graph.ainvoke(state)

        # reflect 递减后 cooldown=1，走 cooldown_noop 路径
        # cooldown_noop 创建 NOOP 决策
        assert len(result["memories"]) == 1
        assert result["memories"][0]["action"] == "noop"
        assert "[冷静期]" in result["memories"][0]["rationale"]
        # cycle_index 增加
        assert result["cycle_index"] == 1
        # cooldown 递减为 1
        assert result["cooldown_remaining"] == 1
        # decide 不会被调用
        decide_callback.assert_not_called()
