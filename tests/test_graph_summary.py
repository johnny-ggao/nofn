"""Tests for LangGraph trading graph summary functionality."""

import pytest
import asyncio
from typing import Any, Dict, List

from src.trading.graph.state import (
    TradingState,
    DecisionMemory,
    DecisionSummary,
    create_initial_state,
    create_memory_from_result,
    state_to_compose_context,
    should_generate_summary,
    prepare_memories_for_summary,
    create_summary_from_memories,
    format_memories_for_summarization,
    _merge_memories,
    _merge_summaries,
)
from src.trading.graph.summarizer import (
    generate_summary_with_llm,
    _generate_simple_summary,
    maybe_generate_summary,
)


class TestDecisionMemory:
    """Test DecisionMemory creation and manipulation."""

    def test_create_memory_from_noop(self):
        """Test creating memory from NOOP decision."""
        memories = create_memory_from_result(
            cycle_index=1,
            timestamp_ms=1700000000000,
            instructions=[],
            trades=[],
            rationale="No signal detected",
        )

        assert len(memories) == 1
        assert memories[0]["action"] == "noop"
        assert memories[0]["executed"] is False
        assert memories[0]["rationale"] == "No signal detected"

    def test_create_memory_from_trade(self):
        """Test creating memory from executed trade."""
        instructions = [
            {
                "instruction_id": "inst-1",
                "action": "open_long",
                "instrument": {"symbol": "BTC/USDT"},
                "quantity": 0.01,
                "meta": {"rationale": "Breakout detected", "confidence": 0.8},
            }
        ]
        trades = [
            {
                "instruction_id": "inst-1",
                "avg_exec_price": 50000.0,
                "realized_pnl": None,
            }
        ]

        memories = create_memory_from_result(
            cycle_index=2,
            timestamp_ms=1700000060000,
            instructions=instructions,
            trades=trades,
            rationale="Overall bullish",
        )

        assert len(memories) == 1
        assert memories[0]["action"] == "open_long"
        assert memories[0]["symbol"] == "BTC/USDT"
        assert memories[0]["executed"] is True
        assert memories[0]["exec_price"] == 50000.0


class TestMemoryMerger:
    """Test memory reducer functions."""

    def test_merge_memories_basic(self):
        """Test basic memory merging."""
        existing = [
            DecisionMemory(
                cycle_index=1, timestamp_ms=1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
        ]
        new = [
            DecisionMemory(
                cycle_index=2, timestamp_ms=2000, action="open_long",
                symbol="BTC/USDT", quantity=0.01, rationale="Test", confidence=0.8,
                executed=True, exec_price=50000.0, realized_pnl=None,
            )
        ]

        merged = _merge_memories(existing, new)
        assert len(merged) == 2
        assert merged[0]["cycle_index"] == 1
        assert merged[1]["cycle_index"] == 2

    def test_merge_memories_truncation(self):
        """Test that memories are truncated to 10."""
        existing = [
            DecisionMemory(
                cycle_index=i, timestamp_ms=i*1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
            for i in range(8)
        ]
        new = [
            DecisionMemory(
                cycle_index=i+8, timestamp_ms=(i+8)*1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
            for i in range(5)
        ]

        merged = _merge_memories(existing, new)
        assert len(merged) == 10
        # Should keep the most recent 10 (cycles 3-12)
        assert merged[0]["cycle_index"] == 3
        assert merged[-1]["cycle_index"] == 12


class TestSummaryGeneration:
    """Test summary generation."""

    def test_should_generate_summary(self):
        """Test summary threshold check."""
        small_memories = [
            DecisionMemory(
                cycle_index=i, timestamp_ms=i*1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
            for i in range(5)
        ]
        assert should_generate_summary(small_memories) is False

        large_memories = [
            DecisionMemory(
                cycle_index=i, timestamp_ms=i*1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
            for i in range(10)
        ]
        assert should_generate_summary(large_memories) is True

    def test_prepare_memories_for_summary(self):
        """Test memory splitting for summarization."""
        memories = [
            DecisionMemory(
                cycle_index=i, timestamp_ms=i*1000, action="noop",
                symbol=None, quantity=None, rationale=None, confidence=None,
                executed=False, exec_price=None, realized_pnl=None,
            )
            for i in range(10)
        ]

        to_summarize, to_keep = prepare_memories_for_summary(memories)

        assert len(to_summarize) == 5
        assert len(to_keep) == 5
        assert to_summarize[0]["cycle_index"] == 0
        assert to_summarize[-1]["cycle_index"] == 4
        assert to_keep[0]["cycle_index"] == 5
        assert to_keep[-1]["cycle_index"] == 9

    def test_create_summary_from_memories(self):
        """Test summary creation."""
        memories = [
            DecisionMemory(
                cycle_index=i,
                timestamp_ms=1700000000000 + i*60000,
                action="open_long" if i % 2 == 0 else "noop",
                symbol="BTC/USDT" if i % 2 == 0 else None,
                quantity=0.01 if i % 2 == 0 else None,
                rationale=None,
                confidence=None,
                executed=i % 2 == 0,
                exec_price=50000.0 if i % 2 == 0 else None,
                realized_pnl=10.0 if i == 2 else None,
            )
            for i in range(5)
        ]

        summary = create_summary_from_memories(memories, "Test summary content")

        assert summary["cycle_range"] == (0, 4)
        assert summary["content"] == "Test summary content"
        assert summary["total_decisions"] == 5
        assert summary["executed_count"] == 3  # cycles 0, 2, 4 executed
        assert summary["total_pnl"] == 10.0

    def test_format_memories_for_summarization(self):
        """Test memory formatting for LLM."""
        memories = [
            DecisionMemory(
                cycle_index=1,
                timestamp_ms=1700000000000,
                action="open_long",
                symbol="BTC/USDT",
                quantity=0.01,
                rationale="Bullish trend",
                confidence=0.8,
                executed=True,
                exec_price=50000.0,
                realized_pnl=None,
            )
        ]

        text = format_memories_for_summarization(memories)

        assert "周期1" in text
        assert "open_long" in text
        assert "BTC/USDT" in text
        assert "已执行" in text
        assert "50000.00" in text
        assert "Bullish trend" in text

    def test_simple_summary_generation(self):
        """Test simple summary without LLM."""
        memories = [
            DecisionMemory(
                cycle_index=i,
                timestamp_ms=1700000000000 + i*60000,
                action="open_long" if i < 3 else "close_long",
                symbol="BTC/USDT",
                quantity=0.01,
                rationale=None,
                confidence=None,
                executed=True,
                exec_price=50000.0,
                realized_pnl=100.0 if i >= 3 else None,
            )
            for i in range(5)
        ]

        summary = _generate_simple_summary(memories)

        assert "周期0-4" in summary
        assert "5次决策" in summary
        assert "执行5次" in summary
        assert "open_long=3次" in summary or "close_long=2次" in summary


class TestStateToComposeContext:
    """Test state to compose context conversion."""

    def test_context_with_summaries(self):
        """Test that summaries are included in compose context."""
        state = TradingState(
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            instructions=[],
            rationale=None,
            trades=[],
            memories=[
                DecisionMemory(
                    cycle_index=10,
                    timestamp_ms=1700000600000,
                    action="noop",
                    symbol=None,
                    quantity=None,
                    rationale="Waiting",
                    confidence=None,
                    executed=False,
                    exec_price=None,
                    realized_pnl=None,
                )
            ],
            summaries=[
                DecisionSummary(
                    cycle_range=(0, 4),
                    time_range=(1700000000000, 1700000240000),
                    content="前5个周期主要观望，执行了2次开多，累计盈利50USDT。",
                    total_decisions=5,
                    executed_count=2,
                    total_pnl=50.0,
                )
            ],
            pending_signals={},
            cycle_index=10,
            strategy_id="test-strategy",
        )

        context = state_to_compose_context(state)

        # Check summaries are included
        assert "history_summaries" in context
        assert len(context["history_summaries"]) == 1

        summary = context["history_summaries"][0]
        assert summary["cycle_range"] == (0, 4)
        assert "前5个周期" in summary["content"]
        assert summary["stats"]["decisions"] == 5
        assert summary["stats"]["pnl"] == 50.0

    def test_context_without_summaries(self):
        """Test compose context when no summaries exist."""
        state = TradingState(
            compose_id="test-compose",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            instructions=[],
            rationale=None,
            trades=[],
            memories=[],
            summaries=[],
            pending_signals={},
            cycle_index=0,
            strategy_id="test-strategy",
        )

        context = state_to_compose_context(state)

        assert context["history_summaries"] == []


class TestSummaryMerger:
    """Test summary reducer function."""

    def test_merge_summaries_basic(self):
        """Test basic summary merging."""
        existing = [
            DecisionSummary(
                cycle_range=(0, 4),
                time_range=(1000, 5000),
                content="First summary",
                total_decisions=5,
                executed_count=2,
                total_pnl=10.0,
            )
        ]
        new = [
            DecisionSummary(
                cycle_range=(5, 9),
                time_range=(6000, 10000),
                content="Second summary",
                total_decisions=5,
                executed_count=3,
                total_pnl=20.0,
            )
        ]

        merged = _merge_summaries(existing, new)

        assert len(merged) == 2
        assert merged[0]["content"] == "First summary"
        assert merged[1]["content"] == "Second summary"

    def test_merge_summaries_truncation(self):
        """Test that summaries are truncated to 5."""
        existing = [
            DecisionSummary(
                cycle_range=(i*5, i*5+4),
                time_range=(i*5000, i*5000+4000),
                content=f"Summary {i}",
                total_decisions=5,
                executed_count=2,
                total_pnl=10.0,
            )
            for i in range(4)
        ]
        new = [
            DecisionSummary(
                cycle_range=(i*5+20, i*5+24),
                time_range=(i*5000+20000, i*5000+24000),
                content=f"Summary {i+4}",
                total_decisions=5,
                executed_count=3,
                total_pnl=20.0,
            )
            for i in range(3)
        ]

        merged = _merge_summaries(existing, new)

        assert len(merged) == 5
        # Should keep most recent 5
        assert merged[0]["content"] == "Summary 2"
        assert merged[-1]["content"] == "Summary 6"


@pytest.mark.asyncio
async def test_maybe_generate_summary():
    """Test conditional summary generation."""
    # Less than threshold - no summary
    small_memories = [
        DecisionMemory(
            cycle_index=i, timestamp_ms=i*1000, action="noop",
            symbol=None, quantity=None, rationale=None, confidence=None,
            executed=False, exec_price=None, realized_pnl=None,
        )
        for i in range(5)
    ]
    result = await maybe_generate_summary(small_memories, threshold=10)
    assert result is None

    # At threshold - generate summary
    large_memories = [
        DecisionMemory(
            cycle_index=i, timestamp_ms=i*1000, action="noop",
            symbol=None, quantity=None, rationale=None, confidence=None,
            executed=False, exec_price=None, realized_pnl=None,
        )
        for i in range(10)
    ]
    result = await maybe_generate_summary(large_memories, threshold=10)
    assert result is not None
    assert result["cycle_range"] == (0, 4)
    assert result["total_decisions"] == 5


@pytest.mark.asyncio
async def test_generate_summary_with_llm_fallback():
    """Test summary generation falls back to simple summary when no LLM provided."""
    memories = [
        DecisionMemory(
            cycle_index=i, timestamp_ms=1700000000000 + i*60000, action="noop",
            symbol=None, quantity=None, rationale=None, confidence=None,
            executed=False, exec_price=None, realized_pnl=None,
        )
        for i in range(5)
    ]

    # Without LLM callback - should use simple summary
    summary = await generate_summary_with_llm(memories, llm_call=None)

    assert "周期0-4" in summary
    assert "5次决策" in summary


class TestConditionalSummarization:
    """Test conditional edge and summarization node workflow."""

    def test_should_summarize_routing(self):
        """Test the _should_summarize conditional edge function."""
        from src.trading.graph.workflow import _should_summarize

        # Less than 10 memories - should go to END
        small_state = TradingState(
            compose_id="test",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            instructions=[],
            rationale=None,
            trades=[],
            memories=[
                DecisionMemory(
                    cycle_index=i, timestamp_ms=i*1000, action="noop",
                    symbol=None, quantity=None, rationale=None, confidence=None,
                    executed=False, exec_price=None, realized_pnl=None,
                )
                for i in range(5)
            ],
            summaries=[],
            pending_signals={},
            cycle_index=5,
            strategy_id="test",
        )
        assert _should_summarize(small_state) == "end"

        # 10 or more memories - should go to summarize
        large_state = TradingState(
            compose_id="test",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            instructions=[],
            rationale=None,
            trades=[],
            memories=[
                DecisionMemory(
                    cycle_index=i, timestamp_ms=i*1000, action="noop",
                    symbol=None, quantity=None, rationale=None, confidence=None,
                    executed=False, exec_price=None, realized_pnl=None,
                )
                for i in range(10)
            ],
            summaries=[],
            pending_signals={},
            cycle_index=10,
            strategy_id="test",
        )
        assert _should_summarize(large_state) == "summarize"

    @pytest.mark.asyncio
    async def test_summarize_node(self):
        """Test the summarize node generates summary correctly."""
        from src.trading.graph.workflow import TradingGraphNodes

        # Create nodes without LLM callback (uses simple summary)
        nodes = TradingGraphNodes(
            decide_callback=lambda ctx: {"instructions": [], "rationale": "test"},
            execute_callback=lambda inst, feat: [],
            summarize_callback=None,
        )

        # State with 10 memories (threshold reached)
        state = TradingState(
            compose_id="test",
            timestamp_ms=1700000000000,
            features=[],
            portfolio={},
            digest={},
            instructions=[],
            rationale=None,
            trades=[],
            memories=[
                DecisionMemory(
                    cycle_index=i,
                    timestamp_ms=1700000000000 + i*60000,
                    action="open_long" if i % 3 == 0 else "noop",
                    symbol="BTC/USDT" if i % 3 == 0 else None,
                    quantity=0.01 if i % 3 == 0 else None,
                    rationale=None,
                    confidence=None,
                    executed=i % 3 == 0,
                    exec_price=50000.0 if i % 3 == 0 else None,
                    realized_pnl=10.0 if i == 6 else None,
                )
                for i in range(10)
            ],
            summaries=[],
            pending_signals={},
            cycle_index=10,
            strategy_id="test",
        )

        # Call summarize node
        result = await nodes.summarize(state)

        # Should generate a summary
        assert "summaries" in result
        assert len(result["summaries"]) == 1

        summary = result["summaries"][0]
        assert summary["cycle_range"] == (0, 4)
        assert summary["total_decisions"] == 5
        assert "周期0-4" in summary["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
