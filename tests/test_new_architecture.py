"""Tests for the new ValueCell-style architecture."""

import pytest
import asyncio

from src.trading import (
    ExchangeConfig,
    LLMModelConfig,
    MarginMode,
    MarketType,
    TradingConfig,
    TradingMode,
    UserRequest,
    InstrumentRef,
    FeatureVector,
    PositionSnapshot,
    PortfolioView,
    TradeDecisionAction,
    TradeDecisionItem,
    TradePlanProposal,
    TradeInstruction,
    TradeSide,
    TxResult,
    TxStatus,
    TradeHistoryEntry,
    TradeType,
    Constraints,
    get_current_timestamp_ms,
)
from src.trading.portfolio import InMemoryPortfolioService
from src.trading.history import InMemoryHistoryRecorder, RollingDigestBuilder
from src.trading.execution import PaperExecutionGateway


class TestModels:
    """Test data models."""

    def test_instrument_ref(self):
        inst = InstrumentRef(symbol="BTC/USDT", exchange_id="binance")
        assert inst.symbol == "BTC/USDT"
        assert inst.exchange_id == "binance"

    def test_user_request(self):
        request = UserRequest(
            llm_model_config=LLMModelConfig(
                provider="openrouter",
                model_id="deepseek/deepseek-chat",
            ),
            exchange_config=ExchangeConfig(
                exchange_id="binance",
                trading_mode=TradingMode.VIRTUAL,
                market_type=MarketType.SWAP,
            ),
            trading_config=TradingConfig(
                symbols=["BTC/USDT", "ETH/USDT"],
                initial_capital=10000.0,
                max_leverage=3.0,
            ),
        )
        assert request.trading_config.symbols == ["BTC/USDT", "ETH/USDT"]
        assert request.exchange_config.trading_mode == TradingMode.VIRTUAL

    def test_trade_decision_item(self):
        item = TradeDecisionItem(
            instrument=InstrumentRef(symbol="BTC/USDT"),
            action=TradeDecisionAction.OPEN_LONG,
            target_qty=0.001,
            leverage=3.0,
            confidence=0.85,
            rationale="Strong uptrend",
        )
        assert item.action == TradeDecisionAction.OPEN_LONG
        assert item.target_qty == 0.001

    def test_trade_instruction(self):
        inst = TradeInstruction(
            instruction_id="test-123",
            compose_id="compose-456",
            instrument=InstrumentRef(symbol="BTC/USDT"),
            action=TradeDecisionAction.OPEN_LONG,
            side=TradeSide.BUY,
            quantity=0.001,
            leverage=3.0,
        )
        assert inst.side == TradeSide.BUY
        assert inst.quantity == 0.001


class TestPortfolioService:
    """Test portfolio service."""

    def test_initial_state(self):
        service = InMemoryPortfolioService(
            initial_capital=10000.0,
            trading_mode=TradingMode.VIRTUAL,
        )
        view = service.get_view()
        assert view.account_balance == 10000.0
        assert len(view.positions) == 0

    def test_apply_trade(self):
        service = InMemoryPortfolioService(
            initial_capital=10000.0,
            trading_mode=TradingMode.VIRTUAL,
            market_type=MarketType.SWAP,
        )

        trade = TradeHistoryEntry(
            trade_id="trade-1",
            instrument=InstrumentRef(symbol="BTC/USDT"),
            side=TradeSide.BUY,
            type=TradeType.LONG,
            quantity=0.1,
            avg_exec_price=50000.0,
            entry_ts=get_current_timestamp_ms(),
            leverage=3.0,
        )

        service.apply_trades([trade])
        view = service.get_view()
        assert "BTC/USDT" in view.positions
        assert view.positions["BTC/USDT"].quantity == 0.1


class TestHistoryRecorder:
    """Test history recorder and digest builder."""

    def test_record_and_get(self):
        recorder = InMemoryHistoryRecorder()
        from src.trading import HistoryRecord

        record = HistoryRecord(
            ts=get_current_timestamp_ms(),
            kind="execution",
            reference_id="compose-123",
            payload={"trades": []},
        )
        recorder.record(record)

        records = recorder.get_records()
        assert len(records) == 1
        assert records[0].kind == "execution"

    def test_digest_builder(self):
        builder = RollingDigestBuilder()
        digest = builder.build([])
        assert digest.by_instrument == {}
        assert digest.sharpe_ratio is None


class TestPaperExecutionGateway:
    """Test paper trading execution."""

    @pytest.mark.asyncio
    async def test_execute_trade(self):
        gateway = PaperExecutionGateway(
            initial_balance=10000.0,
            fee_bps=10.0,
        )

        # Create market features with price
        features = [
            FeatureVector(
                ts=get_current_timestamp_ms(),
                instrument=InstrumentRef(symbol="BTC/USDT"),
                values={"price.last": 50000.0},
            )
        ]

        # Create instruction
        inst = TradeInstruction(
            instruction_id="test-1",
            compose_id="compose-1",
            instrument=InstrumentRef(symbol="BTC/USDT"),
            action=TradeDecisionAction.OPEN_LONG,
            side=TradeSide.BUY,
            quantity=0.01,
        )

        results = await gateway.execute([inst], market_features=features)

        assert len(results) == 1
        result = results[0]
        assert result.status == TxStatus.FILLED
        assert result.filled_qty > 0
        assert result.avg_exec_price is not None

    @pytest.mark.asyncio
    async def test_fetch_balance(self):
        gateway = PaperExecutionGateway(initial_balance=10000.0)
        balance = await gateway.fetch_balance()

        assert "free" in balance
        assert balance["free"]["USDT"] == 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
