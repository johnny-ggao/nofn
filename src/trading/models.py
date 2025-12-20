"""Trading data models - adapted from ValueCell with LangGraph integration.

This module provides all the data structures needed for trading:
- Configuration models (LLMModelConfig, ExchangeConfig, TradingConfig, UserRequest)
- Market data models (InstrumentRef, Candle, FeatureVector)
- Trading models (TradeInstruction, TxResult, TradeHistoryEntry)
- Portfolio models (PositionSnapshot, PortfolioView)
- Decision models (TradeDecisionItem, TradePlanProposal, ComposeContext)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_CAP_FACTOR,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_LEVERAGE,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_DECIDE_INTERVAL,
    DEFAULT_FEE_BPS,
)


def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


# =============================================================================
# Enums
# =============================================================================

class TradingMode(str, Enum):
    """Trading mode for a strategy."""
    LIVE = "live"
    VIRTUAL = "virtual"


class TradeType(str, Enum):
    """Semantic trade type for positions."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeSide(str, Enum):
    """Low-level execution side (exchange primitive)."""
    BUY = "BUY"
    SELL = "SELL"


class MarketType(str, Enum):
    """Market type for trading."""
    SPOT = "spot"
    FUTURE = "future"
    SWAP = "swap"  # Perpetual futures


class MarginMode(str, Enum):
    """Margin mode for leverage trading."""
    ISOLATED = "isolated"
    CROSS = "cross"


class TradeDecisionAction(str, Enum):
    """Position-oriented high-level actions.

    Semantics:
    - OPEN_LONG: open/increase long; if currently short, flatten then open long
    - OPEN_SHORT: open/increase short; if currently long, flatten then open short
    - CLOSE_LONG: reduce/close long toward 0
    - CLOSE_SHORT: reduce/close short toward 0
    - NOOP: no operation
    """
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NOOP = "noop"


class TxStatus(str, Enum):
    """Execution status of a submitted instruction."""
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    ERROR = "error"


class PriceMode(str, Enum):
    """Order price mode."""
    MARKET = "market"
    LIMIT = "limit"


class StrategyStatus(str, Enum):
    """High-level runtime status for strategies."""
    RUNNING = "running"
    STOPPED = "stopped"


class StopReason(str, Enum):
    """Canonical stop reasons."""
    NORMAL_EXIT = "normal_exit"
    CANCELLED = "cancelled"
    ERROR = "error"
    ERROR_CLOSING_POSITIONS = "error_closing_positions"


# =============================================================================
# Helper Functions
# =============================================================================

def derive_side_from_action(action: Optional[TradeDecisionAction]) -> Optional[TradeSide]:
    """Derive execution side (BUY/SELL) from a high-level action."""
    if action is None:
        return None
    if action in (TradeDecisionAction.OPEN_LONG, TradeDecisionAction.CLOSE_SHORT):
        return TradeSide.BUY
    if action in (TradeDecisionAction.OPEN_SHORT, TradeDecisionAction.CLOSE_LONG):
        return TradeSide.SELL
    return None


# =============================================================================
# Configuration Models
# =============================================================================

class SummaryLLMConfig(BaseModel):
    """LLM configuration for memory summarization (optional).

    使用轻量级模型进行历史决策摘要，降低成本。
    """

    enabled: bool = Field(
        default=False,
        description="是否启用独立的摘要 LLM",
    )
    provider: str = Field(
        default=DEFAULT_MODEL_PROVIDER,
        description="Model provider for summarization",
    )
    model_id: str = Field(
        default=DEFAULT_AGENT_MODEL,
        description="Model identifier (recommend lightweight model)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (if different from main LLM)",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL",
    )
    temperature: float = Field(
        default=0.3,
        description="Model temperature (lower for summarization)",
    )


class LLMModelConfig(BaseModel):
    """AI model configuration for strategy."""

    provider: str = Field(
        default=DEFAULT_MODEL_PROVIDER,
        description="Model provider (e.g., 'openrouter', 'openai')",
    )
    model_id: str = Field(
        default=DEFAULT_AGENT_MODEL,
        description="Model identifier",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL",
    )
    temperature: float = Field(
        default=0.4,
        description="Model temperature for generation",
    )

    # 摘要 LLM 配置（可选）
    summary_llm: Optional[SummaryLLMConfig] = Field(
        default=None,
        description="Optional separate LLM for summarization",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        values = dict(data)
        values.setdefault("provider", DEFAULT_MODEL_PROVIDER)
        values.setdefault("model_id", DEFAULT_AGENT_MODEL)
        return values

    def get_summary_config(self) -> "SummaryLLMConfig":
        """获取摘要 LLM 配置。

        如果未配置独立的摘要 LLM，返回基于主 LLM 的配置。
        """
        if self.summary_llm and self.summary_llm.enabled:
            # 使用独立的摘要 LLM
            config = self.summary_llm
            # 如果没有配置独立的 api_key，继承主 LLM 的
            if not config.api_key:
                return SummaryLLMConfig(
                    enabled=True,
                    provider=config.provider,
                    model_id=config.model_id,
                    api_key=self.api_key,
                    base_url=config.base_url,
                    temperature=config.temperature,
                )
            return config

        # 返回基于主 LLM 的配置（但禁用状态）
        return SummaryLLMConfig(
            enabled=False,
            provider=self.provider,
            model_id=self.model_id,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.3,
        )


class ExchangeConfig(BaseModel):
    """Exchange configuration for trading."""

    exchange_id: Optional[str] = Field(
        default=None,
        description="Exchange identifier (e.g., 'binance', 'okx', 'bybit')",
    )
    trading_mode: TradingMode = Field(
        default=TradingMode.VIRTUAL,
        description="Trading mode for this strategy",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Exchange API key",
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Exchange secret key",
    )
    passphrase: Optional[str] = Field(
        default=None,
        description="API passphrase (required for OKX)",
    )
    wallet_address: Optional[str] = Field(
        default=None,
        description="Wallet address (required for Hyperliquid)",
    )
    private_key: Optional[str] = Field(
        default=None,
        description="Private key (required for Hyperliquid)",
    )
    testnet: bool = Field(
        default=False,
        description="Use testnet/sandbox mode",
    )
    market_type: MarketType = Field(
        default=MarketType.SWAP,
        description="Market type: spot, future, or swap (perpetual)",
    )
    margin_mode: MarginMode = Field(
        default=MarginMode.CROSS,
        description="Margin mode: isolated or cross",
    )
    settle_coin: str = Field(
        default="USDT",
        description="Settlement coin for perpetual contracts (USDT or USDC)",
    )
    fee_bps: float = Field(
        default=DEFAULT_FEE_BPS,
        description="Trading fee in basis points for paper trading",
        gt=0,
    )

    @field_validator("settle_coin")
    @classmethod
    def validate_settle_coin(cls, v: str) -> str:
        """Validate and normalize settle_coin."""
        v = v.upper().strip()
        if v not in ("USDT", "USDC", "USD", "BUSD"):
            raise ValueError(f"settle_coin must be USDT, USDC, USD, or BUSD, got {v}")
        return v


class TradingConfig(BaseModel):
    """Trading strategy configuration."""

    strategy_name: Optional[str] = Field(
        default=None,
        description="User-friendly name for this strategy",
    )
    strategy_id: Optional[str] = Field(
        default=None,
        description="Reuse existing strategy id to continue execution",
    )
    initial_capital: Optional[float] = Field(
        default=DEFAULT_INITIAL_CAPITAL,
        description="Initial capital for trading in USD",
        gt=0,
    )
    max_leverage: float = Field(
        default=DEFAULT_MAX_LEVERAGE,
        description="Maximum leverage",
        gt=0,
    )
    max_positions: int = Field(
        default=DEFAULT_MAX_POSITIONS,
        description="Maximum number of concurrent positions",
        gt=0,
    )
    symbols: List[str] = Field(
        ...,
        description="List of crypto symbols to trade (e.g., ['BTC-USDT', 'ETH-USDT'])",
    )
    decide_interval: int = Field(
        default=DEFAULT_DECIDE_INTERVAL,
        description="Decision interval in seconds",
        gt=0,
    )
    template_id: Optional[str] = Field(
        default=None,
        description="Strategy template ID or name (e.g., 'default', 'aggressive')",
    )
    prompt_text: Optional[str] = Field(
        default=None,
        description="Direct prompt text (overrides template_id if provided)",
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt prefix to prepend to template/prompt_text",
    )
    cap_factor: float = Field(
        default=DEFAULT_CAP_FACTOR,
        description="Notional cap factor for per-symbol exposure",
        gt=0,
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one symbol is required")
        return [s.upper() for s in v]


class UserRequest(BaseModel):
    """User-specified strategy request / configuration."""

    llm_model_config: LLMModelConfig = Field(
        default_factory=LLMModelConfig,
        description="AI model configuration",
    )
    exchange_config: ExchangeConfig = Field(
        default_factory=ExchangeConfig,
        description="Exchange configuration",
    )
    trading_config: TradingConfig = Field(
        ...,
        description="Trading strategy configuration",
    )

    @model_validator(mode="before")
    @classmethod
    def _infer_market_type(cls, data):
        """Infer market_type from max_leverage when not provided."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        ex_cfg = dict(values.get("exchange_config") or {})
        mt_value = ex_cfg.get("market_type")
        mt_missing = (
            ("market_type" not in ex_cfg)
            or (mt_value is None)
            or (str(mt_value).strip() == "")
        )
        if mt_missing:
            tr_cfg = dict(values.get("trading_config") or {})
            ml_raw = tr_cfg.get("max_leverage")
            try:
                ml = float(ml_raw) if ml_raw is not None else float(DEFAULT_MAX_LEVERAGE)
            except Exception:
                ml = float(DEFAULT_MAX_LEVERAGE)
            ex_cfg["market_type"] = MarketType.SPOT if ml <= 1.0 else MarketType.SWAP
            values["exchange_config"] = ex_cfg
        return values


# =============================================================================
# Market Data Models
# =============================================================================

class InstrumentRef(BaseModel):
    """Identifies a tradable instrument."""

    symbol: str = Field(..., description="Exchange symbol, e.g., BTC/USDT")
    exchange_id: Optional[str] = Field(
        default=None,
        description="Exchange identifier (e.g., binance)",
    )


class Candle(BaseModel):
    """Aggregated OHLCV candle."""

    ts: int = Field(..., description="Candle end timestamp in ms")
    instrument: InstrumentRef
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = Field(..., description='Interval string, e.g., "1m", "5m"')


class FeatureVector(BaseModel):
    """Computed features for a single instrument at a point in time."""

    ts: int = Field(..., description="Feature vector timestamp in ms")
    instrument: InstrumentRef
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature name to value mapping",
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata about the source window",
    )


class CandleConfig(BaseModel):
    """Configuration for candle data fetching and feature computation."""

    interval: str = Field(
        ...,
        description='Candle interval, e.g., "1s", "1m", "5m", "1h"',
    )
    lookback: int = Field(
        ...,
        description="Number of candles to fetch",
        gt=0,
    )
    feature_computer: str = Field(
        default="default",
        description="Feature computer type to use for this interval",
    )


# =============================================================================
# Portfolio Models
# =============================================================================

class Constraints(BaseModel):
    """Typed constraints model for runtime and composer."""

    # 最大持仓数量
    max_positions: Optional[int] = Field(default=None)
    # 最大杠杆
    max_leverage: Optional[float] = Field(default=None)
    quantity_step: Optional[float] = Field(default=None)
    # 最小交易数量
    min_trade_qty: Optional[float] = Field(default=None)
    # 最大订单数量
    max_order_qty: Optional[float] = Field(default=None)
    # 最小名义价值
    min_notional: Optional[float] = Field(default=None)
    # 最大仓位数量
    max_position_qty: Optional[float] = Field(default=None)


class PositionSnapshot(BaseModel):
    """Current position snapshot for one instrument."""

    instrument: InstrumentRef
    quantity: float = Field(..., description="Position quantity (+long, -short)")
    avg_price: Optional[float] = Field(default=None, description="Average entry price")
    mark_price: Optional[float] = Field(default=None, description="Current mark price")
    unrealized_pnl: Optional[float] = Field(default=None, description="Unrealized PnL")
    unrealized_pnl_pct: Optional[float] = Field(default=None)
    notional: Optional[float] = Field(default=None)
    leverage: Optional[float] = Field(default=None)
    entry_ts: Optional[int] = Field(default=None)
    closed_ts: Optional[int] = Field(default=None)
    pnl_pct: Optional[float] = Field(default=None)
    trade_type: Optional[TradeType] = Field(default=None)


class PortfolioView(BaseModel):
    """Portfolio state used by the composer for decision-making."""

    strategy_id: Optional[str] = Field(default=None)
    ts: int
    account_balance: float = Field(..., description="Account cash balance")
    positions: Dict[str, PositionSnapshot] = Field(
        default_factory=dict,
        description="Map symbol -> PositionSnapshot",
    )
    gross_exposure: Optional[float] = Field(default=None)
    net_exposure: Optional[float] = Field(default=None)
    constraints: Optional[Constraints] = Field(default=None)
    total_value: Optional[float] = Field(default=None)
    total_unrealized_pnl: Optional[float] = Field(default=None)
    total_realized_pnl: Optional[float] = Field(default=None)
    buying_power: Optional[float] = Field(default=None)
    free_cash: Optional[float] = Field(default=None)


# =============================================================================
# Decision Models
# =============================================================================

class TradeDecisionItem(BaseModel):
    """Trade plan item from LLM decision."""

    instrument: InstrumentRef
    action: TradeDecisionAction
    target_qty: float = Field(
        default=0.0,
        description="Operation size for this action (units). Must be positive for actual operations.",
    )
    leverage: Optional[float] = Field(default=None)
    confidence: Optional[float] = Field(default=None, description="Confidence [0,1]")
    rationale: Optional[str] = Field(default=None)
    sl_price: Optional[float] = Field(
        default=None,
        description="Stop loss price. For long: sl < entry, for short: sl > entry",
    )
    tp_price: Optional[float] = Field(
        default=None,
        description="Take profit price. For long: tp > entry, for short: tp < entry",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_instrument(cls, data):
        """Allow string shorthand for InstrumentRef."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        instrument = values.get("instrument")
        if isinstance(instrument, str):
            values["instrument"] = {"symbol": instrument}
        return values


class TradePlanProposal(BaseModel):
    """Structured output from LLM before normalization."""

    ts: Optional[int] = Field(
        default_factory=get_current_timestamp_ms,
        description="Proposal timestamp in ms",
    )
    items: List[TradeDecisionItem] = Field(default_factory=list)
    rationale: Optional[str] = Field(default=None)


# =============================================================================
# Execution Models
# =============================================================================

class TradeInstruction(BaseModel):
    """Executable instruction after normalization."""

    instruction_id: str = Field(
        ...,
        description="Deterministic id for idempotency",
    )
    compose_id: str = Field(
        ...,
        description="Decision cycle id",
    )
    instrument: InstrumentRef
    action: Optional[TradeDecisionAction] = Field(default=None)
    side: TradeSide
    quantity: float = Field(..., description="Order quantity")
    leverage: Optional[float] = Field(default=None)
    price_mode: PriceMode = Field(PriceMode.MARKET)
    limit_price: Optional[float] = Field(default=None)
    max_slippage_bps: Optional[float] = Field(default=None)
    sl_price: Optional[float] = Field(
        default=None,
        description="Stop loss price to place after entry",
    )
    tp_price: Optional[float] = Field(
        default=None,
        description="Take profit price to place after entry",
    )
    meta: Optional[Dict[str, Any]] = Field(default=None)

    @model_validator(mode="after")
    def _validate_action_side_alignment(self):
        """Ensure action aligns with executable side."""
        act = self.action
        if act is None:
            return self
        try:
            if act == TradeDecisionAction.NOOP:
                return self
            if act in (TradeDecisionAction.OPEN_LONG, TradeDecisionAction.CLOSE_SHORT):
                expected = TradeSide.BUY
            elif act in (TradeDecisionAction.OPEN_SHORT, TradeDecisionAction.CLOSE_LONG):
                expected = TradeSide.SELL
            else:
                return self
            if self.side != expected:
                raise ValueError(
                    f"TradeInstruction.action={act} conflicts with side={self.side}"
                )
        except Exception:
            return self
        return self


class TxResult(BaseModel):
    """Result of executing a TradeInstruction."""

    instruction_id: str = Field(..., description="Originating instruction id")
    instrument: InstrumentRef
    side: TradeSide
    requested_qty: float = Field(..., description="Requested order quantity")
    filled_qty: float = Field(..., description="Filled quantity")
    avg_exec_price: Optional[float] = Field(default=None)
    slippage_bps: Optional[float] = Field(default=None)
    fee_cost: Optional[float] = Field(default=None)
    leverage: Optional[float] = Field(default=None)
    status: TxStatus = Field(default=TxStatus.FILLED)
    reason: Optional[str] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None)


# =============================================================================
# History & Digest Models
# =============================================================================

class TradeDigestEntry(BaseModel):
    """Digest stats per instrument."""

    instrument: InstrumentRef
    trade_count: int
    realized_pnl: float
    win_rate: Optional[float] = Field(default=None)
    avg_holding_ms: Optional[int] = Field(default=None)
    last_trade_ts: Optional[int] = Field(default=None)
    avg_entry_price: Optional[float] = Field(default=None)
    max_drawdown: Optional[float] = Field(default=None)
    recent_performance_score: Optional[float] = Field(default=None)


class TradeHistoryEntry(BaseModel):
    """Executed trade record for history and auditing."""

    trade_id: Optional[str] = Field(default=None)
    compose_id: Optional[str] = Field(default=None)
    instruction_id: Optional[str] = Field(default=None)
    strategy_id: Optional[str] = Field(default=None)
    instrument: InstrumentRef
    side: TradeSide
    type: TradeType
    quantity: float
    entry_price: Optional[float] = Field(default=None)
    exit_price: Optional[float] = Field(default=None)
    avg_exec_price: Optional[float] = Field(default=None)
    notional_entry: Optional[float] = Field(default=None)
    notional_exit: Optional[float] = Field(default=None)
    entry_ts: Optional[int] = Field(default=None)
    exit_ts: Optional[int] = Field(default=None)
    trade_ts: Optional[int] = Field(default=None)
    holding_ms: Optional[int] = Field(default=None)
    unrealized_pnl: Optional[float] = Field(default=None)
    realized_pnl: Optional[float] = Field(default=None)
    realized_pnl_pct: Optional[float] = Field(default=None)
    fee_cost: Optional[float] = Field(default=None)
    leverage: Optional[float] = Field(default=None)
    note: Optional[str] = Field(default=None)


class TradeDigest(BaseModel):
    """Compact digest for historical reference."""

    ts: int
    by_instrument: Dict[str, TradeDigestEntry] = Field(default_factory=dict)
    sharpe_ratio: Optional[float] = Field(default=None)


class StrategySummary(BaseModel):
    """Minimal summary for status views."""

    strategy_id: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    model_provider: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)
    exchange_id: Optional[str] = Field(default=None)
    mode: Optional[TradingMode] = Field(default=None)
    status: Optional[StrategyStatus] = Field(default=None)
    realized_pnl: Optional[float] = Field(default=None)
    pnl_pct: Optional[float] = Field(default=None)
    unrealized_pnl: Optional[float] = Field(default=None)
    unrealized_pnl_pct: Optional[float] = Field(default=None)
    total_value: Optional[float] = Field(default=None)
    last_updated_ts: Optional[int] = Field(default=None)


class HistoryRecord(BaseModel):
    """Generic persisted record for analysis."""

    ts: int
    kind: str = Field(..., description='"features" | "compose" | "instructions" | "execution"')
    reference_id: str = Field(..., description="Correlation id (e.g., compose_id)")
    payload: Dict[str, object] = Field(default_factory=dict)


# =============================================================================
# Context & Result Models
# =============================================================================

class ComposeContext(BaseModel):
    """Context assembled for the composer."""

    ts: int
    compose_id: str = Field(..., description="Decision cycle id")
    strategy_id: Optional[str] = Field(default=None)
    features: List[FeatureVector] = Field(default_factory=list)
    portfolio: PortfolioView
    digest: TradeDigest

    # 特征说明（来自特征计算器）
    feature_instructions: str = Field(
        default="",
        description="特征计算器提供的指标说明，帮助 LLM 理解数据含义"
    )

    # 短期记忆（最近决策历史）
    recent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="最近的决策记录，用于保持决策连贯性"
    )
    pending_signals: Dict[str, str] = Field(
        default_factory=dict,
        description="待观察的信号，如'等待回调再加仓'"
    )
    # 历史摘要（压缩的旧决策，用于长期记忆）
    history_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="历史决策摘要，包含周期范围、统计和LLM生成的摘要内容"
    )

    # 交易所真实订单历史（从交易所 API 获取）
    recent_exchange_orders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="从交易所获取的最近订单记录，包含真实的开平仓历史"
    )


class ComposeResult(BaseModel):
    """Result of a compose operation."""

    instructions: List[TradeInstruction]
    rationale: Optional[str] = None


class FeaturesPipelineResult(BaseModel):
    """Result of running a features pipeline."""

    features: List[FeatureVector]
    feature_instructions: str = Field(
        default="",
        description="Feature instructions for LLM, collected from feature computers",
    )


@dataclass
class DecisionCycleResult:
    """Outcome of a single decision cycle."""

    compose_id: str
    timestamp_ms: int
    cycle_index: int
    rationale: Optional[str]
    strategy_summary: StrategySummary
    instructions: List[TradeInstruction]
    trades: List[TradeHistoryEntry]
    history_records: List[HistoryRecord]
    digest: TradeDigest
    portfolio_view: PortfolioView
