"""
Strategy trading models

Data models for strategy-based trading decisions.
"""
from typing import Optional, List, Dict, Any
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class TimeFrame(str, Enum):
    """Time frame for market data"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class TrendDirection(str, Enum):
    """Trend direction"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class MarketIndicators(BaseModel):
    """Technical indicators for a single timeframe"""
    timeframe: TimeFrame

    # Price action
    current_price: Decimal
    high_24h: Decimal
    low_24h: Decimal

    # Moving averages (arrays: most recent 10 points, newest to oldest)
    ema_20: Optional[List[Decimal]] = None
    ema_50: Optional[List[Decimal]] = None
    ema_200: Optional[List[Decimal]] = None

    # MACD (arrays: most recent 10 points, newest to oldest)
    macd_line: Optional[List[Decimal]] = None
    macd_signal: Optional[List[Decimal]] = None
    macd_histogram: Optional[List[Decimal]] = None

    # RSI (arrays: most recent 10 points, newest to oldest)
    rsi_7: Optional[List[Decimal]] = None   # 7-period RSI
    rsi_14: Optional[List[Decimal]] = None  # 14-period RSI

    # ATR (volatility)
    atr: Optional[Decimal] = None
    atr_percent: Optional[Decimal] = None  # ATR as % of price

    # Volume
    volume_24h: Optional[Decimal] = None

    # Market sentiment (Perpetual contracts specific)
    funding_rate: Optional[Decimal] = None  # Current funding rate
    open_interest: Optional[Decimal] = None  # Open interest in USD


class TrendAnalysis(BaseModel):
    """Trend analysis across timeframes"""
    timeframe: TimeFrame
    direction: TrendDirection
    strength: Decimal = Field(..., ge=0, le=100, description="Trend strength 0-100")

    # Reasons for trend determination
    above_ema_20: bool = False
    above_ema_50: bool = False
    macd_positive: bool = False
    macd_rising: bool = False

    # Support/Resistance
    nearest_support: Optional[Decimal] = None
    nearest_resistance: Optional[Decimal] = None


class MarketAnalysis(BaseModel):
    """Complete market analysis for a symbol"""
    symbol: str
    timestamp: datetime

    # Multi-timeframe indicators
    indicators_15m: Optional[MarketIndicators] = None
    indicators_4h: Optional[MarketIndicators] = None

    # Multi-timeframe trends
    trend_15m: Optional[TrendAnalysis] = None
    trend_4h: Optional[TrendAnalysis] = None

    # Overall signal
    signal_strength: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    signal_direction: Optional[TrendDirection] = None

    # Entry conditions
    trend_alignment: bool = Field(default=False, description="Multiple timeframes aligned")
    technical_confirmation: int = Field(default=0, description="Number of confirmed indicators")
    volume_support: bool = Field(default=False, description="Volume supports direction")


class StrategyDecision(str, Enum):
    """Strategy decision actions"""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_POSITION = "close_position"
    HOLD = "hold"
    WAIT = "wait"
    MODIFY_SL_TP = "modify_sl_tp"


class StrategySignal(BaseModel):
    """Strategy trading signal"""
    action: StrategyDecision
    symbol: str
    confidence: Decimal = Field(..., ge=0, le=100, description="Confidence 0-100")

    # Position details (for open actions)
    amount: Optional[Decimal] = None
    leverage: Optional[int] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # Reasoning
    reasoning: str = Field(..., description="Chain-of-thought reasoning")
    signal_checks: Dict[str, bool] = Field(default_factory=dict, description="Signal validation checks")
    risk_reward_ratio: Optional[Decimal] = None

    # Sharpe-based constraints
    sharpe_adjustment: Optional[str] = None  # Description of any Sharpe-based adjustments
    position_size_factor: Decimal = Field(default=Decimal("1.0"), description="Position size multiplier based on performance")


class PerformanceMetrics(BaseModel):
    """Trading performance metrics"""
    # Sharpe ratio (primary metric)
    sharpe_ratio: Decimal = Field(default=Decimal("0"))

    # Returns
    total_pnl: Decimal = Field(default=Decimal("0"))
    total_pnl_percent: Decimal = Field(default=Decimal("0"))

    # Trade statistics
    total_trades: int = Field(default=0)
    winning_trades: int = Field(default=0)
    losing_trades: int = Field(default=0)
    win_rate: Decimal = Field(default=Decimal("0"), ge=0, le=100)

    # Risk metrics
    max_drawdown: Decimal = Field(default=Decimal("0"))
    max_drawdown_percent: Decimal = Field(default=Decimal("0"))

    # Recent performance
    trades_last_hour: int = Field(default=0)
    trades_last_24h: int = Field(default=0)
    pnl_last_24h: Decimal = Field(default=Decimal("0"))

    # Volatility
    returns_std_dev: Decimal = Field(default=Decimal("0"))

    # Constraints
    over_trading_risk: bool = Field(default=False, description=">2 trades per hour")
    consecutive_losses: int = Field(default=0)
    daily_loss_limit_hit: bool = Field(default=False)


class StrategyState(BaseModel):
    """Extended state for strategy trading"""
    # Session info
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Market analysis
    market_analysis: Optional[MarketAnalysis] = None

    # Performance tracking
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)

    # Strategy decision
    strategy_signal: Optional[StrategySignal] = None

    # Current positions (from base state)
    current_positions: List[Any] = Field(default_factory=list)

    # Account balance (from base state)
    account_balance: Optional[Any] = None

    # Execution history (for Sharpe calculation)
    trade_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Error handling
    error: Optional[str] = None
