from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, TypedDict, Annotated
from operator import add

# ==================== 枚举定义 ====================

class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class MarketRegime(Enum):
    """市场状态"""
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class SystemStatus(Enum):
    """系统状态"""
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOPPED = "emergency_stopped"
    MAINTENANCE = "maintenance"

class WorkflowState(Enum):
    """工作流状态节点"""
    FETCH_DATA = "fetch_market_data"
    ANALYZE = "analyze_market"
    GENERATE_SIGNALS = "generate_signals"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTE_TRADES = "execute_trades"
    MONITOR = "monitor_positions"
    OPTIMIZE = "update_strategy"
    ERROR_HANDLER = "handle_errors"

class RoutingDecision(Enum):
    """路由决策"""
    ANALYZE = "analyze"
    SIGNAL = "signal"
    EXECUTE = "execute"
    MONITOR = "monitor"
    OPTIMIZE = "optimize"
    ERROR = "error"
    SKIP = "skip"
    REJECT = "reject"
    APPROVED = "approved"
    REJECTED = "rejected"
    ADJUST = "adjust"

class RecoveryAction(Enum):
    """恢复策略动作"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    EMERGENCY_STOP = "emergency_stop"

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# ==================== 数据类定义 ====================

class TradingState(TypedDict):
    # 市场数据
    market_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    sentiment_scores: Dict[str, float]

    # 决策状态
    current_analysis: str
    trading_signals: List[str]
    risk_assessment: Dict

    # 执行状态
    pending_orders: List[Dict]
    active_positions: List[Dict]
    execution_results: List[Dict]

    # 记忆与学习（⚠️ 不再使用 Annotated[List, add]，改用普通 List 避免重复累积）
    # 历史数据的累积和清理由 update_strategy 节点手动管理
    historical_decisions: List[Dict]  # 决策历史
    trade_history: List[Dict]  # 交易历史
    performance_metrics: Dict[str, float]
    strategy_parameters: Dict[str, Any]

    # 控制流
    next_action: str
    retry_count: int
    error_log: List[str]

@dataclass
class TradingSignal:
    signal: SignalType
    confidence: float  # 0-100
    position_size: float  # 0-100%
    entry_price: float
    stop_loss: float
    take_profit: List[float]  # 多个止盈目标
    risk_level: RiskLevel
    reasoning: str
    timestamp: str

@dataclass
class MarketData:
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    high_24h: float
    low_24h: float
    bid_ask_spread: float
    order_book_depth: Dict
    recent_trades: List