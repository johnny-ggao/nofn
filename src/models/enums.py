"""
交易系统枚举类型定义
"""
from enum import Enum


class TradingAction(str, Enum):
    """交易动作类型"""
    OPEN_LONG = "open_long"  # 开多仓
    OPEN_SHORT = "open_short"  # 开空仓
    CLOSE_POSITION = "close_position"  # 平仓
    MODIFY_SL_TP = "modify_sl_tp"  # 修改止损止盈
    CANCEL_ORDER = "cancel_order"  # 取消订单
    QUERY_POSITION = "query_position"  # 查询持仓
    QUERY_BALANCE = "query_balance"  # 查询余额


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单
    STOP_MARKET = "stop_market"  # 止损市价单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "long"  # 多头
    SHORT = "short"  # 空头


class ExchangeName(str, Enum):
    """支持的交易所"""
    HYPERLIQUID = "hyperliquid"
    BINANCE = "binance"
    OKX = "okx"


class ExecutionStatus(str, Enum):
    """执行状态"""
    SUCCESS = "success"  # 成功
    FAILED = "failed"  # 失败
    PENDING = "pending"  # 等待中
    PARTIAL = "partial"  # 部分成功


# 以下枚举已废弃，仅为兼容性保留
# class AgentType(str, Enum):
#     """Agent 类型（已废弃）"""
#     TRADING = "trading"
#     RISK = "risk"
#     SUPERVISOR = "supervisor"
#
# class RiskCheckResult(str, Enum):
#     """风控检查结果（已废弃）"""
#     APPROVED = "approved"
#     REJECTED = "rejected"
#     NEEDS_CONFIRMATION = "needs_confirmation"