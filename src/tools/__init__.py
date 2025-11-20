"""
Trading Tools for LangChain Agent

工具分类：
1. 查询类工具（query_tools.py）- 只读，安全
2. 分析类工具（analysis_tools.py）- 计算和分析
3. 交易类工具（trading_tools.py）- 执行交易，需要权限控制
4. 性能类工具（performance_tools.py）- 绩效追踪
"""

from .query_tools import (
    get_account_balance,
    get_current_positions,
    get_market_price,
    get_candles_data,
)

from .analysis_tools import (
    calculate_technical_indicators,
    analyze_market_trend,
    calculate_risk_reward_ratio,
    calculate_position_size,
)

from .trading_tools import (
    open_long_position,
    open_short_position,
    close_position,
    set_stop_loss,
    set_take_profit,
)

from .performance_tools import (
    get_trading_performance,
    get_recent_trades,
)

# 适配器设置函数
from .query_tools import set_adapter as set_query_adapter
from .analysis_tools import set_adapter as set_analysis_adapter
from .trading_tools import set_adapter as set_trading_adapter
from .performance_tools import set_adapter as set_performance_adapter


def set_adapter(adapter):
    """设置所有工具模块的适配器

    Args:
        adapter: 交易所适配器实例

    注意:
        必须在使用任何工具之前调用此函数
    """
    set_query_adapter(adapter)
    set_analysis_adapter(adapter)
    set_trading_adapter(adapter)
    set_performance_adapter(adapter)

# 所有工具列表
ALL_TOOLS = [
    # 查询类
    get_account_balance,
    get_current_positions,
    get_market_price,
    get_candles_data,

    # 分析类
    calculate_technical_indicators,
    analyze_market_trend,
    calculate_risk_reward_ratio,
    calculate_position_size,

    # 交易类
    open_long_position,
    open_short_position,
    close_position,
    set_stop_loss,
    set_take_profit,

    # 性能类
    get_trading_performance,
    get_recent_trades,
]

__all__ = [
    "ALL_TOOLS",
    "set_adapter",
    # 查询类
    "get_account_balance",
    "get_current_positions",
    "get_market_price",
    "get_candles_data",
    # 分析类
    "calculate_technical_indicators",
    "analyze_market_trend",
    "calculate_risk_reward_ratio",
    "calculate_position_size",
    # 交易类
    "open_long_position",
    "open_short_position",
    "close_position",
    "set_stop_loss",
    "set_take_profit",
    # 性能类
    "get_trading_performance",
    "get_recent_trades",
]
