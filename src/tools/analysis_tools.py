"""
分析类工具 - Analysis Tools

特点：
- 计算和分析功能
- 不修改状态
- 提供决策支持
"""
from langchain_core.tools import tool
from typing import Dict, List, Optional

from ..adapters import BaseExchangeAdapter
from ..models.strategy import TimeFrame


# 全局变量
_adapter: BaseExchangeAdapter
_market_analyzer = None


def set_adapter(adapter):
    """设置适配器实例"""
    global _adapter, _market_analyzer
    _adapter = adapter
    # 延迟导入避免循环依赖
    from ..agents.hansen.market_analyzer import MarketAnalyzer
    _market_analyzer = MarketAnalyzer(adapter=adapter)


def get_adapter() -> BaseExchangeAdapter:
    """获取适配器实例"""
    if _adapter is None:
        raise RuntimeError("Adapter not set. Call set_adapter() first.")
    return _adapter


def get_market_analyzer():
    """获取市场分析器实例"""
    if _market_analyzer is None:
        raise RuntimeError("MarketAnalyzer not initialized. Call set_adapter() first.")
    return _market_analyzer


async def _calculate_indicators_internal(
    symbol: str,
    timeframe: str = "15m"
) -> Dict:
    """内部函数：计算技术指标（供工具和其他函数调用）"""
    adapter = get_adapter()
    analyzer = get_market_analyzer()

    # 获取K线数据
    candles = await adapter.get_candles(symbol, timeframe, limit=100)

    # 转换时间周期
    tf_map = {
        "1m": TimeFrame.M1,
        "15m": TimeFrame.M15,
        "1h": TimeFrame.H1,
        "4h": TimeFrame.H4,
    }
    tf = tf_map.get(timeframe, TimeFrame.M15)

    # 计算指标
    indicators_map = await analyzer.calculate_indicators(
        symbol=symbol,
        candle_data={tf: candles}
    )

    indicators = indicators_map.get(tf, {})

    # 辅助函数：安全获取数组的第一个值或单值
    def safe_float(value):
        if value is None:
            return None
        # 如果是列表，取第一个元素（最新值）
        if isinstance(value, list):
            if len(value) > 0 and value[0] is not None:
                return float(value[0])
            return None
        # 如果是单值，直接转换
        return float(value)

    return {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        # 列表类型指标，取第一个元素（最新值）
        "ema_20": safe_float(indicators.get("ema_20")),
        "ema_50": safe_float(indicators.get("ema_50")),
        "rsi_14": safe_float(indicators.get("rsi_14")),  # 使用 RSI-14
        "macd_line": safe_float(indicators.get("macd_line")),
        "macd_signal": safe_float(indicators.get("macd_signal")),
        "macd_histogram": safe_float(indicators.get("macd_histogram")),
        # 单值类型指标
        "atr": safe_float(indicators.get("atr")),
        "atr_percent": safe_float(indicators.get("atr_percent")),
        "volume_24h": safe_float(indicators.get("volume_24h")),
        "obv": safe_float(indicators.get("obv")),
        "stoch_k": safe_float(indicators.get("stoch_k")),
        "stoch_d": safe_float(indicators.get("stoch_d")),
    }


@tool
async def calculate_technical_indicators(
    symbol: str,
    timeframe: str = "15m"
) -> Dict:
    """计算技术指标

    Args:
        symbol: 交易对，如 "BTC/USDC:USDC"
        timeframe: 时间周期，可选: "1m", "15m", "1h", "4h"

    Returns:
        Dict: 技术指标
            - ema_20: 20周期指数移动平均线（最新值）
            - ema_50: 50周期指数移动平均线（最新值）
            - rsi_14: 14周期相对强弱指标 (0-100)
            - macd_line: MACD线（最新值）
            - macd_signal: MACD信号线（最新值）
            - macd_histogram: MACD柱状图（最新值）
            - atr: 平均真实波幅
            - atr_percent: ATR百分比
            - volume_24h: 24小时成交量
            - obv: 能量潮指标
            - stoch_k: 随机指标K值
            - stoch_d: 随机指标D值

    示例:
        result = await calculate_technical_indicators("BTC/USDC:USDC", "15m")
        # {"rsi_14": 45.3, "ema_20": 65000, "macd_line": 120, ...}

    用途:
        - 判断超买超卖（RSI > 70 超买，< 30 超卖）
        - 判断趋势方向（EMA 多空排列）
        - 判断动量（MACD 金叉死叉）
    """
    try:
        return await _calculate_indicators_internal(symbol, timeframe)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "timeframe": timeframe,
        }


@tool
async def analyze_market_trend(
    symbol: str,
    timeframe: str = "15m"
) -> Dict:
    """分析市场趋势（客观指标分析）

    Args:
        symbol: 交易对
        timeframe: 时间周期

    Returns:
        Dict: 趋势分析结果（仅包含客观指标和比较关系）
            - ema_20: EMA20当前值
            - ema_50: EMA50当前值
            - ema_20_above_50: EMA20是否在EMA50上方
            - rsi_14: RSI14当前值
            - rsi_level: RSI所处区间 (0-30, 30-70, 70-100)
            - macd_line: MACD线当前值
            - macd_signal: MACD信号线当前值
            - macd_above_signal: MACD是否在信号线上方
            - macd_histogram: MACD柱状图当前值

    示例:
        result = await analyze_market_trend("BTC/USDC:USDC", "15m")
        # {"ema_20": 65000, "ema_50": 64500, "ema_20_above_50": true, "rsi_14": 45, ...}

    用途:
        - 获取关键技术指标的当前值
        - 获取指标间的客观比较关系
        - 提供无主观判断的市场数据
    """
    try:
        # 调用内部函数获取技术指标
        indicators_result = await _calculate_indicators_internal(symbol, timeframe)

        if not indicators_result.get("success"):
            return indicators_result

        indicators = indicators_result

        # 客观比较（不使用主观词汇）
        ema_20 = indicators.get("ema_20")
        ema_50 = indicators.get("ema_50")
        rsi_14 = indicators.get("rsi_14")
        macd_line = indicators.get("macd_line")
        macd_signal = indicators.get("macd_signal")
        macd_histogram = indicators.get("macd_histogram")

        # 计算客观关系
        ema_20_above_50 = None
        if ema_20 is not None and ema_50 is not None:
            ema_20_above_50 = ema_20 > ema_50

        # RSI 区间（客观描述）
        rsi_level = None
        if rsi_14 is not None:
            if rsi_14 < 30:
                rsi_level = "0-30"
            elif rsi_14 > 70:
                rsi_level = "70-100"
            else:
                rsi_level = "30-70"

        # MACD 比较
        macd_above_signal = None
        if macd_line is not None and macd_signal is not None:
            macd_above_signal = macd_line > macd_signal

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            # EMA 指标
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_20_above_50": ema_20_above_50,
            # RSI 指标
            "rsi_14": rsi_14,
            "rsi_level": rsi_level,
            # MACD 指标
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_above_signal": macd_above_signal,
            "macd_histogram": macd_histogram,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
        }


def _calculate_risk_reward_ratio_internal(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> Dict:
    """内部函数：计算风险回报比（供工具和其他函数调用）"""
    # 计算风险和回报
    risk_amount = abs(entry_price - stop_loss)
    reward_amount = abs(take_profit - entry_price)

    if risk_amount == 0:
        return {
            "success": False,
            "error": "止损价格不能等于入场价格"
        }

    ratio = reward_amount / risk_amount

    risk_percent = (risk_amount / entry_price) * 100
    reward_percent = (reward_amount / entry_price) * 100

    # 建议
    if ratio >= 3:
        recommendation = "优秀 - 风险回报比>=3，符合交易标准"
    elif ratio >= 2:
        recommendation = "良好 - 风险回报比>=2，可以考虑"
    elif ratio >= 1:
        recommendation = "一般 - 风险回报比>=1，需谨慎"
    else:
        recommendation = "不佳 - 风险回报比<1，不建议交易"

    return {
        "success": True,
        "ratio": round(ratio, 2),
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(reward_amount, 2),
        "risk_percent": round(risk_percent, 2),
        "reward_percent": round(reward_percent, 2),
        "recommendation": recommendation,
    }


@tool
def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> Dict:
    """计算风险回报比

    Args:
        entry_price: 入场价格
        stop_loss: 止损价格
        take_profit: 止盈价格

    Returns:
        Dict: 风险回报分析
            - ratio: 风险回报比
            - risk_amount: 风险金额（点数）
            - reward_amount: 回报金额（点数）
            - risk_percent: 风险百分比
            - reward_percent: 回报百分比
            - recommendation: 建议

    示例:
        result = calculate_risk_reward_ratio(65000, 64000, 68000)
        # {"ratio": 3.0, "risk_amount": 1000, "reward_amount": 3000}

    注意:
        - 建议风险回报比 >= 1:3（即 ratio >= 3）
        - 止损和止盈必须合理设置
    """
    try:
        return _calculate_risk_reward_ratio_internal(entry_price, stop_loss, take_profit)
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
async def calculate_position_size(
    symbol: str,
    risk_percentage: float = 1.0,
    stop_loss_distance: Optional[float] = None
) -> Dict:
    """计算合理仓位大小

    Args:
        symbol: 交易对
        risk_percentage: 风险百分比（账户的%），默认1%
        stop_loss_distance: 止损距离（价格点数），可选

    Returns:
        Dict: 仓位计算结果
            - amount: 建议交易数量
            - leverage: 建议杠杆
            - margin_required: 所需保证金
            - risk_amount: 风险金额
            - max_loss: 最大亏损

    示例:
        result = await calculate_position_size("BTC/USDC:USDC", 1.0, 1000)
        # {"amount": 0.01, "leverage": 5, "margin_required": 130}

    注意:
        - risk_percentage 建议不超过 2%
        - 会根据账户余额自动计算
    """
    try:
        from .query_tools import _get_account_balance_internal

        # 获取账户余额
        balance_result = await _get_account_balance_internal()
        if not balance_result.get("success"):
            return balance_result

        available = balance_result["available"]

        # 计算风险金额
        risk_amount = available * (risk_percentage / 100)

        # 如果提供了止损距离，计算仓位
        if stop_loss_distance and stop_loss_distance > 0:
            # 简化计算：amount = risk_amount / stop_loss_distance
            amount = risk_amount / stop_loss_distance

            # 建议杠杆（简化）
            leverage = min(5, int(available / risk_amount)) if risk_amount > 0 else 1

            return {
                "success": True,
                "symbol": symbol,
                "amount": round(amount, 4),
                "leverage": leverage,
                "margin_required": round(risk_amount, 2),
                "risk_amount": round(risk_amount, 2),
                "max_loss": round(risk_amount, 2),
                "recommendation": f"建议仓位{amount:.4f}，杠杆{leverage}x"
            }
        else:
            # 没有止损距离，给出保守建议
            return {
                "success": True,
                "symbol": symbol,
                "risk_amount": round(risk_amount, 2),
                "recommendation": f"风险金额{risk_amount:.2f}，请设置止损后重新计算"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
