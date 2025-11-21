"""
查询类工具 - Query Tools

特点：
- 只读操作，安全
- 不会修改任何状态
- 适合测试阶段使用
"""
from langchain_core.tools import tool
from typing import Dict, List, Optional
from decimal import Decimal

from src.adapters import BaseExchangeAdapter

# 全局变量，用于存储 adapter 实例
# 在创建 agent 时通过 set_adapter() 注入
_adapter = None


def set_adapter(adapter):
    """设置交易所适配器实例"""
    global _adapter
    _adapter = adapter


def get_adapter() -> BaseExchangeAdapter:
    """获取交易所适配器实例"""
    if _adapter is None:
        raise RuntimeError("Adapter not set. Call set_adapter() first.")
    return _adapter


async def _get_account_balance_internal() -> Dict:
    """内部函数：获取账户余额（供工具和其他函数调用）"""
    adapter = get_adapter()
    balance = await adapter.get_balance()

    result = {
        "success": True,
        "total": float(balance.total),
        "available": float(balance.available),
        "frozen": float(balance.frozen) if hasattr(balance, 'frozen') else 0.0,
    }

    # 添加可选字段
    if hasattr(balance, 'equity') and balance.equity is not None:
        result["equity"] = float(balance.equity)

    if hasattr(balance, 'margin') and balance.margin is not None:
        result["margin"] = float(balance.margin)

    if hasattr(balance, 'unrealized_pnl') and balance.unrealized_pnl is not None:
        result["unrealized_pnl"] = float(balance.unrealized_pnl)

    return result


@tool
async def get_account_balance() -> Dict:
    """获取账户余额信息

    Returns:
        Dict: 账户余额详情
            - total: 总资产
            - available: 可用资金
            - frozen: 冻结金额
            - margin: 保证金（如果有）
            - equity: 净资产

    示例:
        result = await get_account_balance()
        # {"total": 10000.0, "available": 8000.0, "frozen": 2000.0}
    """
    try:
        return await _get_account_balance_internal()
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def _get_current_positions_internal(symbol: Optional[str] = None) -> Dict:
    """内部函数：获取当前持仓（供工具和其他函数调用）"""
    adapter = get_adapter()
    positions = await adapter.get_positions(symbol)

    positions_data = []
    total_margin = 0.0

    for p in positions:
        pos_data = {
            "symbol": p.symbol,
            "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
            "amount": float(p.amount),
            "entry_price": float(p.entry_price),
            "current_price": float(p.current_price) if hasattr(p, 'current_price') else None,
            "unrealized_pnl": float(p.unrealized_pnl) if p.unrealized_pnl else 0.0,
            "leverage": float(p.leverage) if hasattr(p, 'leverage') else 1,
            "margin": float(p.margin) if hasattr(p, 'margin') else 0.0,
            "stop_loss": float(p.stop_loss) if p.stop_loss else None,
            "take_profit": float(p.take_profit) if p.take_profit else None,
        }
        positions_data.append(pos_data)
        total_margin += pos_data["margin"]

    return {
        "success": True,
        "positions": positions_data,
        "total_count": len(positions_data),
        "total_margin": total_margin,
    }


@tool
async def get_current_positions(symbol: Optional[str] = None) -> Dict:
    """获取当前持仓信息

    Args:
        symbol: 交易对，如 "BTC/USDC:USDC"。不传则返回所有持仓

    Returns:
        Dict: 持仓列表
            - positions: 持仓详情列表
            - total_count: 持仓总数
            - total_margin: 总保证金

    示例:
        # 获取所有持仓
        result = await get_current_positions()

        # 获取特定交易对持仓
        result = await get_current_positions("BTC/USDC:USDC")
    """
    try:
        return await _get_current_positions_internal(symbol)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "positions": [],
            "total_count": 0,
        }


async def _get_market_price_internal(symbol: str) -> Dict:
    """内部函数：获取实时市场价格（供工具和其他函数调用）"""
    adapter = get_adapter()
    ticker = await adapter.get_ticker(symbol)

    return {
        "success": True,
        "symbol": symbol,
        "bid": float(ticker["bid"]) if ticker.get("bid") else None,
        "ask": float(ticker["ask"]) if ticker.get("ask") else None,
        "last": float(ticker["last"]) if ticker.get("last") else None,
        "volume_24h": float(ticker["volume"]) if ticker.get("volume") else None,
        "change_24h": float(ticker["change_24h"]) if ticker.get("change_24h") else None,
    }


@tool
async def get_market_price(symbol: str) -> Dict:
    """获取实时市场价格

    Args:
        symbol: 交易对，如 "BTC/USDC:USDC"

    Returns:
        Dict: 价格信息
            - symbol: 交易对
            - bid: 买价
            - ask: 卖价
            - last: 最新成交价
            - volume_24h: 24小时成交量
            - change_24h: 24小时涨跌幅

    示例:
        result = await get_market_price("BTC/USDC:USDC")
        # {"symbol": "BTC/USDC:USDC", "last": 65000.0, "bid": 64999.0, ...}
    """
    try:
        return await _get_market_price_internal(symbol)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
        }


@tool
async def get_candles_data(
    symbol: str,
    timeframe: str = "5m",
    limit: int = 100
) -> Dict:
    """获取K线数据

    Args:
        symbol: 交易对，如 "BTC/USDC:USDC"
        timeframe: 时间周期，可选: "1m", "5m", "15m", "1h", "4h"
        limit: 返回数量，默认100根K线

    Returns:
        Dict: K线数据
            - symbol: 交易对
            - timeframe: 时间周期
            - candles: K线列表
            - count: K线数量

    示例:
        result = await get_candles_data("BTC/USDC:USDC", "5m", 100)
        # {"candles": [{timestamp, open, high, low, close, volume}, ...]}

    注意:
        - 返回的K线按时间升序排列（最新的在最后）
        - 可用于技术分析和趋势判断
    """
    try:
        adapter = get_adapter()
        candles = await adapter.get_candles(symbol, timeframe, limit)

        candles_data = []
        for candle in candles:
            candles_data.append({
                "timestamp": candle.timestamp.isoformat() if hasattr(candle.timestamp, 'isoformat') else str(candle.timestamp),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            })

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles_data,
            "count": len(candles_data),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": [],
            "count": 0,
        }
