"""
性能类工具 - Performance Tools

特点：
- 追踪交易绩效
- 提供历史数据
- 帮助评估策略表现
"""
from langchain_core.tools import tool
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal


# 全局变量
_adapter = None


def set_adapter(adapter):
    """设置适配器实例"""
    global _adapter
    _adapter = adapter


def get_adapter():
    """获取适配器"""
    if _adapter is None:
        raise RuntimeError("Adapter not set. Call set_adapter() first.")
    return _adapter


@tool
async def get_trading_performance(
    symbol: Optional[str] = None,
    days: int = 7
) -> Dict:
    """获取交易绩效统计

    Args:
        symbol: 交易对，不传则统计所有交易对
        days: 统计天数，默认7天

    Returns:
        Dict: 绩效统计
            - total_trades: 总交易次数
            - winning_trades: 盈利交易次数
            - losing_trades: 亏损交易次数
            - win_rate: 胜率 (0-100)
            - total_pnl: 总盈亏
            - avg_win: 平均盈利
            - avg_loss: 平均亏损
            - largest_win: 最大盈利
            - largest_loss: 最大亏损
            - profit_factor: 盈亏比 (total_profit / total_loss)
            - avg_holding_time: 平均持仓时间（分钟）

    示例:
        # 获取所有交易对的7天绩效
        result = await get_trading_performance()

        # 获取BTC的30天绩效
        result = await get_trading_performance("BTC/USDC:USDC", 30)

    用途:
        - 评估策略表现
        - 识别需要改进的地方
        - 追踪进度
    """
    try:
        adapter = get_adapter()

        # 计算开始时间
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # 获取历史交易记录（这里需要adapter支持）
        # 注意：某些交易所可能不支持历史查询，这种情况需要本地数据库
        try:
            # 尝试从交易所获取
            trades = await adapter.get_trades_history(symbol, start_time, end_time)
        except (AttributeError, NotImplementedError):
            # 如果交易所不支持，返回提示信息
            return {
                "success": False,
                "error": "当前交易所不支持历史交易查询",
                "recommendation": "建议使用本地数据库记录交易历史"
            }

        if not trades or len(trades) == 0:
            return {
                "success": True,
                "total_trades": 0,
                "message": f"过去{days}天没有交易记录"
            }

        # 统计交易数据
        total_trades = len(trades)
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        profits = []
        losses = []
        holding_times = []

        for trade in trades:
            pnl = float(trade.get("pnl", 0) or 0)

            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
                profits.append(pnl)
            elif pnl < 0:
                losing_trades += 1
                total_loss += abs(pnl)
                losses.append(abs(pnl))

            # 计算持仓时间（如果有开仓和平仓时间）
            if trade.get("open_time") and trade.get("close_time"):
                holding_time = (trade["close_time"] - trade["open_time"]).total_seconds() / 60
                holding_times.append(holding_time)

        # 计算统计指标
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = total_profit - total_loss
        avg_win = (total_profit / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
        largest_win = max(profits) if profits else 0
        largest_loss = max(losses) if losses else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        avg_holding_time = (sum(holding_times) / len(holding_times)) if holding_times else 0

        # 评级
        if win_rate >= 60 and profit_factor >= 2:
            rating = "优秀"
        elif win_rate >= 50 and profit_factor >= 1.5:
            rating = "良好"
        elif win_rate >= 40 and profit_factor >= 1:
            rating = "一般"
        else:
            rating = "需要改进"

        return {
            "success": True,
            "symbol": symbol or "ALL",
            "period_days": days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_holding_time": round(avg_holding_time, 2),
            "rating": rating,
            "summary": f"{days}天内交易{total_trades}次，胜率{win_rate:.1f}%，总盈亏{total_pnl:.2f}，评级:{rating}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"获取绩效统计失败: {str(e)}"
        }


@tool
async def get_recent_trades(
    symbol: Optional[str] = None,
    limit: int = 10
) -> Dict:
    """获取最近交易记录

    Args:
        symbol: 交易对，不传则返回所有交易对
        limit: 返回数量，默认10条

    Returns:
        Dict: 交易记录列表
            - trades: 交易详情列表
            - count: 记录数量

    每条交易包含:
        - symbol: 交易对
        - side: 方向 (LONG/SHORT)
        - action: 动作 (OPEN/CLOSE)
        - amount: 数量
        - price: 价格
        - pnl: 盈亏（平仓时）
        - timestamp: 时间
        - order_id: 订单ID

    示例:
        # 获取最近10条交易
        result = await get_recent_trades()

        # 获取BTC最近5条交易
        result = await get_recent_trades("BTC/USDC:USDC", 5)

    用途:
        - 查看交易历史
        - 分析最近操作
        - 追踪订单状态
    """
    try:
        adapter = get_adapter()

        # 获取最近的订单历史
        try:
            orders = await adapter.get_orders_history(symbol, limit=limit)
        except (AttributeError, NotImplementedError):
            # 如果不支持订单历史查询，尝试从活跃订单和持仓推断
            return {
                "success": False,
                "error": "当前交易所不支持订单历史查询",
                "recommendation": "建议使用本地数据库记录订单历史"
            }

        if not orders or len(orders) == 0:
            return {
                "success": True,
                "trades": [],
                "count": 0,
                "message": "没有找到交易记录"
            }

        # 格式化交易记录
        trades_data = []
        for order in orders[:limit]:
            trade_info = {
                "symbol": order.get("symbol"),
                "side": order.get("side", "").upper(),
                "action": order.get("type", "").upper(),  # OPEN/CLOSE
                "amount": float(order.get("amount", 0)),
                "price": float(order.get("price", 0)),
                "filled": float(order.get("filled", 0)),
                "status": order.get("status", "").upper(),
                "timestamp": order.get("timestamp").isoformat() if order.get("timestamp") else None,
                "order_id": order.get("id"),
            }

            # 如果是平仓订单，包含盈亏信息
            if order.get("pnl") is not None:
                trade_info["pnl"] = float(order.get("pnl"))

            # 如果有手续费信息
            if order.get("fee"):
                trade_info["fee"] = float(order.get("fee", {}).get("cost", 0))

            trades_data.append(trade_info)

        # 计算总盈亏（如果有）
        total_pnl = sum(
            float(t.get("pnl", 0))
            for t in trades_data
            if t.get("pnl") is not None
        )

        return {
            "success": True,
            "trades": trades_data,
            "count": len(trades_data),
            "total_pnl": round(total_pnl, 2) if total_pnl != 0 else None,
            "summary": f"最近{len(trades_data)}条交易记录"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"获取交易记录失败: {str(e)}",
            "trades": [],
            "count": 0,
        }
