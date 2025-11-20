"""
交易类工具 - Trading Tools

特点：
- 执行实际交易
- 需要严格权限控制
- 必须进行风险检查
- 需要明确的止损设置

警告：
- 这些工具会真实执行交易
- 务必在测试模式下充分验证
- 建议从小仓位开始
"""
from langchain_core.tools import tool
from typing import Dict, Optional
from decimal import Decimal


# 全局变量
_adapter = None
_trading_executor = None
_risk_manager = None


def set_adapter(adapter):
    """设置适配器实例"""
    global _adapter, _trading_executor, _risk_manager
    _adapter = adapter
    # 延迟导入避免循环依赖
    from ..agents.hansen.trading_executor import TradingExecutor
    from ..agents.hansen.risk_manager import RiskManager
    _trading_executor = TradingExecutor(adapter=adapter)
    _risk_manager = RiskManager()


def get_adapter():
    """获取适配器"""
    if _adapter is None:
        raise RuntimeError("Adapter not set. Call set_adapter() first.")
    return _adapter


def get_trading_executor():
    """获取交易执行器"""
    if _trading_executor is None:
        raise RuntimeError("TradingExecutor not initialized.")
    return _trading_executor


def get_risk_manager():
    """获取风险管理器"""
    if _risk_manager is None:
        raise RuntimeError("RiskManager not initialized.")
    return _risk_manager


@tool
async def open_long_position(
    symbol: str,
    amount: float,
    leverage: int = 1,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> Dict:
    """开多仓（做多）

    Args:
        symbol: 交易对，如 "BTC/USDC:USDC"
        amount: 交易数量（币的数量）
        leverage: 杠杆倍数，建议1-5x
        stop_loss: 止损价格（强制要求）
        take_profit: 止盈价格（可选）

    Returns:
        Dict: 交易结果
            - success: 是否成功
            - order_id: 订单ID
            - filled_price: 成交价格
            - amount: 成交数量
            - margin_required: 所需保证金

    示例:
        # 开BTC多仓，数量0.01，5倍杠杆，止损64000，止盈68000
        result = await open_long_position(
            symbol="BTC/USDC:USDC",
            amount=0.01,
            leverage=5,
            stop_loss=64000,
            take_profit=68000
        )

    警告:
        - 必须设置止损，否则拒绝交易
        - 杠杆过高会增加风险
        - 会自动进行风险检查
        - 建议风险回报比 >= 1:3

    注意:
        - 这是真实交易，会消耗保证金
        - 务必确认参数正确
    """
    try:
        # 1. 强制止损检查
        if not stop_loss or stop_loss <= 0:
            return {
                "success": False,
                "error": "必须设置止损价格（stop_loss）",
                "reason": "风险控制要求：所有交易必须设置止损"
            }

        # 2. 获取当前价格
        from .query_tools import _get_market_price_internal
        price_result = await _get_market_price_internal(symbol)
        if not price_result.get("success"):
            return {
                "success": False,
                "error": f"无法获取价格: {price_result.get('error')}"
            }

        current_price = price_result["last"]

        # 3. 验证止损合理性（多仓止损必须低于当前价）
        if stop_loss >= current_price:
            return {
                "success": False,
                "error": f"多仓止损价格({stop_loss})必须低于当前价格({current_price})"
            }

        # 4. 验证止盈合理性（如果设置）
        if take_profit and take_profit <= current_price:
            return {
                "success": False,
                "error": f"多仓止盈价格({take_profit})必须高于当前价格({current_price})"
            }

        # 5. 计算风险回报比
        if take_profit:
            from .analysis_tools import _calculate_risk_reward_ratio_internal
            rr_result = _calculate_risk_reward_ratio_internal(current_price, stop_loss, take_profit)
            if rr_result.get("success") and rr_result["ratio"] < 2.0:
                return {
                    "success": False,
                    "error": f"风险回报比({rr_result['ratio']:.2f})过低，建议 >= 2",
                    "recommendation": "调整止盈止损以提高风险回报比"
                }

        # 6. 风险检查（暂时简化，后续可增强）
        # risk_manager = get_risk_manager()
        # 这里可以添加更多风险检查...

        # 7. 执行交易
        executor = get_trading_executor()

        # 准备交易信号
        signal = {
            "symbol": symbol,
            "action": "open_long",
            "amount": amount,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        results = await executor.execute_signals([signal], [])

        if results and len(results) > 0:
            result = results[0]
            if result.get("success"):
                return {
                    "success": True,
                    "order_id": result.get("result", {}).get("order_id"),
                    "symbol": symbol,
                    "side": "LONG",
                    "amount": amount,
                    "leverage": leverage,
                    "filled_price": result.get("result", {}).get("price"),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "message": "多仓开仓成功"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "开仓失败")
                }
        else:
            return {
                "success": False,
                "error": "执行交易时无返回结果"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"开仓失败: {str(e)}"
        }


@tool
async def open_short_position(
    symbol: str,
    amount: float,
    leverage: int = 1,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> Dict:
    """开空仓（做空）

    Args:
        symbol: 交易对
        amount: 交易数量
        leverage: 杠杆倍数
        stop_loss: 止损价格（强制要求）
        take_profit: 止盈价格（可选）

    Returns:
        Dict: 交易结果

    注意:
        - 空仓止损必须高于当前价格
        - 空仓止盈必须低于当前价格
        - 其他规则同开多仓
    """
    try:
        # 1. 强制止损检查
        if not stop_loss or stop_loss <= 0:
            return {
                "success": False,
                "error": "必须设置止损价格（stop_loss）"
            }

        # 2. 获取当前价格
        from .query_tools import _get_market_price_internal
        price_result = await _get_market_price_internal(symbol)
        if not price_result.get("success"):
            return {
                "success": False,
                "error": f"无法获取价格: {price_result.get('error')}"
            }

        current_price = price_result["last"]

        # 3. 验证止损合理性（空仓止损必须高于当前价）
        if stop_loss <= current_price:
            return {
                "success": False,
                "error": f"空仓止损价格({stop_loss})必须高于当前价格({current_price})"
            }

        # 4. 验证止盈合理性（如果设置）
        if take_profit and take_profit >= current_price:
            return {
                "success": False,
                "error": f"空仓止盈价格({take_profit})必须低于当前价格({current_price})"
            }

        # 5. 计算风险回报比
        if take_profit:
            from .analysis_tools import _calculate_risk_reward_ratio_internal
            rr_result = _calculate_risk_reward_ratio_internal(current_price, stop_loss, take_profit)
            if rr_result.get("success") and rr_result["ratio"] < 2.0:
                return {
                    "success": False,
                    "error": f"风险回报比({rr_result['ratio']:.2f})过低，建议 >= 2"
                }

        # 6. 执行交易
        executor = get_trading_executor()

        signal = {
            "symbol": symbol,
            "action": "open_short",
            "amount": amount,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        results = await executor.execute_signals([signal], [])

        if results and len(results) > 0:
            result = results[0]
            if result.get("success"):
                return {
                    "success": True,
                    "order_id": result.get("result", {}).get("order_id"),
                    "symbol": symbol,
                    "side": "SHORT",
                    "amount": amount,
                    "leverage": leverage,
                    "filled_price": result.get("result", {}).get("price"),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "message": "空仓开仓成功"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "开仓失败")
                }

        return {
            "success": False,
            "error": "执行交易时无返回结果"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"开仓失败: {str(e)}"
        }


@tool
async def close_position(symbol: str) -> Dict:
    """平仓

    Args:
        symbol: 交易对

    Returns:
        Dict: 平仓结果
            - success: 是否成功
            - symbol: 交易对
            - realized_pnl: 实现盈亏
            - close_price: 平仓价格

    示例:
        result = await close_position("BTC/USDC:USDC")
        # {"success": True, "realized_pnl": 150.0, "close_price": 65500}

    注意:
        - 会平掉该交易对的所有持仓
        - 无法撤销，请确认后操作
    """
    try:
        executor = get_trading_executor()

        # 准备平仓信号
        signal = {
            "symbol": symbol,
            "action": "close_position",
        }

        results = await executor.execute_signals([signal], [])

        if results and len(results) > 0:
            result = results[0]
            if result.get("success"):
                return {
                    "success": True,
                    "symbol": symbol,
                    "realized_pnl": result.get("result", {}).get("pnl", 0),
                    "close_price": result.get("result", {}).get("price"),
                    "message": "平仓成功"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "平仓失败")
                }

        return {
            "success": False,
            "error": "执行平仓时无返回结果"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"平仓失败: {str(e)}"
        }


@tool
async def set_stop_loss(
    symbol: str,
    stop_loss_price: float
) -> Dict:
    """设置止损

    Args:
        symbol: 交易对
        stop_loss_price: 止损价格

    Returns:
        Dict: 设置结果

    注意:
        - 会为该交易对的持仓设置止损
        - 多仓止损必须低于当前价格
        - 空仓止损必须高于当前价格
    """
    try:
        from .query_tools import _get_current_positions_internal, _get_market_price_internal

        # 1. 检查是否有持仓
        positions_result = await _get_current_positions_internal(symbol)
        if not positions_result.get("success"):
            return positions_result

        positions = positions_result.get("positions", [])
        if not positions:
            return {
                "success": False,
                "error": f"没有找到 {symbol} 的持仓"
            }

        position = positions[0]
        side = position["side"]

        # 2. 获取当前价格
        price_result = await _get_market_price_internal(symbol)
        if not price_result.get("success"):
            return price_result

        current_price = price_result["last"]

        # 3. 验证止损合理性
        if side == "LONG" and stop_loss_price >= current_price:
            return {
                "success": False,
                "error": f"多仓止损({stop_loss_price})必须低于当前价({current_price})"
            }
        elif side == "SHORT" and stop_loss_price <= current_price:
            return {
                "success": False,
                "error": f"空仓止损({stop_loss_price})必须高于当前价({current_price})"
            }

        # 4. 调用 adapter 修改止损
        adapter = get_adapter()
        result = await adapter.modify_stop_loss_take_profit(
            position=position,
            stop_loss=Decimal(str(stop_loss_price)),
            take_profit=None  # 只修改止损，保持止盈不变
        )

        if result.status.value == "success":
            return {
                "success": True,
                "symbol": symbol,
                "stop_loss": stop_loss_price,
                "message": f"止损设置成功: {stop_loss_price}"
            }
        else:
            return {
                "success": False,
                "error": result.error or result.message
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"设置止损失败: {str(e)}"
        }


@tool
async def set_take_profit(
    symbol: str,
    take_profit_price: float
) -> Dict:
    """设置止盈

    Args:
        symbol: 交易对
        take_profit_price: 止盈价格

    Returns:
        Dict: 设置结果

    注意:
        - 多仓止盈必须高于当前价格
        - 空仓止盈必须低于当前价格
    """
    try:
        from .query_tools import _get_current_positions_internal, _get_market_price_internal

        # 1. 检查持仓
        positions_result = await _get_current_positions_internal(symbol)
        if not positions_result.get("success"):
            return positions_result

        positions = positions_result.get("positions", [])
        if not positions:
            return {
                "success": False,
                "error": f"没有找到 {symbol} 的持仓"
            }

        position = positions[0]
        side = position["side"]

        # 2. 获取当前价格
        price_result = await _get_market_price_internal(symbol)
        if not price_result.get("success"):
            return price_result

        current_price = price_result["last"]

        # 3. 验证止盈合理性
        if side == "LONG" and take_profit_price <= current_price:
            return {
                "success": False,
                "error": f"多仓止盈({take_profit_price})必须高于当前价({current_price})"
            }
        elif side == "SHORT" and take_profit_price >= current_price:
            return {
                "success": False,
                "error": f"空仓止盈({take_profit_price})必须低于当前价({current_price})"
            }

        # 4. 调用 adapter 修改止盈
        adapter = get_adapter()
        result = await adapter.modify_stop_loss_take_profit(
            position=position,
            stop_loss=None,  # 保持止损不变
            take_profit=Decimal(str(take_profit_price))
        )

        if result.status.value == "success":
            return {
                "success": True,
                "symbol": symbol,
                "take_profit": take_profit_price,
                "message": f"止盈设置成功: {take_profit_price}"
            }
        else:
            return {
                "success": False,
                "error": result.error or result.message
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"设置止盈失败: {str(e)}"
        }
