"""
Trading Executor - 交易执行模块

负责执行交易信号，包括开仓、平仓、修改止盈止损等操作
"""
from typing import Dict, List, Any
from decimal import Decimal
from datetime import datetime
from termcolor import cprint

from ...adapters.base import BaseExchangeAdapter
from ...models.enums import PositionSide, OrderType, ExecutionStatus


class TradingExecutor:
    """
    交易执行器

    负责将交易信号转换为实际的交易操作
    """

    def __init__(self, adapter: BaseExchangeAdapter):
        """
        初始化交易执行器

        Args:
            adapter: 交易所适配器
        """
        self.adapter = adapter
        self.execution_history = []

    async def execute_signals(
        self,
        signals: List[Dict[str, Any]],
        positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量执行交易信号

        Args:
            signals: 交易信号列表
            positions: 当前持仓列表

        Returns:
            执行结果列表
        """
        results = []

        for signal in signals:
            try:
                result = await self.execute_single_signal(signal, positions)
                results.append(result)

                # 记录到历史
                self.execution_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "signal": signal,
                    "result": result,
                })

            except Exception as e:
                error_result = {
                    "signal": signal,
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(error_result)
                cprint(f"❌ Signal execution failed: {e}", "red")

        return results

    async def execute_single_signal(
        self,
        signal: Dict[str, Any],
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        执行单个交易信号

        Args:
            signal: 交易信号
            positions: 当前持仓列表

        Returns:
            执行结果
        """
        action = signal.get("action")

        if action == "open_long":
            return await self._open_long(signal)
        elif action == "open_short":
            return await self._open_short(signal)
        elif action == "close_position":
            return await self._close_position(signal, positions)
        elif action == "modify_sl_tp":
            return await self._modify_sl_tp(signal, positions)
        else:
            return {
                "signal": signal,
                "error": f"Unknown action: {action}",
                "success": False,
                "timestamp": datetime.now().isoformat(),
            }

    async def _open_long(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """开多仓"""
        try:
            result = await self.adapter.open_position(
                symbol=signal["symbol"],
                side=PositionSide.LONG,
                amount=Decimal(str(signal["amount"])),
                order_type=OrderType.MARKET,
                leverage=signal.get("leverage", 1),
                stop_loss=Decimal(str(signal["stop_loss"])) if signal.get("stop_loss") else None,
                take_profit=Decimal(str(signal["take_profit"])) if signal.get("take_profit") else None,
            )

            # 检查执行结果
            is_success = result.status == ExecutionStatus.SUCCESS

            if is_success:
                cprint(f"✓ Opened LONG position: {signal['symbol']} x{signal['amount']}", "green")
            else:
                cprint(f"❌ Failed to open LONG position: {signal['symbol']}", "red")
                cprint(f"   Status: {result.status}, Message: {result.message}", "red")
                if result.error:
                    cprint(f"   Error Details: {result.error}", "red")

            return {
                "signal": signal,
                "result": result.model_dump(),
                "success": is_success,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            cprint(f"❌ Failed to open LONG: {e}", "red")
            raise

    async def _open_short(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """开空仓"""
        try:
            result = await self.adapter.open_position(
                symbol=signal["symbol"],
                side=PositionSide.SHORT,
                amount=Decimal(str(signal["amount"])),
                order_type=OrderType.MARKET,
                leverage=signal.get("leverage", 1),
                stop_loss=Decimal(str(signal["stop_loss"])) if signal.get("stop_loss") else None,
                take_profit=Decimal(str(signal["take_profit"])) if signal.get("take_profit") else None,
            )

            # 检查执行结果
            is_success = result.status == ExecutionStatus.SUCCESS

            if is_success:
                cprint(f"✓ Opened SHORT position: {signal['symbol']} x{signal['amount']}", "green")
            else:
                cprint(f"❌ Failed to open SHORT position: {signal['symbol']}", "red")
                cprint(f"   Status: {result.status}, Message: {result.message}", "red")
                if result.error:
                    cprint(f"   Error Details: {result.error}", "red")

            return {
                "signal": signal,
                "result": result.model_dump(),
                "success": is_success,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            cprint(f"❌ Failed to open SHORT: {e}", "red")
            raise

    async def _close_position(
        self,
        signal: Dict[str, Any],
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """平仓"""
        try:
            # 找到对应的持仓
            target_position = None
            for pos in positions:
                if pos["symbol"] == signal["symbol"]:
                    target_position = pos
                    break

            if not target_position:
                return {
                    "signal": signal,
                    "error": "No matching position found",
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            result = await self.adapter.close_position(
                symbol=target_position["symbol"],
                position_id=target_position["position_id"],
            )

            # 检查执行结果
            is_success = result.status == ExecutionStatus.SUCCESS

            if is_success:
                cprint(f"✓ Closed position: {signal['symbol']}", "green")
            else:
                cprint(f"❌ Failed to close position: {signal['symbol']}", "red")
                cprint(f"   Status: {result.status}, Message: {result.message}", "red")
                if result.error:
                    cprint(f"   Error Details: {result.error}", "red")

            return {
                "signal": signal,
                "result": result.model_dump(),
                "success": is_success,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            cprint(f"❌ Failed to close position: {e}", "red")
            raise

    async def _modify_sl_tp(
        self,
        signal: Dict[str, Any],
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """修改止盈止损"""
        try:
            # 找到对应的持仓
            target_position = None
            for pos in positions:
                if pos["symbol"] == signal["symbol"]:
                    target_position = pos
                    break

            if not target_position:
                return {
                    "signal": signal,
                    "error": "No matching position found",
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                }

            result = await self.adapter.modify_stop_loss_take_profit(
                target_position,
                stop_loss=Decimal(str(signal["stop_loss"])) if signal.get("stop_loss") else None,
                take_profit=Decimal(str(signal["take_profit"])) if signal.get("take_profit") else None,
            )

            # 检查执行结果
            is_success = result.status == ExecutionStatus.SUCCESS

            if is_success:
                cprint(f"✓ Modified SL/TP: {signal['symbol']}", "green")
            else:
                cprint(f"❌ Failed to modify SL/TP: {signal['symbol']}", "red")
                cprint(f"   Status: {result.status}, Message: {result.message}", "red")
                if result.error:
                    cprint(f"   Error Details: {result.error}", "red")

            return {
                "signal": signal,
                "result": result.model_dump(),
                "success": is_success,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            cprint(f"❌ Failed to modify SL/TP: {e}", "red")
            raise

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history.copy()

    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()
