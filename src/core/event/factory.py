"""
事件工厂

参考 ValueCell 的 ResponseFactory，提供统一的事件创建方法
"""
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime

from .types import TradingEvent, TradingEventType, SystemEvent, SystemEventType


class EventFactory:
    """
    事件工厂

    提供创建各类事件的工厂方法，确保事件格式一致
    """

    # ==================== 订单事件 ====================

    @staticmethod
    def order_submitted(
        symbol: str,
        order_id: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "market",
        **kwargs,
    ) -> TradingEvent:
        """订单已提交"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_SUBMITTED,
            symbol=symbol,
            order_id=order_id,
            side=side,
            amount=amount,
            price=price,
            metadata={"order_type": order_type, **kwargs},
        )

    @staticmethod
    def order_accepted(
        symbol: str,
        order_id: str,
        **kwargs,
    ) -> TradingEvent:
        """订单已接受"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_ACCEPTED,
            symbol=symbol,
            order_id=order_id,
            metadata=kwargs,
        )

    @staticmethod
    def order_rejected(
        symbol: str,
        order_id: str,
        error_message: str,
        error_code: Optional[str] = None,
        **kwargs,
    ) -> TradingEvent:
        """订单被拒绝"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_REJECTED,
            symbol=symbol,
            order_id=order_id,
            error_message=error_message,
            error_code=error_code,
            metadata=kwargs,
        )

    @staticmethod
    def order_filled(
        symbol: str,
        order_id: str,
        side: str,
        filled_amount: Decimal,
        average_price: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """订单已完全成交"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_FILLED,
            symbol=symbol,
            order_id=order_id,
            side=side,
            filled_amount=filled_amount,
            average_price=average_price,
            metadata=kwargs,
        )

    @staticmethod
    def order_partially_filled(
        symbol: str,
        order_id: str,
        filled_amount: Decimal,
        remaining_amount: Decimal,
        average_price: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """订单部分成交"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_PARTIALLY_FILLED,
            symbol=symbol,
            order_id=order_id,
            filled_amount=filled_amount,
            average_price=average_price,
            metadata={"remaining_amount": str(remaining_amount), **kwargs},
        )

    @staticmethod
    def order_cancelled(
        symbol: str,
        order_id: str,
        reason: Optional[str] = None,
        **kwargs,
    ) -> TradingEvent:
        """订单已取消"""
        return TradingEvent(
            event_type=TradingEventType.ORDER_CANCELLED,
            symbol=symbol,
            order_id=order_id,
            metadata={"reason": reason, **kwargs} if reason else kwargs,
        )

    # ==================== 持仓事件 ====================

    @staticmethod
    def position_opened(
        symbol: str,
        position_id: str,
        side: str,
        amount: Decimal,
        entry_price: Decimal,
        leverage: int = 1,
        **kwargs,
    ) -> TradingEvent:
        """持仓已开启"""
        return TradingEvent(
            event_type=TradingEventType.POSITION_OPENED,
            symbol=symbol,
            position_id=position_id,
            side=side,
            amount=amount,
            entry_price=entry_price,
            leverage=leverage,
            metadata=kwargs,
        )

    @staticmethod
    def position_closed(
        symbol: str,
        position_id: str,
        side: str,
        amount: Decimal,
        entry_price: Decimal,
        exit_price: Decimal,
        realized_pnl: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """持仓已关闭"""
        return TradingEvent(
            event_type=TradingEventType.POSITION_CLOSED,
            symbol=symbol,
            position_id=position_id,
            side=side,
            amount=amount,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            metadata=kwargs,
        )

    @staticmethod
    def position_liquidated(
        symbol: str,
        position_id: str,
        side: str,
        amount: Decimal,
        liquidation_price: Decimal,
        realized_pnl: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """持仓被强平"""
        return TradingEvent(
            event_type=TradingEventType.POSITION_LIQUIDATED,
            symbol=symbol,
            position_id=position_id,
            side=side,
            amount=amount,
            exit_price=liquidation_price,
            realized_pnl=realized_pnl,
            metadata=kwargs,
        )

    # ==================== 止盈止损事件 ====================

    @staticmethod
    def stop_loss_set(
        symbol: str,
        position_id: str,
        stop_loss: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """止损已设置"""
        return TradingEvent(
            event_type=TradingEventType.STOP_LOSS_SET,
            symbol=symbol,
            position_id=position_id,
            stop_loss=stop_loss,
            metadata=kwargs,
        )

    @staticmethod
    def stop_loss_triggered(
        symbol: str,
        position_id: str,
        stop_loss: Decimal,
        realized_pnl: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """止损已触发"""
        return TradingEvent(
            event_type=TradingEventType.STOP_LOSS_TRIGGERED,
            symbol=symbol,
            position_id=position_id,
            stop_loss=stop_loss,
            exit_price=stop_loss,
            realized_pnl=realized_pnl,
            metadata=kwargs,
        )

    @staticmethod
    def take_profit_set(
        symbol: str,
        position_id: str,
        take_profit: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """止盈已设置"""
        return TradingEvent(
            event_type=TradingEventType.TAKE_PROFIT_SET,
            symbol=symbol,
            position_id=position_id,
            take_profit=take_profit,
            metadata=kwargs,
        )

    @staticmethod
    def take_profit_triggered(
        symbol: str,
        position_id: str,
        take_profit: Decimal,
        realized_pnl: Decimal,
        **kwargs,
    ) -> TradingEvent:
        """止盈已触发"""
        return TradingEvent(
            event_type=TradingEventType.TAKE_PROFIT_TRIGGERED,
            symbol=symbol,
            position_id=position_id,
            take_profit=take_profit,
            exit_price=take_profit,
            realized_pnl=realized_pnl,
            metadata=kwargs,
        )

    # ==================== 风险事件 ====================

    @staticmethod
    def risk_limit_breached(
        symbol: str,
        limit_type: str,
        current_value: float,
        limit_value: float,
        **kwargs,
    ) -> TradingEvent:
        """风险限制被突破"""
        return TradingEvent(
            event_type=TradingEventType.RISK_LIMIT_BREACHED,
            symbol=symbol,
            metadata={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                **kwargs,
            },
        )

    @staticmethod
    def margin_warning(
        symbol: str,
        margin_ratio: float,
        warning_threshold: float,
        **kwargs,
    ) -> TradingEvent:
        """保证金警告"""
        return TradingEvent(
            event_type=TradingEventType.MARGIN_WARNING,
            symbol=symbol,
            margin_ratio=margin_ratio,
            metadata={
                "warning_threshold": warning_threshold,
                **kwargs,
            },
        )

    # ==================== 系统事件 ====================

    @staticmethod
    def system_started(
        component: str = "system",
        message: Optional[str] = None,
        **kwargs,
    ) -> SystemEvent:
        """系统已启动"""
        return SystemEvent(
            event_type=SystemEventType.SYSTEM_STARTED,
            component=component,
            message=message or f"{component} started",
            details=kwargs,
        )

    @staticmethod
    def system_stopped(
        component: str = "system",
        message: Optional[str] = None,
        **kwargs,
    ) -> SystemEvent:
        """系统已停止"""
        return SystemEvent(
            event_type=SystemEventType.SYSTEM_STOPPED,
            component=component,
            message=message or f"{component} stopped",
            details=kwargs,
        )

    @staticmethod
    def system_error(
        component: str,
        error_message: str,
        error_traceback: Optional[str] = None,
        **kwargs,
    ) -> SystemEvent:
        """系统错误"""
        return SystemEvent(
            event_type=SystemEventType.SYSTEM_ERROR,
            component=component,
            error_message=error_message,
            error_traceback=error_traceback,
            details=kwargs,
        )

    @staticmethod
    def exchange_connected(
        exchange_name: str,
        **kwargs,
    ) -> SystemEvent:
        """交易所已连接"""
        return SystemEvent(
            event_type=SystemEventType.EXCHANGE_CONNECTED,
            component="exchange",
            message=f"Connected to {exchange_name}",
            details={"exchange": exchange_name, **kwargs},
        )

    @staticmethod
    def exchange_disconnected(
        exchange_name: str,
        reason: Optional[str] = None,
        **kwargs,
    ) -> SystemEvent:
        """交易所断开连接"""
        return SystemEvent(
            event_type=SystemEventType.EXCHANGE_DISCONNECTED,
            component="exchange",
            message=f"Disconnected from {exchange_name}",
            details={"exchange": exchange_name, "reason": reason, **kwargs},
        )

    @staticmethod
    def decision_made(
        decision: str,
        symbol: str,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        **kwargs,
    ) -> SystemEvent:
        """决策已做出"""
        return SystemEvent(
            event_type=SystemEventType.DECISION_MADE,
            component="decision",
            message=f"Decision: {decision} for {symbol}",
            details={
                "decision": decision,
                "symbol": symbol,
                "confidence": confidence,
                "reasoning": reasoning,
                **kwargs,
            },
        )

    @staticmethod
    def reflection_completed(
        lessons_learned: list,
        trade_count: int,
        total_pnl: Decimal,
        **kwargs,
    ) -> SystemEvent:
        """反思已完成"""
        return SystemEvent(
            event_type=SystemEventType.REFLECTION_COMPLETED,
            component="learning",
            message=f"Reflection completed: {len(lessons_learned)} lessons from {trade_count} trades",
            details={
                "lessons_learned": lessons_learned,
                "trade_count": trade_count,
                "total_pnl": str(total_pnl),
                **kwargs,
            },
        )
