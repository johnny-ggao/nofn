"""
上下文和记忆管理模块
提供交易上下文管理功能
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from termcolor import cprint


class TradingContextManager:
    """
    交易上下文管理器

    功能：
    1. 管理决策历史（短期 + 长期摘要）
    2. 管理交易历史（滑动窗口）
    3. 管理性能指标（摘要）
    4. 自动清理过期数据
    """

    def __init__(
        self,
        max_decision_history: int = 10,
        max_trade_history: int = 50,
        performance_window_hours: int = 24,
    ):
        """
        初始化上下文管理器

        Args:
            max_decision_history: 最大决策历史条数（短期记忆）
            max_trade_history: 最大交易历史条数（滑动窗口）
            performance_window_hours: 性能统计窗口（小时）
        """
        self.max_decision_history = max_decision_history
        self.max_trade_history = max_trade_history
        self.performance_window_hours = performance_window_hours

        # 短期记忆：最近的决策
        self.recent_decisions: List[Dict] = []

        # 短期记忆：最近的交易
        self.recent_trades: List[Dict] = []

        # 长期记忆：摘要
        self.performance_summary: Dict[str, Any] = {}

    def add_decision(self, decision: Dict):
        """
        添加决策记录（自动管理滑动窗口）

        Args:
            decision: 决策记录
        """
        self.recent_decisions.append(decision)

        # 滑动窗口：只保留最近 N 条
        if len(self.recent_decisions) > self.max_decision_history:
            removed = self.recent_decisions.pop(0)
            cprint(
                f"  🗑️ 移除旧决策记录: {removed.get('timestamp', 'N/A')}",
                "yellow"
            )

    def add_trade(self, trade: Dict):
        """
        添加交易记录（自动管理滑动窗口 + 去重）

        Args:
            trade: 交易记录
        """
        # 去重：检查 trade_id
        trade_id = trade.get("trade_id")
        if trade_id:
            existing_ids = {t.get("trade_id") for t in self.recent_trades}
            if trade_id in existing_ids:
                cprint(
                    f"  ⚠️ 重复交易记录（已跳过）: {trade_id}",
                    "yellow"
                )
                return

        self.recent_trades.append(trade)

        # 滑动窗口：只保留最近 N 条
        if len(self.recent_trades) > self.max_trade_history:
            removed = self.recent_trades.pop(0)
            cprint(
                f"  🗑️ 移除旧交易记录: {removed.get('trade_id', 'N/A')}",
                "yellow"
            )

    def update_performance_summary(self, metrics: Dict[str, Any]):
        """
        更新性能摘要（只保留关键指标）

        Args:
            metrics: 性能指标
        """
        # 只保留摘要信息，不保留完整历史
        self.performance_summary = {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "total_trades": metrics.get("total_trades", 0),
            "win_rate": metrics.get("win_rate", 0),
            "total_pnl": metrics.get("total_pnl", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "trades_last_hour": metrics.get("trades_last_hour", 0),
            "trades_last_24h": metrics.get("trades_last_24h", 0),
            "pnl_last_24h": metrics.get("pnl_last_24h", 0),
            "over_trading_risk": metrics.get("over_trading_risk", False),
            "consecutive_losses": metrics.get("consecutive_losses", 0),
            "last_updated": datetime.now().isoformat(),
        }

    def get_context_summary(self) -> Dict[str, Any]:
        """
        获取上下文摘要（用于 LLM 输入）

        Returns:
            上下文摘要字典
        """
        return {
            "recent_decisions_count": len(self.recent_decisions),
            "recent_trades_count": len(self.recent_trades),
            "recent_decisions": self.recent_decisions[-5:],  # 最近5条
            "recent_trades": self.recent_trades[-10:],  # 最近10条
            "performance": self.performance_summary,
        }

    def cleanup_old_data(self):
        """
        清理过期数据（基于时间窗口）
        """
        now = datetime.now()
        cutoff_time = now - timedelta(hours=self.performance_window_hours)

        # 清理旧交易
        original_count = len(self.recent_trades)
        self.recent_trades = [
            t for t in self.recent_trades
            if self._parse_timestamp(t.get("timestamp")) > cutoff_time
        ]

        if len(self.recent_trades) < original_count:
            removed_count = original_count - len(self.recent_trades)
            cprint(
                f"  🧹 清理了 {removed_count} 条过期交易记录（>{self.performance_window_hours}小时）",
                "cyan"
            )

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """解析时间戳字符串"""
        if not timestamp_str:
            return datetime.min

        try:
            return datetime.fromisoformat(timestamp_str)
        except:
            return datetime.min

    def get_memory_stats(self) -> Dict[str, int]:
        """
        获取内存使用统计

        Returns:
            内存统计字典
        """
        return {
            "decisions_count": len(self.recent_decisions),
            "trades_count": len(self.recent_trades),
            "decisions_max": self.max_decision_history,
            "trades_max": self.max_trade_history,
        }

    def get_recent_decisions(self) -> List[Dict]:
        """获取最近的决策记录"""
        return self.recent_decisions

    def get_recent_trades(self) -> List[Dict]:
        """获取最近的交易记录"""
        return self.recent_trades

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_summary


# ==================== 已弃用：LangChainMemoryAdapter ====================
# 注意：新的 LangChain 实现不再使用此类
# 原因：需要额外的 langchain-community 依赖
# 新实现使用简单的消息列表管理，更易理解和学习
#
# 如果需要高级的 Memory 功能（自动摘要等），可以：
# 1. 安装 langchain-community: uv add langchain-community
# 2. 取消下面代码的注释
# ========================================================================


# ==================== Usage Examples ====================
# See examples below for how to use TradingContextManager
#
# Example 1: Create context manager
# context_manager = TradingContextManager(
#     max_decision_history=10,
#     max_trade_history=50,
#     performance_window_hours=24,
# )
#
# Example 2: Add decision record
# context_manager.add_decision({
#     "timestamp": datetime.now().isoformat(),
#     "signals": [...],
#     "risk_assessment": {...},
# })
#
# Example 3: Add trade record (auto deduplication)
# context_manager.add_trade({
#     "trade_id": "trade_001",
#     "timestamp": datetime.now().isoformat(),
#     "symbol": "BTC/USDC:USDC",
#     "side": "LONG",
#     "pnl": 0.0,
# })
#
# Example 4: Get context summary for LLM
# summary = context_manager.get_context_summary()
#
# Example 5: Cleanup old data
# context_manager.cleanup_old_data()
