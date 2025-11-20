"""
Performance Calculator - 性能计算器

计算交易系统的各种性能指标：
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Max Drawdown)
- 胜率 (Win Rate)
- 盈亏比 (Profit Factor)
- 风险价值 (VaR)
等
"""
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import math


class PerformanceCalculator:
    """
    性能计算器

    基于交易历史计算各种性能指标
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化性能计算器

        Args:
            risk_free_rate: 无风险利率（年化，默认 2%）
        """
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        periods_per_year: int = 252
    ) -> float:
        """
        计算夏普比率

        Args:
            returns: 收益率列表
            periods_per_year: 每年的交易周期数（日频=252，小时频=252*24）

        Returns:
            夏普比率
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)

        # 平均收益率
        mean_return = np.mean(returns_array)

        # 收益率标准差
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return 0.0

        # 年化夏普比率
        sharpe = (mean_return - self.risk_free_rate / periods_per_year) / std_return
        sharpe_annualized = sharpe * np.sqrt(periods_per_year)

        return float(sharpe_annualized)

    def calculate_max_drawdown(
        self,
        equity_curve: List[float]
    ) -> Dict[str, float]:
        """
        计算最大回撤

        Args:
            equity_curve: 权益曲线（账户余额历史）

        Returns:
            {
                "max_drawdown": 最大回撤金额,
                "max_drawdown_pct": 最大回撤百分比,
                "peak": 峰值,
                "trough": 谷值
            }
        """
        if not equity_curve or len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "peak": 0.0,
                "trough": 0.0
            }

        equity_array = np.array(equity_curve)

        # 计算累计最大值
        running_max = np.maximum.accumulate(equity_array)

        # 计算回撤
        drawdown = running_max - equity_array

        # 最大回撤
        max_dd_idx = np.argmax(drawdown)
        max_drawdown = float(drawdown[max_dd_idx])

        # 峰值和谷值
        peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
        peak = float(equity_array[peak_idx])
        trough = float(equity_array[max_dd_idx])

        # 最大回撤百分比
        max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else 0.0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "peak": peak,
            "trough": trough
        }

    def calculate_win_rate(
        self,
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算胜率相关指标

        Args:
            trades: 交易列表

        Returns:
            {
                "total_trades": 总交易数,
                "winning_trades": 盈利交易数,
                "losing_trades": 亏损交易数,
                "win_rate": 胜率 (%),
                "avg_win": 平均盈利,
                "avg_loss": 平均亏损,
                "profit_factor": 盈亏比
            }
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }

        closed_trades = [t for t in trades if t.get("status") == "closed"]
        total_trades = len(closed_trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }

        # 分类盈亏交易
        winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in closed_trades if t.get("pnl", 0) < 0]

        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        # 计算平均盈亏
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0.0

        # 总盈利和总亏损
        total_profit = sum([t["pnl"] for t in winning_trades])
        total_loss = abs(sum([t["pnl"] for t in losing_trades]))

        # 盈亏比 (Profit Factor)
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_count,
            "losing_trades": losing_count,
            "win_rate": (winning_count / total_trades * 100) if total_trades > 0 else 0.0,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor)
        }

    def calculate_sortino_ratio(
        self,
        returns: List[float],
        periods_per_year: int = 252
    ) -> float:
        """
        计算索提诺比率（只考虑下行波动）

        Args:
            returns: 收益率列表
            periods_per_year: 每年的交易周期数

        Returns:
            索提诺比率
        """
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)

        # 平均收益率
        mean_return = np.mean(returns_array)

        # 下行偏差（只考虑负收益）
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return float('inf')  # 没有下行风险

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        # 年化索提诺比率
        sortino = (mean_return - self.risk_free_rate / periods_per_year) / downside_std
        sortino_annualized = sortino * np.sqrt(periods_per_year)

        return float(sortino_annualized)

    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """
        计算风险价值 (Value at Risk)

        Args:
            returns: 收益率列表
            confidence_level: 置信水平（默认 95%）

        Returns:
            VaR 值
        """
        if not returns or len(returns) < 10:
            return 0.0

        returns_array = np.array(returns)

        # 使用历史模拟法
        var = np.percentile(returns_array, (1 - confidence_level) * 100)

        return float(var)

    def calculate_cvar(
        self,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """
        计算条件风险价值 (Conditional VaR / Expected Shortfall)

        Args:
            returns: 收益率列表
            confidence_level: 置信水平（默认 95%）

        Returns:
            CVaR 值
        """
        if not returns or len(returns) < 10:
            return 0.0

        returns_array = np.array(returns)
        var = self.calculate_var(returns, confidence_level)

        # CVaR 是超过 VaR 的平均损失
        cvar = np.mean(returns_array[returns_array <= var])

        return float(cvar)

    def calculate_calmar_ratio(
        self,
        returns: List[float],
        equity_curve: List[float],
        periods_per_year: int = 252
    ) -> float:
        """
        计算卡玛比率 (Calmar Ratio)
        = 年化收益率 / 最大回撤

        Args:
            returns: 收益率列表
            equity_curve: 权益曲线
            periods_per_year: 每年的交易周期数

        Returns:
            卡玛比率
        """
        if not returns or not equity_curve:
            return 0.0

        # 年化收益率
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if equity_curve[0] > 0 else 0
        periods = len(returns)
        years = periods / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        dd_info = self.calculate_max_drawdown(equity_curve)
        max_dd_pct = dd_info["max_drawdown_pct"] / 100

        if max_dd_pct == 0:
            return float('inf')

        calmar = annual_return / max_dd_pct

        return float(calmar)

    def calculate_comprehensive_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        periods_per_year: int = 252
    ) -> Dict[str, Any]:
        """
        计算综合性能指标

        Args:
            trades: 交易历史
            equity_curve: 权益曲线
            periods_per_year: 每年的交易周期数

        Returns:
            综合性能指标字典
        """
        # 计算收益率序列
        returns = []
        if len(equity_curve) > 1:
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)

        # 胜率相关
        win_rate_metrics = self.calculate_win_rate(trades)

        # 风险调整收益
        sharpe = self.calculate_sharpe_ratio(returns, periods_per_year)
        sortino = self.calculate_sortino_ratio(returns, periods_per_year)

        # 回撤
        dd_metrics = self.calculate_max_drawdown(equity_curve)

        # 风险价值
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)

        # 卡玛比率
        calmar = self.calculate_calmar_ratio(returns, equity_curve, periods_per_year)

        # 总收益
        total_pnl = sum([t.get("pnl", 0) for t in trades if t.get("status") == "closed"])
        total_pnl_pct = ((equity_curve[-1] / equity_curve[0] - 1) * 100) if len(equity_curve) > 0 and equity_curve[0] > 0 else 0

        return {
            # 基本统计
            "total_trades": win_rate_metrics["total_trades"],
            "winning_trades": win_rate_metrics["winning_trades"],
            "losing_trades": win_rate_metrics["losing_trades"],
            "win_rate": win_rate_metrics["win_rate"],

            # 盈亏
            "total_pnl": float(total_pnl),
            "total_pnl_pct": float(total_pnl_pct),
            "avg_win": win_rate_metrics["avg_win"],
            "avg_loss": win_rate_metrics["avg_loss"],
            "profit_factor": win_rate_metrics["profit_factor"],

            # 风险调整收益
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,

            # 回撤
            "max_drawdown": dd_metrics["max_drawdown"],
            "max_drawdown_pct": dd_metrics["max_drawdown_pct"],

            # 风险
            "var_95": var_95,
            "cvar_95": cvar_95,

            # 账户
            "starting_balance": float(equity_curve[0]) if equity_curve else 0,
            "ending_balance": float(equity_curve[-1]) if equity_curve else 0,
            "peak_balance": float(max(equity_curve)) if equity_curve else 0,
        }
