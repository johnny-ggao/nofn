"""Rolling digest builder implementation."""

from typing import Dict, List, Optional
import math

from ..models import (
    HistoryRecord,
    InstrumentRef,
    TradeDigest,
    TradeDigestEntry,
    get_current_timestamp_ms,
)
from .interfaces import BaseDigestBuilder


class RollingDigestBuilder(BaseDigestBuilder):
    """Builds trade digests with rolling statistics.

    Computes per-instrument statistics and overall Sharpe ratio.
    """

    def __init__(self, lookback_days: int = 7) -> None:
        """Initialize digest builder.

        Args:
            lookback_days: Number of days to consider for statistics
        """
        self._lookback_ms = lookback_days * 24 * 60 * 60 * 1000

    def build(self, records: List[HistoryRecord]) -> TradeDigest:
        """Build a digest from history records."""
        ts = get_current_timestamp_ms()
        cutoff_ts = ts - self._lookback_ms

        # Extract execution records with trades
        trades_data: List[Dict] = []
        for record in records:
            if record.kind != "execution":
                continue
            if record.ts < cutoff_ts:
                continue
            payload_trades = record.payload.get("trades", [])
            if payload_trades:
                trades_data.extend(payload_trades)

        # Build per-instrument stats
        by_instrument: Dict[str, TradeDigestEntry] = {}
        returns: List[float] = []

        # Group by symbol
        symbol_trades: Dict[str, List[Dict]] = {}
        for trade in trades_data:
            inst = trade.get("instrument", {})
            symbol = inst.get("symbol") if isinstance(inst, dict) else str(inst)
            if not symbol:
                continue
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        for symbol, symbol_trade_list in symbol_trades.items():
            trade_count = len(symbol_trade_list)
            realized_pnl = 0.0
            wins = 0
            holding_times: List[int] = []
            entry_prices: List[float] = []
            last_ts: Optional[int] = None
            max_dd = 0.0
            running_pnl = 0.0
            peak_pnl = 0.0

            for t in symbol_trade_list:
                pnl = float(t.get("realized_pnl") or 0.0)
                realized_pnl += pnl
                running_pnl += pnl

                if pnl > 0:
                    wins += 1

                holding_ms = t.get("holding_ms")
                if holding_ms:
                    holding_times.append(int(holding_ms))

                entry_px = t.get("entry_price")
                if entry_px:
                    entry_prices.append(float(entry_px))

                trade_ts = t.get("trade_ts") or t.get("exit_ts") or t.get("entry_ts")
                if trade_ts:
                    if last_ts is None or int(trade_ts) > last_ts:
                        last_ts = int(trade_ts)

                # Track drawdown
                if running_pnl > peak_pnl:
                    peak_pnl = running_pnl
                dd = peak_pnl - running_pnl
                if dd > max_dd:
                    max_dd = dd

                # Track returns for Sharpe
                if pnl != 0:
                    returns.append(pnl)

            win_rate = (wins / trade_count) if trade_count > 0 else None
            avg_holding = (
                int(sum(holding_times) / len(holding_times)) if holding_times else None
            )
            avg_entry = (
                sum(entry_prices) / len(entry_prices) if entry_prices else None
            )

            by_instrument[symbol] = TradeDigestEntry(
                instrument=InstrumentRef(symbol=symbol),
                trade_count=trade_count,
                realized_pnl=realized_pnl,
                win_rate=win_rate,
                avg_holding_ms=avg_holding,
                last_trade_ts=last_ts,
                avg_entry_price=avg_entry,
                max_drawdown=max_dd if max_dd > 0 else None,
            )

        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(returns)

        return TradeDigest(
            ts=ts,
            by_instrument=by_instrument,
            sharpe_ratio=sharpe_ratio,
        )

    def _calculate_sharpe(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0,
        annualization_factor: float = 365.0,
    ) -> Optional[float]:
        """Calculate Sharpe ratio from returns.

        Args:
            returns: List of P&L values
            risk_free_rate: Risk-free rate (daily)
            annualization_factor: Factor for annualization

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(returns) < 2:
            return None

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return None

        # Daily Sharpe ratio (assuming returns are per-trade, approximate daily)
        sharpe = (mean_return - risk_free_rate) / std_dev

        # Annualize (optional, but useful for interpretation)
        # sharpe_annualized = sharpe * math.sqrt(annualization_factor)

        return round(sharpe, 3)
