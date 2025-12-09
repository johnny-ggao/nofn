"""In-memory portfolio service implementation."""

from typing import Dict, List, Optional

from ..models import (
    Constraints,
    FeatureVector,
    MarketType,
    PortfolioView,
    PositionSnapshot,
    TradeHistoryEntry,
    TradeSide,
    TradeType,
    TradingMode,
    get_current_timestamp_ms,
)
from .interfaces import BasePortfolioService


class InMemoryPortfolioService(BasePortfolioService):
    """In-memory portfolio service for tracking positions and balances.

    Supports both virtual (paper) and live trading modes.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_mode: TradingMode = TradingMode.VIRTUAL,
        market_type: MarketType = MarketType.SWAP,
        constraints: Optional[Constraints] = None,
        strategy_id: Optional[str] = None,
    ) -> None:
        """Initialize portfolio service.

        Args:
            initial_capital: Starting capital in USD
            trading_mode: Virtual or live trading
            market_type: Spot, future, or swap
            constraints: Trading constraints
            strategy_id: Optional strategy ID
        """
        self._initial_capital = initial_capital
        self._cash_balance = initial_capital
        self._trading_mode = trading_mode
        self._market_type = market_type
        self._constraints = constraints
        self._strategy_id = strategy_id

        # Position tracking: symbol -> PositionSnapshot
        self._positions: Dict[str, PositionSnapshot] = {}

        # Realized P&L tracking
        self._realized_pnl = 0.0

    def get_view(self) -> PortfolioView:
        """Get the current portfolio view."""
        ts = get_current_timestamp_ms()

        # Calculate totals
        total_unrealized_pnl = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0

        for pos in self._positions.values():
            if pos.unrealized_pnl is not None:
                total_unrealized_pnl += pos.unrealized_pnl
            if pos.notional is not None:
                gross_exposure += abs(pos.notional)
                net_exposure += pos.notional

        # Calculate total value (equity)
        total_value = self._cash_balance + total_unrealized_pnl

        # Calculate buying power
        if self._market_type == MarketType.SPOT:
            buying_power = max(0.0, self._cash_balance)
        else:
            # Derivatives: buying power = equity * max_leverage - gross_exposure
            max_lev = (self._constraints.max_leverage or 1.0) if self._constraints else 1.0
            buying_power = max(0.0, total_value * max_lev - gross_exposure)

        # Free cash approximation
        free_cash = max(0.0, self._cash_balance)

        return PortfolioView(
            strategy_id=self._strategy_id,
            ts=ts,
            account_balance=self._cash_balance,
            positions=dict(self._positions),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            constraints=self._constraints,
            total_value=total_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=self._realized_pnl,
            buying_power=buying_power,
            free_cash=free_cash,
        )

    def apply_trades(
        self,
        trades: List[TradeHistoryEntry],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> None:
        """Apply trades to update portfolio state."""
        for trade in trades:
            self._apply_single_trade(trade)

        # Update prices if market features provided
        if market_features:
            self.update_prices(market_features)

    def _apply_single_trade(self, trade: TradeHistoryEntry) -> None:
        """Apply a single trade to the portfolio."""
        symbol = trade.instrument.symbol
        qty = trade.quantity
        price = trade.avg_exec_price or trade.entry_price or 0.0
        fee = trade.fee_cost or 0.0

        # Get or create position
        pos = self._positions.get(symbol)
        if pos is None:
            pos = PositionSnapshot(
                instrument=trade.instrument,
                quantity=0.0,
                avg_price=None,
                mark_price=None,
                unrealized_pnl=None,
                entry_ts=None,
            )

        current_qty = pos.quantity
        current_avg = pos.avg_price or 0.0

        # Determine new quantity based on side
        if trade.side == TradeSide.BUY:
            new_qty = current_qty + qty
        else:  # SELL
            new_qty = current_qty - qty

        # Calculate new average price
        if abs(new_qty) < 1e-12:
            # Position closed
            new_avg = 0.0
            entry_ts = None

            # Realize P&L on close
            if current_avg > 0 and price > 0:
                if current_qty > 0:  # Was long
                    pnl = (price - current_avg) * min(qty, abs(current_qty))
                else:  # Was short
                    pnl = (current_avg - price) * min(qty, abs(current_qty))
                self._realized_pnl += pnl - fee
                self._cash_balance += pnl - fee
        elif current_qty * new_qty >= 0 and abs(new_qty) > abs(current_qty):
            # Increasing position
            if current_qty == 0:
                new_avg = price
                entry_ts = get_current_timestamp_ms()
            else:
                total_cost = abs(current_qty) * current_avg + qty * price
                new_avg = total_cost / abs(new_qty)
                entry_ts = pos.entry_ts
        elif current_qty * new_qty >= 0:
            # Reducing position (same direction)
            new_avg = current_avg
            entry_ts = pos.entry_ts

            # Partial close P&L
            if current_avg > 0 and price > 0:
                close_qty = abs(current_qty) - abs(new_qty)
                if current_qty > 0:
                    pnl = (price - current_avg) * close_qty
                else:
                    pnl = (current_avg - price) * close_qty
                self._realized_pnl += pnl - fee
                self._cash_balance += pnl - fee
        else:
            # Flipping position
            # First close existing, then open new
            if current_avg > 0 and price > 0:
                if current_qty > 0:
                    close_pnl = (price - current_avg) * abs(current_qty)
                else:
                    close_pnl = (current_avg - price) * abs(current_qty)
                self._realized_pnl += close_pnl - fee
                self._cash_balance += close_pnl - fee

            new_avg = price
            entry_ts = get_current_timestamp_ms()

        # Update cash for new position cost (virtual mode)
        if self._trading_mode == TradingMode.VIRTUAL:
            if self._market_type == MarketType.SPOT:
                # Spot: direct cash exchange
                if trade.side == TradeSide.BUY:
                    self._cash_balance -= qty * price + fee
                else:
                    self._cash_balance += qty * price - fee
            # For derivatives, margin is handled via leverage

        # Determine trade type
        trade_type = TradeType.LONG if new_qty > 0 else TradeType.SHORT if new_qty < 0 else None

        # Calculate notional
        notional = abs(new_qty) * (price if price > 0 else (current_avg or 0.0))

        # Update position
        if abs(new_qty) < 1e-12:
            # Remove closed position
            if symbol in self._positions:
                del self._positions[symbol]
        else:
            self._positions[symbol] = PositionSnapshot(
                instrument=trade.instrument,
                quantity=new_qty,
                avg_price=new_avg if new_avg > 0 else None,
                mark_price=price if price > 0 else None,
                unrealized_pnl=0.0,  # Will be updated by update_prices
                notional=notional,
                leverage=trade.leverage,
                entry_ts=entry_ts,
                trade_type=trade_type,
            )

    def update_prices(self, market_features: List[FeatureVector]) -> None:
        """Update position prices from market features."""
        # Build price lookup
        prices: Dict[str, float] = {}
        for feat in market_features:
            symbol = feat.instrument.symbol
            values = feat.values or {}
            price = (
                values.get("price.last")
                or values.get("price.close")
                or values.get("last_price")
                or values.get("close")
            )
            if price:
                prices[symbol] = float(price)

        # Update each position
        for symbol, pos in self._positions.items():
            price = prices.get(symbol)
            if price is None:
                # Try normalized symbol
                normalized = symbol.replace("-", "/")
                price = prices.get(normalized)

            if price is None or pos.avg_price is None:
                continue

            # Calculate unrealized P&L
            if pos.quantity > 0:  # Long
                unrealized_pnl = (price - pos.avg_price) * pos.quantity
            else:  # Short
                unrealized_pnl = (pos.avg_price - price) * abs(pos.quantity)

            # Calculate notional
            notional = abs(pos.quantity) * price

            # Update position
            self._positions[symbol] = PositionSnapshot(
                instrument=pos.instrument,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                mark_price=price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=(unrealized_pnl / (pos.avg_price * abs(pos.quantity)) * 100)
                if pos.avg_price and pos.quantity else None,
                notional=notional,
                leverage=pos.leverage,
                entry_ts=pos.entry_ts,
                trade_type=pos.trade_type,
            )

    def set_balance(self, balance: float) -> None:
        """Set account balance (for live mode sync)."""
        self._cash_balance = balance

    def sync_positions(self, exchange_positions: List[Dict]) -> None:
        """Sync positions from exchange data.

        Args:
            exchange_positions: List of position dicts from exchange (CCXT format)
        """
        from termcolor import cprint

        # Clear existing positions and rebuild from exchange data
        synced_symbols = set()

        for pos in exchange_positions:
            # Extract symbol
            symbol = pos.get("symbol")
            if not symbol:
                continue

            # Extract quantity (CCXT uses 'contracts' for futures, 'info' may have raw data)
            contracts = pos.get("contracts") or pos.get("contractSize") or 0
            side = pos.get("side", "").lower()

            # Some exchanges use positive/negative for long/short
            if side == "short" and contracts > 0:
                contracts = -contracts
            elif side == "long" and contracts < 0:
                contracts = abs(contracts)

            # Skip zero positions
            if abs(contracts) < 1e-12:
                continue

            synced_symbols.add(symbol)

            # Extract other fields
            entry_price = pos.get("entryPrice") or pos.get("avgPrice") or 0.0
            mark_price = pos.get("markPrice") or pos.get("lastPrice") or entry_price
            unrealized_pnl = pos.get("unrealizedPnl") or pos.get("unrealizedProfit") or 0.0
            notional = pos.get("notional") or (abs(contracts) * mark_price if mark_price else 0.0)
            leverage = pos.get("leverage") or 1.0
            timestamp = pos.get("timestamp") or get_current_timestamp_ms()

            # Determine trade type
            trade_type = TradeType.LONG if contracts > 0 else TradeType.SHORT

            # Create InstrumentRef
            from ..models import InstrumentRef
            instrument = InstrumentRef(symbol=symbol)

            # Update position
            self._positions[symbol] = PositionSnapshot(
                instrument=instrument,
                quantity=contracts,
                avg_price=entry_price if entry_price > 0 else None,
                mark_price=mark_price if mark_price > 0 else None,
                unrealized_pnl=unrealized_pnl,
                notional=notional,
                leverage=leverage,
                entry_ts=timestamp,
                trade_type=trade_type,
            )

            cprint(
                f"同步持仓 {symbol}: qty={contracts}, entry={entry_price}, "
                f"pnl={unrealized_pnl:.2f}",
                "magenta"
            )

        # Remove positions that are no longer on exchange
        symbols_to_remove = [s for s in self._positions if s not in synced_symbols]
        for symbol in symbols_to_remove:
            cprint(f"移除已平仓位: {symbol}", "white")
            del self._positions[symbol]

        if synced_symbols:
            cprint(f"已同步 {len(synced_symbols)} 个持仓: {synced_symbols}", "white")

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash_balance = self._initial_capital
        self._positions.clear()
        self._realized_pnl = 0.0

    def __repr__(self) -> str:
        return (
            f"InMemoryPortfolioService("
            f"balance={self._cash_balance:.2f}, "
            f"positions={len(self._positions)}, "
            f"realized_pnl={self._realized_pnl:.2f})"
        )
