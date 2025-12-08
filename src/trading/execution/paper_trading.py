"""Paper trading execution gateway for simulation."""

from typing import Dict, List, Optional
import time

from loguru import logger

from ..models import (
    FeatureVector,
    InstrumentRef,
    TradeInstruction,
    TradeSide,
    TxResult,
    TxStatus,
    derive_side_from_action,
)
from .interfaces import BaseExecutionGateway


class PaperExecutionGateway(BaseExecutionGateway):
    """Paper trading (simulated) execution gateway.

    Uses market features to simulate order fills with realistic slippage and fees.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        settle_coin: str = "USDT",
    ) -> None:
        """Initialize paper trading gateway.

        Args:
            initial_balance: Starting balance in USD
            fee_bps: Trading fee in basis points (default 10 = 0.1%)
            slippage_bps: Simulated slippage in basis points (default 5 = 0.05%)
            settle_coin: Settlement coin (USDT, USDC, etc.)
        """
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._settle_coin = settle_coin.upper()
        self._positions: Dict[str, Dict] = {}

    async def execute(
        self,
        instructions: List[TradeInstruction],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> List[TxResult]:
        """Execute trade instructions in simulation.

        Args:
            instructions: Trade instructions to execute
            market_features: Market features with current prices

        Returns:
            List of transaction results
        """
        if not instructions:
            return []

        # Build price lookup from market features
        prices: Dict[str, float] = {}
        if market_features:
            for feat in market_features:
                symbol = feat.instrument.symbol
                values = feat.values or {}
                # Try different price keys
                price = (
                    values.get("price.last")
                    or values.get("price.close")
                    or values.get("last_price")
                    or values.get("close")
                )
                if price:
                    prices[symbol] = float(price)

        results: List[TxResult] = []

        for inst in instructions:
            symbol = inst.instrument.symbol
            side = (
                getattr(inst, "side", None)
                or derive_side_from_action(getattr(inst, "action", None))
                or TradeSide.BUY
            )

            # Get price
            price = prices.get(symbol)
            if price is None:
                # Try normalized symbol
                normalized = symbol.replace("-", "/")
                price = prices.get(normalized)

            if price is None:
                logger.warning(f"No price found for {symbol} in paper trading")
                results.append(
                    TxResult(
                        instruction_id=inst.instruction_id,
                        instrument=inst.instrument,
                        side=side,
                        requested_qty=float(inst.quantity),
                        filled_qty=0.0,
                        status=TxStatus.REJECTED,
                        reason="no_price_available",
                        meta=inst.meta,
                    )
                )
                continue

            # Apply slippage
            slippage_mult = 1 + (self._slippage_bps / 10000.0)
            if side == TradeSide.BUY:
                exec_price = price * slippage_mult
            else:
                exec_price = price / slippage_mult

            # Calculate fee
            notional = float(inst.quantity) * exec_price
            fee_cost = notional * (self._fee_bps / 10000.0)

            # Simulate fill
            filled_qty = float(inst.quantity)

            # Update balance
            if side == TradeSide.BUY:
                cost = notional + fee_cost
                if cost > self._balance:
                    # Partial fill based on available balance
                    available = self._balance - fee_cost
                    if available > 0:
                        filled_qty = available / exec_price
                        notional = filled_qty * exec_price
                        fee_cost = notional * (self._fee_bps / 10000.0)
                    else:
                        filled_qty = 0.0
                self._balance -= (filled_qty * exec_price + fee_cost)
            else:
                self._balance += (filled_qty * exec_price - fee_cost)

            # Update position
            if filled_qty > 0:
                self._update_position(symbol, side, filled_qty, exec_price)

            # Determine status
            status = TxStatus.FILLED
            if filled_qty < inst.quantity * 0.99:
                status = TxStatus.PARTIAL
            if filled_qty == 0:
                status = TxStatus.REJECTED

            results.append(
                TxResult(
                    instruction_id=inst.instruction_id,
                    instrument=inst.instrument,
                    side=side,
                    requested_qty=float(inst.quantity),
                    filled_qty=filled_qty,
                    avg_exec_price=exec_price,
                    slippage_bps=self._slippage_bps,
                    fee_cost=fee_cost,
                    leverage=inst.leverage,
                    status=status,
                    reason=None if status == TxStatus.FILLED else "insufficient_balance",
                    meta=inst.meta,
                )
            )

        return results

    def _update_position(
        self, symbol: str, side: TradeSide, qty: float, price: float
    ) -> None:
        """Update internal position tracking."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0.0,
                "avg_price": 0.0,
                "entry_ts": None,
            }

        pos = self._positions[symbol]
        current_qty = pos["quantity"]

        if side == TradeSide.BUY:
            new_qty = current_qty + qty
            if new_qty != 0 and current_qty >= 0:
                # Long or increasing long
                total_cost = (current_qty * pos["avg_price"]) + (qty * price)
                pos["avg_price"] = total_cost / new_qty
            pos["quantity"] = new_qty
        else:  # SELL
            new_qty = current_qty - qty
            if new_qty != 0 and current_qty <= 0:
                # Short or increasing short
                total_cost = (abs(current_qty) * pos["avg_price"]) + (qty * price)
                pos["avg_price"] = total_cost / abs(new_qty)
            pos["quantity"] = new_qty

        if pos["entry_ts"] is None or (current_qty == 0 and new_qty != 0):
            pos["entry_ts"] = int(time.time() * 1000)

        if pos["quantity"] == 0:
            pos["avg_price"] = 0.0
            pos["entry_ts"] = None

    async def fetch_balance(self) -> Dict:
        """Fetch simulated account balance."""
        return {
            "free": {self._settle_coin: self._balance},
            "used": {self._settle_coin: 0.0},
            "total": {self._settle_coin: self._balance},
        }

    async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Fetch simulated positions."""
        positions = []
        for symbol, pos in self._positions.items():
            if symbols and symbol not in symbols:
                continue
            if pos["quantity"] != 0:
                positions.append({
                    "symbol": symbol,
                    "contracts": pos["quantity"],
                    "entryPrice": pos["avg_price"],
                    "unrealizedPnl": 0.0,  # Would need current prices to calculate
                    "timestamp": pos["entry_ts"],
                })
        return positions

    async def close(self) -> None:
        """Close the gateway (no-op for paper trading)."""
        pass

    def reset(self) -> None:
        """Reset to initial state."""
        self._balance = self._initial_balance
        self._positions.clear()

    def __repr__(self) -> str:
        return f"PaperExecutionGateway(balance={self._balance:.2f}, fee_bps={self._fee_bps})"
