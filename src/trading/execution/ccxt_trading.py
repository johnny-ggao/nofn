"""CCXT-based real exchange execution gateway.

Supports:
- Spot trading
- Futures/Perpetual contracts (USDT-margined, coin-margined)
- Leverage trading (cross/isolated margin)
- Multiple exchanges via CCXT unified API

Adapted from ValueCell with improvements.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Dict, List, Optional

import ccxt.async_support as ccxt
from loguru import logger

from ..models import (
    FeatureVector,
    PriceMode,
    TradeInstruction,
    TradeSide,
    TxResult,
    TxStatus,
    derive_side_from_action,
)
from .interfaces import BaseExecutionGateway


class CCXTExecutionGateway(BaseExecutionGateway):
    """Async execution gateway using CCXT unified API for real exchanges.

    Features:
    - Supports spot, futures, and perpetual contracts
    - Automatic leverage and margin mode setup
    - Symbol format normalization (BTC-USD -> BTC/USD:USD for futures)
    - Proper error handling and partial fill support
    - Fee tracking from exchange responses
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        secret_key: str = "",
        passphrase: Optional[str] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        testnet: bool = False,
        default_type: str = "swap",
        margin_mode: str = "cross",
        position_mode: str = "oneway",
        ccxt_options: Optional[Dict] = None,
    ) -> None:
        """Initialize CCXT exchange gateway.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'okx', 'bybit')
            api_key: API key for authentication
            secret_key: Secret key for authentication
            passphrase: Optional passphrase (required for OKX)
            wallet_address: Wallet address (required for Hyperliquid)
            private_key: Private key (required for Hyperliquid)
            testnet: Whether to use testnet/sandbox mode
            default_type: Default market type ('spot', 'future', 'swap')
            margin_mode: Default margin mode ('isolated' or 'cross')
            position_mode: Position mode ('oneway' or 'hedged')
            ccxt_options: Additional CCXT exchange options
        """
        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        self.default_type = default_type
        self.margin_mode = margin_mode
        self.position_mode = position_mode
        self._ccxt_options = ccxt_options or {}

        # Track leverage settings per symbol
        self._leverage_cache: Dict[str, float] = {}
        self._margin_mode_cache: Dict[str, str] = {}

        # Exchange instance (lazy-initialized)
        self._exchange: Optional[ccxt.Exchange] = None

    def _choose_default_type_for_exchange(self) -> str:
        """Return a safe defaultType for the selected exchange."""
        if self.exchange_id == "binance" and self.default_type == "swap":
            return "future"
        return self.default_type

    async def _get_exchange(self) -> ccxt.Exchange:
        """Get or create the CCXT exchange instance."""
        if self._exchange is not None:
            return self._exchange

        # Get exchange class by name
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
        except AttributeError:
            raise ValueError(
                f"Exchange '{self.exchange_id}' not supported by CCXT. "
                f"Available: {', '.join(ccxt.exchanges)}"
            )

        # Build configuration
        config = {
            "enableRateLimit": True,
            "options": {
                "defaultType": self._choose_default_type_for_exchange(),
                **self._ccxt_options,
            },
        }

        # Hyperliquid uses wallet-based authentication
        if self.exchange_id == "hyperliquid":
            if self.wallet_address:
                config["walletAddress"] = self.wallet_address
            if self.private_key:
                config["privateKey"] = self.private_key
            if "builderFee" not in config["options"]:
                config["options"]["builderFee"] = False
            if "approvedBuilderFee" not in config["options"]:
                config["options"]["approvedBuilderFee"] = False
        else:
            config["apiKey"] = self.api_key
            config["secret"] = self.secret_key
            if self.passphrase:
                config["password"] = self.passphrase

        # Create exchange instance
        self._exchange = exchange_class(config)

        # Enable sandbox/testnet mode if requested
        if self.testnet:
            self._exchange.set_sandbox_mode(True)

        # Set position mode if supported
        try:
            if self._exchange.has.get("setPositionMode"):
                hedged = self.position_mode.lower() in ("hedged", "dual", "hedge")
                await self._exchange.set_position_mode(hedged)
        except Exception as e:
            logger.warning(f"Could not set position mode: {e}")

        # Load markets
        try:
            await self._exchange.load_markets()
        except Exception as e:
            raise RuntimeError(f"Failed to load markets for {self.exchange_id}: {e}") from e

        return self._exchange

    def _normalize_symbol(self, symbol: str, market_type: Optional[str] = None) -> str:
        """Normalize symbol format for CCXT.

        Examples:
            BTC-USD -> BTC/USD (spot)
            BTC-USDT -> BTC/USDT:USDT (USDT futures on colon exchanges)
        """
        mtype = market_type or self.default_type
        base_symbol = symbol.replace("-", "/")

        # For futures/swap, append settlement currency for non-Binance exchanges
        if mtype in ("future", "swap") and self.exchange_id not in ("binance",):
            if ":" not in base_symbol:
                parts = base_symbol.split("/")
                if len(parts) == 2:
                    base_symbol = f"{parts[0]}/{parts[1]}:{parts[1]}"

        return base_symbol

    async def _setup_leverage(
        self, symbol: str, leverage: Optional[float], exchange: ccxt.Exchange
    ) -> None:
        """Set leverage for a symbol if needed and supported."""
        if leverage is None:
            leverage = 1.0

        if self._leverage_cache.get(symbol) == leverage:
            return

        if not exchange.has.get("setLeverage"):
            return

        try:
            params = {}
            if self.exchange_id == "okx":
                params["marginMode"] = self.margin_mode
            await exchange.set_leverage(int(leverage), symbol, params)
            self._leverage_cache[symbol] = leverage
        except Exception as e:
            logger.warning(f"Could not set leverage for {symbol}: {e}")

    async def _setup_margin_mode(self, symbol: str, exchange: ccxt.Exchange) -> None:
        """Set margin mode for a symbol if needed and supported."""
        if self._margin_mode_cache.get(symbol) == self.margin_mode:
            return

        if not exchange.has.get("setMarginMode"):
            return

        try:
            await exchange.set_margin_mode(self.margin_mode, symbol)
            self._margin_mode_cache[symbol] = self.margin_mode
        except Exception as e:
            logger.warning(f"Could not set margin mode for {symbol}: {e}")

    def _sanitize_client_order_id(self, raw_id: str) -> str:
        """Sanitize client order id to satisfy exchange constraints."""
        if not raw_id:
            return ""

        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_:"
        )
        max_len = 32

        if self.exchange_id == "gate":
            max_len = 28
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
            )
        elif self.exchange_id == "okx":
            max_len = 32
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
        elif self.exchange_id in ("binance", "bybit"):
            max_len = 36

        safe = "".join(ch for ch in raw_id if ch in allowed_chars)

        if safe and len(safe) <= max_len:
            return safe

        hashed = hashlib.md5(raw_id.encode()).hexdigest()
        return hashed[:max_len]

    def _normalize_reduce_only_meta(self, meta: Dict) -> Dict:
        """Normalize reduceOnly parameter for exchange compatibility."""
        result = dict(meta or {})
        exid = self.exchange_id.lower() if self.exchange_id else ""

        reduce_only_value = result.pop("reduceOnly", None)
        if reduce_only_value is None:
            reduce_only_value = result.pop("reduce_only", None)

        if exid in ("gate", "bybit"):
            param_name = "reduce_only"
        else:
            param_name = "reduceOnly"

        if reduce_only_value is None:
            result[param_name] = False
        else:
            result[param_name] = bool(reduce_only_value)

        return result

    def _build_order_params(self, inst: TradeInstruction, order_type: str) -> Dict:
        """Build exchange-specific order params."""
        params: Dict = self._normalize_reduce_only_meta(inst.meta or {})
        exid = self.exchange_id

        # Idempotency / client order id
        if exid != "hyperliquid":
            raw_client_id = params.get("clientOrderId", inst.instruction_id)
            if raw_client_id:
                client_id = self._sanitize_client_order_id(raw_client_id)
                params["clientOrderId"] = client_id

        # Default tdMode for OKX
        if exid == "okx":
            params.setdefault(
                "tdMode", "isolated" if self.margin_mode == "isolated" else "cross"
            )

        # Default time-in-force for limit orders
        if order_type == "limit":
            if exid == "binance":
                params.setdefault("timeInForce", "GTC")
            elif exid == "bybit":
                params.setdefault("time_in_force", "GoodTillCancel")

        # Strip positionSide/posSide for oneway mode
        try:
            mode = (self.position_mode or "oneway").lower()
            if mode in ("oneway", "single", "net"):
                params.pop("positionSide", None)
                params.pop("posSide", None)
        except Exception:
            pass

        return params

    async def _check_minimums(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        amount: float,
        price: Optional[float],
    ) -> Optional[str]:
        """Check if order meets minimum requirements."""
        markets = getattr(exchange, "markets", {}) or {}
        market = markets.get(symbol, {})
        limits = market.get("limits") or {}

        # Amount minimum
        min_amount = None
        amt_limits = limits.get("amount") or {}
        if amt_limits.get("min") is not None:
            try:
                min_amount = float(amt_limits["min"])
            except Exception:
                pass
        if min_amount is None:
            info = market.get("info") or {}
            min_sz = info.get("minSz")
            if min_sz is not None:
                try:
                    min_amount = float(min_sz)
                except Exception:
                    pass
        if min_amount is not None and amount < min_amount:
            return f"amount<{min_amount}"

        # Notional minimum
        min_cost = None
        cost_limits = limits.get("cost") or {}
        if cost_limits.get("min") is not None:
            try:
                min_cost = float(cost_limits["min"])
            except Exception:
                pass
        if min_cost is not None:
            est_price = price
            if est_price is None and exchange.has.get("fetchTicker"):
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    est_price = float(
                        ticker.get("last") or ticker.get("bid") or ticker.get("ask") or 0.0
                    )
                except Exception:
                    pass
            if est_price and est_price > 0:
                notional = amount * est_price
                if notional < min_cost:
                    return f"notional<{min_cost}"
        return None

    def _extract_fee_from_order(
        self, order: Dict, symbol: str, filled_qty: float, avg_price: float
    ) -> float:
        """Extract fee cost from order response with exchange-specific fallbacks."""
        fee_cost = 0.0

        try:
            # Standard CCXT unified fee field
            if "fee" in order and order["fee"]:
                fee_info = order["fee"]
                cost = fee_info.get("cost")
                if cost is not None and cost > 0:
                    return float(cost)

            # Exchange-specific extraction from 'info' field
            info = order.get("info", {})

            if self.exchange_id == "binance":
                fills = info.get("fills", [])
                for fill in fills:
                    commission = float(fill.get("commission", 0.0))
                    commission_asset = fill.get("commissionAsset", "")
                    if commission_asset in ("USDT", "BUSD", "USD", "USDC"):
                        fee_cost += commission

            elif self.exchange_id == "okx":
                fee_str = info.get("fee") or info.get("fillFee")
                if fee_str:
                    fee_cost = abs(float(fee_str))

            elif self.exchange_id == "bybit":
                cum_fee = info.get("cumExecFee") or info.get("execFee")
                if cum_fee:
                    fee_cost = float(cum_fee)

            elif self.exchange_id in ("gate", "gateio"):
                fee_str = info.get("fee")
                if fee_str:
                    fee_cost = float(fee_str)

            elif self.exchange_id == "kucoin":
                fee_str = info.get("fee")
                if fee_str:
                    fee_cost = float(fee_str)

            elif self.exchange_id == "mexc":
                fills = info.get("fills", [])
                for fill in fills:
                    commission = float(fill.get("commission", 0.0))
                    fee_cost += commission

            elif self.exchange_id == "bitget":
                fee_detail = info.get("feeDetail", {})
                if fee_detail:
                    total_fee = float(fee_detail.get("totalFee", 0.0))
                    if total_fee > 0:
                        fee_cost = total_fee

            elif self.exchange_id == "hyperliquid":
                fee_str = info.get("fee")
                if fee_str:
                    fee_cost = float(fee_str)

        except Exception as e:
            logger.warning(f"Error extracting fee for {symbol}: {e}")

        return fee_cost

    def _apply_exchange_specific_precision(
        self, symbol: str, amount: float, price: float | None, exchange: ccxt.Exchange
    ) -> tuple[float, float | None]:
        """Apply exchange-specific precision rules."""
        try:
            try:
                amount = float(exchange.amount_to_precision(symbol, amount))
            except Exception as e:
                logger.warning(f"Amount {amount} failed precision check for {symbol}: {e}")
                amount = 0.0

            if price is not None:
                price = float(exchange.price_to_precision(symbol, price))

            # Hyperliquid specific handling
            if self.exchange_id == "hyperliquid":
                market = (getattr(exchange, "markets", {}) or {}).get(symbol) or {}
                price_precision = market.get("precision", {}).get("price")
                if price is not None and price_precision == 1.0:
                    price = float(int(price))

            return amount, price

        except Exception as e:
            logger.warning(f"Precision application failed for {symbol}: {e}")
            return amount, price

    async def execute(
        self,
        instructions: List[TradeInstruction],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> List[TxResult]:
        """Execute trade instructions on the real exchange via CCXT."""
        if not instructions:
            logger.info("没有需要执行的指令")
            return []

        logger.info(f"{len(instructions)} 条指令正在被执行...")
        exchange = await self._get_exchange()
        results: List[TxResult] = []

        for inst in instructions:
            side = (
                getattr(inst, "side", None)
                or derive_side_from_action(getattr(inst, "action", None))
                or TradeSide.BUY
            )
            logger.info(f"处理 {inst.instrument.symbol} {side.value} qty={inst.quantity}")
            try:
                result = await self._execute_single(inst, exchange)
                results.append(result)
            except Exception as e:
                results.append(
                    TxResult(
                        instruction_id=inst.instruction_id,
                        instrument=inst.instrument,
                        side=side,
                        requested_qty=float(inst.quantity),
                        filled_qty=0.0,
                        status=TxStatus.ERROR,
                        reason=str(e),
                        meta=inst.meta,
                    )
                )

        return results

    async def _execute_single(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        """Execute a single trade instruction."""
        action = (inst.action.value if getattr(inst, "action", None) else None) or str(
            (inst.meta or {}).get("action") or ""
        ).lower()

        if action == "open_long":
            return await self._exec_open_long(inst, exchange)
        if action == "open_short":
            return await self._exec_open_short(inst, exchange)
        if action == "close_long":
            return await self._exec_close_long(inst, exchange)
        if action == "close_short":
            return await self._exec_close_short(inst, exchange)
        if action == "noop":
            return await self._exec_noop(inst)

        return await self._submit_order(inst, exchange)

    async def _submit_order(
        self,
        inst: TradeInstruction,
        exchange: ccxt.Exchange,
        params_override: Optional[Dict] = None,
    ) -> TxResult:
        """Submit order to exchange."""
        # Normalize symbol
        symbol = self._normalize_symbol(inst.instrument.symbol)

        # Resolve symbol against loaded markets
        markets = getattr(exchange, "markets", {}) or {}
        if symbol not in markets:
            if ":" in symbol:
                alt = symbol.split(":")[0]
                if alt in markets:
                    symbol = alt
            else:
                parts = symbol.split("/")
                if len(parts) == 2:
                    base, quote = parts
                    alt = f"{base}/{quote}:{quote}"
                    if alt in markets:
                        symbol = alt
                    elif quote in ("USD", "USDT"):
                        alt_quote = "USDT" if quote == "USD" else "USD"
                        alt2 = f"{base}/{alt_quote}"
                        alt3 = f"{base}/{alt_quote}:{alt_quote}"
                        if alt2 in markets:
                            symbol = alt2
                        elif alt3 in markets:
                            symbol = alt3

        # Setup leverage and margin mode for opening positions
        action = (inst.action.value if getattr(inst, "action", None) else None) or str(
            (inst.meta or {}).get("action") or ""
        ).lower()
        is_opening = action in ("open_long", "open_short")

        if is_opening:
            await self._setup_leverage(symbol, inst.leverage, exchange)
            await self._setup_margin_mode(symbol, exchange)

        # Map instruction to CCXT parameters
        local_side = (
            getattr(inst, "side", None)
            or derive_side_from_action(getattr(inst, "action", None))
            or TradeSide.BUY
        )
        side = "buy" if local_side == TradeSide.BUY else "sell"
        order_type = "limit" if inst.price_mode == PriceMode.LIMIT else "market"
        amount = float(inst.quantity)
        price = float(inst.limit_price) if inst.limit_price else None

        # For OKX derivatives, convert amount to contracts
        ct_val = None
        try:
            market = (getattr(exchange, "markets", {}) or {}).get(symbol) or {}
            if self.exchange_id == "okx" and market.get("contract"):
                try:
                    ct_val = float(market.get("contractSize") or 0.0)
                except Exception:
                    pass
                if not ct_val:
                    info = market.get("info") or {}
                    try:
                        ct_val = float(info.get("ctVal") or 0.0)
                    except Exception:
                        pass
                if ct_val and ct_val > 0:
                    amount = amount / ct_val
        except Exception:
            pass

        # Apply precision
        amount, price = self._apply_exchange_specific_precision(symbol, amount, price, exchange)

        # If amount became zero, skip order
        if amount <= 0:
            return TxResult(
                instruction_id=inst.instruction_id,
                instrument=inst.instrument,
                side=local_side,
                requested_qty=float(inst.quantity),
                filled_qty=0.0,
                status=TxStatus.REJECTED,
                reason="amount_too_small_for_precision",
                meta=inst.meta,
            )

        # Check minimums
        try:
            reject_reason = await self._check_minimums(exchange, symbol, amount, price)
        except Exception as e:
            logger.warning(f"Minimum check failed for {symbol}: {e}")
            reject_reason = f"minimum_check_failed:{e}"
        if reject_reason is not None:
            logger.warning(f"Skipping order due to {reject_reason}")
            return TxResult(
                instruction_id=inst.instruction_id,
                instrument=inst.instrument,
                side=local_side,
                requested_qty=float(inst.quantity),
                filled_qty=0.0,
                status=TxStatus.REJECTED,
                reason=reject_reason,
                meta=inst.meta,
            )

        # Build order params
        params = self._build_order_params(inst, order_type)
        if params_override:
            try:
                params.update(params_override)
            except Exception:
                pass

        # Strip positionSide after overrides
        try:
            mode = (self.position_mode or "oneway").lower()
            if mode in ("oneway", "single", "net"):
                params.pop("positionSide", None)
                params.pop("posSide", None)
        except Exception:
            pass

        # Hyperliquid special handling for market orders
        if self.exchange_id == "hyperliquid" and order_type == "market":
            try:
                if price is None:
                    ticker = await exchange.fetch_ticker(symbol)
                    price = float(ticker.get("last") or ticker.get("close") or 0.0)

                if price > 0:
                    slippage_pct = (inst.max_slippage_bps or 50.0) / 10000.0
                    if side == "buy":
                        price = price * (1 + slippage_pct)
                    else:
                        price = price * (1 - slippage_pct)

                    _, price = self._apply_exchange_specific_precision(
                        symbol, amount, price, exchange
                    )
                    params["timeInForce"] = "Ioc"
            except Exception as e:
                logger.warning(f"Could not setup Hyperliquid market order: {e}")

        # Create order
        try:
            logger.info(
                f"Creating {order_type} order: {side} {amount} {symbol} @ {price if price else 'market'}"
            )
            order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            logger.info(
                f"Order created: id={order.get('id')}, status={order.get('status')}, filled={order.get('filled')}"
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"ERROR creating order for {symbol}: {error_msg}")
            return TxResult(
                instruction_id=inst.instruction_id,
                instrument=inst.instrument,
                side=local_side,
                requested_qty=amount,
                filled_qty=0.0,
                status=TxStatus.ERROR,
                reason=f"create_order_failed: {error_msg}",
                meta=inst.meta,
            )

        # For market orders, wait for fill
        if order_type == "market":
            order_id = order.get("id")
            if order_id and exchange.has.get("fetchOrder"):
                try:
                    await asyncio.sleep(0.5)
                    order = await exchange.fetch_order(order_id, symbol)
                except Exception as e:
                    logger.warning(f"Could not fetch order status for {symbol}: {e}")

        # Parse order response
        filled_qty = float(order.get("filled", 0.0))

        # For OKX derivatives, convert filled_qty back to base units
        if self.exchange_id == "okx" and ct_val and ct_val > 0 and filled_qty > 0:
            filled_qty = filled_qty * ct_val

        avg_price = float(order.get("average") or 0.0)
        fee_cost = self._extract_fee_from_order(order, symbol, filled_qty, avg_price)

        # Calculate slippage
        slippage_bps = None
        if avg_price and inst.limit_price and inst.price_mode == PriceMode.LIMIT:
            expected = float(inst.limit_price)
            slippage = abs(avg_price - expected) / expected * 10000.0
            slippage_bps = slippage

        # Determine status
        status = TxStatus.FILLED
        if filled_qty < amount * 0.99:
            status = TxStatus.PARTIAL
        if filled_qty == 0:
            status = TxStatus.REJECTED

        return TxResult(
            instruction_id=inst.instruction_id,
            instrument=inst.instrument,
            side=local_side,
            requested_qty=amount,
            filled_qty=filled_qty,
            avg_exec_price=avg_price if avg_price > 0 else None,
            slippage_bps=slippage_bps,
            fee_cost=fee_cost if fee_cost > 0 else None,
            leverage=inst.leverage,
            status=status,
            reason=order.get("status") if status != TxStatus.FILLED else None,
            meta=inst.meta,
        )

    async def _exec_open_long(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": False}
        else:
            overrides = {"reduceOnly": False}
        return await self._submit_order(inst, exchange, overrides)

    async def _exec_open_short(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": False}
        else:
            overrides = {"reduceOnly": False}
        return await self._submit_order(inst, exchange, overrides)

    async def _exec_close_long(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": True}
        else:
            overrides = {"reduceOnly": True}
        return await self._submit_order(inst, exchange, overrides)

    async def _exec_close_short(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": True}
        else:
            overrides = {"reduceOnly": True}
        return await self._submit_order(inst, exchange, overrides)

    async def _exec_noop(self, inst: TradeInstruction) -> TxResult:
        side = (
            getattr(inst, "side", None)
            or derive_side_from_action(getattr(inst, "action", None))
            or TradeSide.BUY
        )
        return TxResult(
            instruction_id=inst.instruction_id,
            instrument=inst.instrument,
            side=side,
            requested_qty=float(inst.quantity),
            filled_qty=0.0,
            status=TxStatus.REJECTED,
            reason="noop",
            meta=inst.meta,
        )

    async def fetch_balance(self) -> Dict:
        """Fetch account balance from exchange."""
        exchange = await self._get_exchange()
        return await exchange.fetch_balance()

    async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Fetch current positions from exchange."""
        exchange = await self._get_exchange()

        if not exchange.has.get("fetchPositions"):
            return []

        normalized_symbols = None
        if symbols:
            normalized_symbols = [self._normalize_symbol(s) for s in symbols]

        try:
            positions = await exchange.fetch_positions(normalized_symbols)
            return positions
        except Exception as e:
            logger.warning(f"Could not fetch positions: {e}")
            return []

    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an open order."""
        exchange = await self._get_exchange()
        normalized_symbol = self._normalize_symbol(symbol)
        return await exchange.cancel_order(order_id, normalized_symbol)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch open orders from exchange."""
        exchange = await self._get_exchange()
        normalized_symbol = self._normalize_symbol(symbol) if symbol else None
        return await exchange.fetch_open_orders(normalized_symbol)

    async def close(self) -> None:
        """Close the exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    def __repr__(self) -> str:
        mode = "testnet" if self.testnet else "live"
        return (
            f"CCXTExecutionGateway(exchange={self.exchange_id}, "
            f"type={self.default_type}, margin={self.margin_mode}, mode={mode})"
        )


async def create_ccxt_gateway(
    exchange_id: str,
    api_key: str,
    secret_key: str,
    passphrase: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    testnet: bool = False,
    market_type: str = "swap",
    margin_mode: str = "cross",
    position_mode: str = "oneway",
    **ccxt_options,
) -> CCXTExecutionGateway:
    """Factory function to create and initialize a CCXT execution gateway."""
    gateway = CCXTExecutionGateway(
        exchange_id=exchange_id,
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        wallet_address=wallet_address,
        private_key=private_key,
        testnet=testnet,
        default_type=market_type,
        margin_mode=margin_mode,
        position_mode=position_mode,
        ccxt_options=ccxt_options,
    )

    # Pre-load markets to validate connection
    await gateway._get_exchange()

    return gateway
