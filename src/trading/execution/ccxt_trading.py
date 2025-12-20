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
from termcolor import cprint

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

        # Set position mode if supported (futures only)
        await self._ensure_position_mode()

        # Load markets
        try:
            await self._exchange.load_markets()
        except Exception as e:
            raise RuntimeError(f"Failed to load markets for {self.exchange_id}: {e}") from e

        return self._exchange

    async def _ensure_position_mode(self) -> None:
        """确保持仓模式符合配置，仅在需要时设置。

        检查当前持仓模式，如果已经是目标模式则跳过设置。
        """
        if not self._exchange:
            return

        # 检查交易所是否支持设置持仓模式
        if not self._exchange.has.get("setPositionMode"):
            return

        # 现货不需要设置持仓模式
        if self.default_type == "spot":
            return

        target_hedged = self.position_mode.lower() in ("hedged", "dual", "hedge")

        # 尝试获取当前持仓模式
        try:
            # Binance: fetch_position_mode 返回 {"dualSidePosition": true/false}
            if hasattr(self._exchange, "fapiPrivateGetPositionSideDual"):
                result = await self._exchange.fapiPrivateGetPositionSideDual()
                current_hedged = result.get("dualSidePosition", False)

                if current_hedged == target_hedged:
                    mode_name = "hedge" if current_hedged else "one-way"
                    cprint(f"Position mode already set to {mode_name}, skipping", "magenta")
                    return

                # 需要切换模式
                cprint(
                    f"Switching position mode: {'one-way' if current_hedged else 'hedge'} -> "
                    f"{'hedge' if target_hedged else 'one-way'}"
                , "cyan")
        except Exception as e:
            cprint(f"Could not fetch current position mode: {e}", "magenta")

        # 设置持仓模式
        try:
            await self._exchange.set_position_mode(target_hedged)
            mode_name = "hedge" if target_hedged else "one-way"
            cprint(f"Position mode set to {mode_name}", "white")
        except Exception as e:
            err_str = str(e)
            # Binance -4059: 已经是目标模式
            if "-4059" in err_str or "No need to change" in err_str:
                cprint(f"Position mode already correct: {e}", "white")
            else:
                cprint(f"Could not set position mode: {e}", "yellow")

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
            cprint(f"Could not set leverage for {symbol}: {e}", "yellow")

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
            cprint(f"Could not set margin mode for {symbol}: {e}", "yellow")

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
            cprint(f"Error extracting fee for {symbol}: {e}", "yellow")

        return fee_cost

    def _apply_exchange_specific_precision(
        self, symbol: str, amount: float, price: float | None, exchange: ccxt.Exchange
    ) -> tuple[float, float | None]:
        """Apply exchange-specific precision rules."""
        try:
            try:
                amount = float(exchange.amount_to_precision(symbol, amount))
            except Exception as e:
                cprint(f"Amount {amount} failed precision check for {symbol}: {e}", "yellow")
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
            cprint(f"Precision application failed for {symbol}: {e}", "yellow")
            return amount, price

    async def execute(
        self,
        instructions: List[TradeInstruction],
        market_features: Optional[List[FeatureVector]] = None,
    ) -> List[TxResult]:
        """Execute trade instructions on the real exchange via CCXT."""
        if not instructions:
            cprint("没有需要执行的指令", "white")
            return []

        cprint(f"{len(instructions)} 条指令正在被执行...", "white")
        exchange = await self._get_exchange()
        results: List[TxResult] = []

        for inst in instructions:
            side = (
                getattr(inst, "side", None)
                or derive_side_from_action(getattr(inst, "action", None))
                or TradeSide.BUY
            )
            cprint(f"处理 {inst.instrument.symbol} {side.value} qty={inst.quantity}", "white")
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
                    elif quote in ("USD", "USDT", "USDC"):
                        # Try alternative quote currencies
                        # Priority: exact match > USDT > USDC > USD
                        alt_quotes = []
                        if quote == "USD":
                            alt_quotes = ["USDT", "USDC"]
                        elif quote == "USDT":
                            alt_quotes = ["USD", "USDC"]
                        elif quote == "USDC":
                            alt_quotes = ["USD", "USDT"]

                        found = False
                        for alt_quote in alt_quotes:
                            alt2 = f"{base}/{alt_quote}"
                            alt3 = f"{base}/{alt_quote}:{alt_quote}"
                            if alt2 in markets:
                                symbol = alt2
                                found = True
                                break
                            elif alt3 in markets:
                                symbol = alt3
                                found = True
                                break

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
            cprint(f"Minimum check failed for {symbol}: {e}", "yellow")
            reject_reason = f"minimum_check_failed:{e}"
        if reject_reason is not None:
            cprint(f"Skipping order due to {reject_reason}", "yellow")
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
                cprint(f"Could not setup Hyperliquid market order: {e}", "yellow")

        # Create order
        try:
            cprint(
                f"Creating {order_type} order: {side} {amount} {symbol} @ {price if price else 'market'}"
            , "cyan")
            order = await exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            cprint(
                f"Order created: id={order.get('id', "cyan")}, status={order.get('status')}, filled={order.get('filled')}"
            )
        except Exception as e:
            error_msg = str(e)
            cprint(f"ERROR creating order for {symbol}: {error_msg}", "red")
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
                    cprint(f"Could not fetch order status for {symbol}: {e}", "yellow")

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

    async def _place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        exchange: ccxt.Exchange,
    ) -> Optional[Dict]:
        """Place a stop loss order after entry.

        Args:
            symbol: Normalized symbol
            side: 'buy' or 'sell' (opposite of position direction)
            quantity: Position quantity to protect
            stop_price: Trigger price for stop loss
            exchange: CCXT exchange instance

        Returns:
            Order response dict or None if failed
        """
        try:
            # Check if exchange supports stop orders
            if not exchange.has.get("createOrder"):
                cprint(f"{self.exchange_id} does not support createOrder", "yellow")
                return None

            # Apply precision to stop price
            _, stop_price = self._apply_exchange_specific_precision(
                symbol, quantity, stop_price, exchange
            )

            # Build stop order params based on exchange
            params: Dict = {}

            if self.exchange_id == "binance":
                # Binance futures uses conditional orders for stop loss
                # Must use workingType to specify trigger price type
                order_type = "STOP_MARKET"
                params["stopPrice"] = stop_price
                params["reduceOnly"] = True
                params["workingType"] = "MARK_PRICE"  # or "CONTRACT_PRICE"
                # Note: For Binance futures, this will use the /fapi/v1/order endpoint
                # which supports STOP_MARKET orders

            elif self.exchange_id == "okx":
                # OKX uses conditional orders
                order_type = "market"
                params["stopLossPrice"] = stop_price
                params["tdMode"] = "isolated" if self.margin_mode == "isolated" else "cross"
                params["reduceOnly"] = True

            elif self.exchange_id == "bybit":
                # Bybit uses conditional orders
                order_type = "market"
                params["triggerPrice"] = stop_price
                params["triggerDirection"] = 2 if side == "sell" else 1  # 1=rise, 2=fall
                params["reduce_only"] = True

            elif self.exchange_id == "gate":
                # Gate.io uses price_type for stop orders
                order_type = "market"
                params["stopPrice"] = stop_price
                params["reduce_only"] = True

            elif self.exchange_id == "hyperliquid":
                # Hyperliquid uses trigger orders
                order_type = "market"
                params["triggerPrice"] = stop_price
                params["reduceOnly"] = True

            else:
                # Generic fallback - may not work for all exchanges
                order_type = "market"
                params["stopPrice"] = stop_price
                params["reduceOnly"] = True

            cprint(
                f"Placing stop loss: {side} {quantity} {symbol} @ trigger {stop_price}"
            , "cyan")

            # For Binance, use create_order with type=STOP_MARKET
            if self.exchange_id == "binance":
                # Binance futures requires specific parameters for stop orders
                # Using fapiPrivatePostOrder directly for more control
                try:
                    order = await exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side,
                        amount=quantity,
                        price=None,  # Market order, no limit price
                        params=params,
                    )
                except Exception as e:
                    # If normal endpoint fails, try using the private API directly
                    if "-4120" in str(e):
                        cprint(
                            f"Standard API failed ({e}), attempting alternative method...",
                            "yellow"
                        )
                        # For Binance futures, we can use a different approach
                        # Create the order using fapiPrivatePostOrder
                        order_params = {
                            "symbol": symbol.replace("/", "").replace(":USDC", ""),
                            "side": side.upper(),
                            "type": order_type,
                            "quantity": quantity,
                            "stopPrice": str(stop_price),
                            "reduceOnly": "true",
                            "workingType": "MARK_PRICE",
                        }
                        order = await exchange.fapiPrivatePostOrder(order_params)
                    else:
                        raise
            else:
                # For other exchanges, try createStopOrder or create_order with params
                if exchange.has.get("createStopOrder"):
                    order = await exchange.create_stop_order(
                        symbol=symbol,
                        type="market",
                        side=side,
                        amount=quantity,
                        price=stop_price,
                        params=params,
                    )
                elif exchange.has.get("createStopLossOrder"):
                    order = await exchange.create_stop_loss_order(
                        symbol=symbol,
                        type="market",
                        side=side,
                        amount=quantity,
                        price=stop_price,
                        params=params,
                    )
                else:
                    # Fallback to regular create_order with stop params
                    order = await exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side,
                        amount=quantity,
                        price=None,
                        params=params,
                    )

            cprint(
                f"Stop loss order placed: id={order.get('id', "cyan")}, status={order.get('status')}"
            )
            return order

        except Exception as e:
            cprint(f"Failed to place stop loss for {symbol}: {e}", "red")
            return None

    async def _place_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float,
        exchange: ccxt.Exchange,
    ) -> Optional[Dict]:
        """Place a take profit order after entry.

        Args:
            symbol: Normalized symbol
            side: 'buy' or 'sell' (opposite of position direction)
            quantity: Position quantity
            tp_price: Trigger price for take profit
            exchange: CCXT exchange instance

        Returns:
            Order response dict or None if failed
        """
        try:
            if not exchange.has.get("createOrder"):
                return None

            _, tp_price = self._apply_exchange_specific_precision(
                symbol, quantity, tp_price, exchange
            )

            params: Dict = {}

            if self.exchange_id == "binance":
                # Binance futures uses conditional orders for take profit
                order_type = "TAKE_PROFIT_MARKET"
                params["stopPrice"] = tp_price
                params["reduceOnly"] = True
                params["workingType"] = "MARK_PRICE"  # or "CONTRACT_PRICE"

            elif self.exchange_id == "okx":
                order_type = "market"
                params["takeProfitPrice"] = tp_price
                params["tdMode"] = "isolated" if self.margin_mode == "isolated" else "cross"
                params["reduceOnly"] = True

            elif self.exchange_id == "bybit":
                order_type = "market"
                params["triggerPrice"] = tp_price
                params["triggerDirection"] = 1 if side == "sell" else 2
                params["reduce_only"] = True

            else:
                order_type = "market"
                params["stopPrice"] = tp_price
                params["reduceOnly"] = True

            cprint(
                f"Placing take profit: {side} {quantity} {symbol} @ trigger {tp_price}"
            , "cyan")

            if self.exchange_id == "binance":
                # Binance futures requires specific parameters for take profit orders
                try:
                    order = await exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side,
                        amount=quantity,
                        price=None,
                        params=params,
                    )
                except Exception as e:
                    # If normal endpoint fails, try using the private API directly
                    if "-4120" in str(e):
                        cprint(
                            f"Standard API failed ({e}), attempting alternative method...",
                            "yellow"
                        )
                        order_params = {
                            "symbol": symbol.replace("/", "").replace(":USDC", ""),
                            "side": side.upper(),
                            "type": order_type,
                            "quantity": quantity,
                            "stopPrice": str(tp_price),
                            "reduceOnly": "true",
                            "workingType": "MARK_PRICE",
                        }
                        order = await exchange.fapiPrivatePostOrder(order_params)
                    else:
                        raise
            elif exchange.has.get("createTakeProfitOrder"):
                order = await exchange.create_take_profit_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=quantity,
                    price=tp_price,
                    params=params,
                )
            else:
                order = await exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=None,
                    params=params,
                )

            cprint(
                f"Take profit order placed: id={order.get('id', "cyan")}, status={order.get('status')}"
            )
            return order

        except Exception as e:
            cprint(f"Failed to place take profit for {symbol}: {e}", "red")
            return None

    async def _exec_open_long(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": False}
        else:
            overrides = {"reduceOnly": False}

        # Execute the entry order
        result = await self._submit_order(inst, exchange, overrides)

        # Place separate SL/TP orders after entry is filled
        # Note: Binance Futures does NOT support attaching SL/TP to entry orders,
        # must place separate conditional orders
        if result.status == TxStatus.FILLED and result.filled_qty > 0:
            symbol = self._normalize_symbol(inst.instrument.symbol)

            if inst.sl_price or inst.tp_price:
                cprint(
                    f"开多仓成功，下止损止盈单: sl={inst.sl_price}, tp={inst.tp_price}",
                    "cyan",
                )

            # Place stop loss (sell to close long)
            if inst.sl_price:
                await self._place_stop_loss_order(
                    symbol=symbol,
                    side="sell",
                    quantity=result.filled_qty,
                    stop_price=inst.sl_price,
                    exchange=exchange,
                )

            # Place take profit (sell to close long)
            if inst.tp_price:
                await self._place_take_profit_order(
                    symbol=symbol,
                    side="sell",
                    quantity=result.filled_qty,
                    tp_price=inst.tp_price,
                    exchange=exchange,
                )

        return result

    async def _exec_open_short(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": False}
        else:
            overrides = {"reduceOnly": False}

        # Execute the entry order
        result = await self._submit_order(inst, exchange, overrides)

        # Place separate SL/TP orders after entry is filled
        # Note: Binance Futures does NOT support attaching SL/TP to entry orders,
        # must place separate conditional orders
        if result.status == TxStatus.FILLED and result.filled_qty > 0:
            symbol = self._normalize_symbol(inst.instrument.symbol)

            if inst.sl_price or inst.tp_price:
                cprint(
                    f"开空仓成功，下止损止盈单: sl={inst.sl_price}, tp={inst.tp_price}",
                    "cyan",
                )

            # Place stop loss (buy to close short)
            if inst.sl_price:
                await self._place_stop_loss_order(
                    symbol=symbol,
                    side="buy",
                    quantity=result.filled_qty,
                    stop_price=inst.sl_price,
                    exchange=exchange,
                )

            # Place take profit (buy to close short)
            if inst.tp_price:
                await self._place_take_profit_order(
                    symbol=symbol,
                    side="buy",
                    quantity=result.filled_qty,
                    tp_price=inst.tp_price,
                    exchange=exchange,
                )

        return result

    async def _cancel_open_orders_for_symbol(
        self, symbol: str, exchange: ccxt.Exchange
    ) -> int:
        """Cancel all open orders (including SL/TP conditional orders) for a symbol.

        Args:
            symbol: Normalized symbol (e.g., 'BTC/USDC:USDC')
            exchange: CCXT exchange instance

        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0

        cprint(f"Cancelling all open orders for {symbol}...", "magenta")

        # For Binance Futures: need to cancel both regular orders AND Algo Orders
        # Algo Orders are used for STOP_MARKET, TAKE_PROFIT_MARKET conditionals
        if self.exchange_id == "binance":
            # Step 1: Cancel regular orders using cancel_all_orders
            try:
                result = await exchange.cancel_all_orders(symbol)
                if result:
                    if isinstance(result, list):
                        cancelled_count = len(result)
                    else:
                        cancelled_count = 1
                    cprint(
                        f"Cancelled regular orders for {symbol} (count: {cancelled_count})",
                        "cyan",
                    )
            except Exception as e:
                cprint(f"cancel_all_orders failed for {symbol}: {e}", "yellow")

            # Step 2: Cancel Algo Orders (conditional orders like STOP_MARKET, TAKE_PROFIT_MARKET)
            algo_cancelled = await self._cancel_binance_algo_orders(symbol, exchange)
            cancelled_count += algo_cancelled
        else:
            # For other exchanges, cancel orders one by one
            cancelled_count = await self._cancel_orders_manually(symbol, exchange)

        return cancelled_count

    async def _cancel_binance_algo_orders(
        self, symbol: str, exchange: ccxt.Exchange
    ) -> int:
        """Cancel Binance Futures Algo Orders (conditional orders) for a symbol.

        Binance Futures stores STOP_MARKET and TAKE_PROFIT_MARKET orders as "Algo Orders"
        which require special API endpoints to query and cancel:
        - Query: GET /fapi/v1/openAlgoOrders -> fapiPrivateGetOpenAlgoOrders
        - Cancel: DELETE /fapi/v1/algoOrder -> fapiPrivateDeleteAlgoOrder
        - Cancel all: DELETE /fapi/v1/algo/openOrders -> fapiPrivateDeleteAlgoOpenOrders

        Args:
            symbol: Normalized symbol (e.g., 'BTC/USDC:USDC')
            exchange: CCXT exchange instance

        Returns:
            Number of Algo Orders cancelled
        """
        cancelled_count = 0

        try:
            # Convert symbol to Binance format (e.g., 'BTC/USDC:USDC' -> 'BTCUSDC')
            binance_symbol = symbol.replace("/", "").replace(":USDC", "").replace(":USDT", "")

            # Try to cancel all algo orders at once first (more efficient)
            # Using DELETE /fapi/v1/algo/openOrders
            try:
                if hasattr(exchange, "fapiPrivateDeleteAlgoOpenOrders"):
                    result = await exchange.fapiPrivateDeleteAlgoOpenOrders({
                        "symbol": binance_symbol,
                    })
                    cprint(f"Cancelled all Algo Orders for {symbol}: {result}", "cyan")
                    # Result contains list of cancelled orders
                    if isinstance(result, dict) and "orders" in result:
                        cancelled_count = len(result["orders"])
                    elif isinstance(result, list):
                        cancelled_count = len(result)
                    else:
                        cancelled_count = 1 if result else 0
                    return cancelled_count
                elif hasattr(exchange, "fapiprivate_delete_algoopenorders"):
                    result = await exchange.fapiprivate_delete_algoopenorders({
                        "symbol": binance_symbol,
                    })
                    cprint(f"Cancelled all Algo Orders for {symbol}: {result}", "cyan")
                    if isinstance(result, dict) and "orders" in result:
                        cancelled_count = len(result["orders"])
                    elif isinstance(result, list):
                        cancelled_count = len(result)
                    else:
                        cancelled_count = 1 if result else 0
                    return cancelled_count
            except Exception as e:
                # If batch cancel fails, fall back to individual cancellation
                cprint(f"Batch cancel Algo Orders failed: {e}, trying individual cancel", "yellow")

            # Fetch open Algo Orders
            # Note: CCXT uses fapiPrivateGetOpenAlgoOrders (not fapiPrivateGetAlgoOpenOrders)
            algo_orders = None
            try:
                if hasattr(exchange, "fapiPrivateGetOpenAlgoOrders"):
                    algo_orders = await exchange.fapiPrivateGetOpenAlgoOrders({
                        "symbol": binance_symbol,
                    })
                elif hasattr(exchange, "fapiprivate_get_openalgoorders"):
                    algo_orders = await exchange.fapiprivate_get_openalgoorders({
                        "symbol": binance_symbol,
                    })
                else:
                    cprint(
                        "Binance Algo Orders API not available in this CCXT version",
                        "yellow",
                    )
                    return 0
            except Exception as e:
                cprint(f"Failed to fetch Algo Orders: {e}", "yellow")
                return 0

            # Check if we got orders
            orders_list = algo_orders.get("orders", []) if isinstance(algo_orders, dict) else algo_orders
            if not orders_list:
                cprint(f"No Algo Orders found for {symbol}", "magenta")
                return 0

            cprint(f"Found {len(orders_list)} Algo Orders for {symbol}", "magenta")

            # Cancel each Algo Order
            for order in orders_list:
                algo_id = order.get("algoId")
                algo_type = order.get("algoType", "unknown")
                algo_status = order.get("algoStatus", "unknown")

                if not algo_id:
                    continue

                # Only cancel active orders
                if algo_status not in ("NEW", "PARTIALLY_FILLED"):
                    continue

                try:
                    # Cancel using DELETE /fapi/v1/algoOrder
                    if hasattr(exchange, "fapiPrivateDeleteAlgoOrder"):
                        await exchange.fapiPrivateDeleteAlgoOrder({
                            "symbol": binance_symbol,
                            "algoId": algo_id,
                        })
                    elif hasattr(exchange, "fapiprivate_delete_algoorder"):
                        await exchange.fapiprivate_delete_algoorder({
                            "symbol": binance_symbol,
                            "algoId": algo_id,
                        })
                    else:
                        cprint(f"No method to cancel Algo Order {algo_id}", "yellow")
                        continue

                    cancelled_count += 1
                    cprint(
                        f"Cancelled Algo Order {algo_id}: {algo_type} ({algo_status})",
                        "cyan",
                    )
                except Exception as e:
                    cprint(f"Failed to cancel Algo Order {algo_id}: {e}", "yellow")

        except Exception as e:
            cprint(f"Failed to fetch/cancel Algo Orders for {symbol}: {e}", "yellow")

        if cancelled_count > 0:
            cprint(f"Cancelled {cancelled_count} Algo Orders for {symbol}", "cyan")

        return cancelled_count

    async def _cancel_orders_manually(
        self, symbol: str, exchange: ccxt.Exchange
    ) -> int:
        """Manually fetch and cancel each open order for a symbol.

        Fallback method when cancel_all_orders is not available or fails.
        """
        cancelled_count = 0
        try:
            open_orders = await exchange.fetch_open_orders(symbol)
            if open_orders:
                cprint(
                    f"Found {len(open_orders)} open orders for {symbol}, cancelling...",
                    "magenta",
                )
                for order in open_orders:
                    order_id = order.get("id")
                    order_type = order.get("type", "unknown")
                    order_side = order.get("side", "unknown")
                    try:
                        await exchange.cancel_order(order_id, symbol)
                        cancelled_count += 1
                        cprint(
                            f"Cancelled order {order_id}: {order_type} {order_side}",
                            "cyan",
                        )
                    except Exception as e:
                        cprint(f"Failed to cancel order {order_id}: {e}", "yellow")
            else:
                cprint(f"No open orders found for {symbol}", "magenta")
        except Exception as e:
            cprint(f"Failed to fetch/cancel open orders for {symbol}: {e}", "yellow")

        return cancelled_count

    async def _exec_close_long(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": True}
        else:
            overrides = {"reduceOnly": True}

        # Execute the close order
        result = await self._submit_order(inst, exchange, overrides)

        # If close was successful, cancel any pending SL/TP orders for this symbol
        if result.status == TxStatus.FILLED and result.filled_qty > 0:
            symbol = self._normalize_symbol(inst.instrument.symbol)
            await self._cancel_open_orders_for_symbol(symbol, exchange)

        return result

    async def _exec_close_short(
        self, inst: TradeInstruction, exchange: ccxt.Exchange
    ) -> TxResult:
        if self.exchange_id == "bybit":
            overrides = {"reduce_only": True}
        else:
            overrides = {"reduceOnly": True}

        # Execute the close order
        result = await self._submit_order(inst, exchange, overrides)

        # If close was successful, cancel any pending SL/TP orders for this symbol
        if result.status == TxStatus.FILLED and result.filled_qty > 0:
            symbol = self._normalize_symbol(inst.instrument.symbol)
            await self._cancel_open_orders_for_symbol(symbol, exchange)

        return result

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
            cprint(f"Could not fetch positions: {e}", "yellow")
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

    async def fetch_recent_trades(
        self,
        symbols: Optional[List[str]] = None,
        since_ms: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Fetch recent trades (my trades) from exchange.

        获取最近的成交记录，用于了解最近的开平仓情况。

        Args:
            symbols: 要查询的交易对列表，None 表示查询所有配置的交易对
            since_ms: 起始时间戳（毫秒），None 表示获取最近的
            limit: 每个交易对返回的最大记录数

        Returns:
            成交记录列表，每条记录包含：
            - symbol: 交易对
            - side: buy/sell
            - amount: 成交数量
            - price: 成交价格
            - cost: 成交金额
            - fee: 手续费
            - timestamp: 时间戳
            - datetime: ISO 时间字符串
            - info: 原始交易所数据
        """
        exchange = await self._get_exchange()

        if not exchange.has.get("fetchMyTrades"):
            cprint(f"{self.exchange_id} does not support fetchMyTrades", "yellow")
            return []

        all_trades: List[Dict] = []

        # 如果没有指定 symbols，返回空（需要外部传入）
        if not symbols:
            return []

        for symbol in symbols:
            try:
                normalized_symbol = self._normalize_symbol(symbol)
                trades = await exchange.fetch_my_trades(
                    symbol=normalized_symbol,
                    since=since_ms,
                    limit=limit,
                )

                # 标准化并添加到结果
                for trade in trades:
                    # CCXT 返回的 trade 结构已经比较标准化
                    # 添加一些便于理解的字段
                    trade_info = {
                        "symbol": symbol,  # 使用原始 symbol
                        "normalized_symbol": normalized_symbol,
                        "side": trade.get("side"),  # buy/sell
                        "amount": trade.get("amount"),
                        "price": trade.get("price"),
                        "cost": trade.get("cost"),
                        "fee": trade.get("fee"),
                        "timestamp": trade.get("timestamp"),
                        "datetime": trade.get("datetime"),
                        "order_id": trade.get("order"),
                        "trade_id": trade.get("id"),
                        # 推断交易类型（开仓/平仓）- 需要结合持仓判断
                        # 这里只提供原始数据，让上层逻辑判断
                        "info": trade.get("info", {}),
                    }
                    all_trades.append(trade_info)

            except Exception as e:
                cprint(f"Failed to fetch trades for {symbol}: {e}", "yellow")

        # 按时间倒序排列（最新的在前）
        all_trades.sort(key=lambda t: t.get("timestamp", 0), reverse=True)

        return all_trades

    async def fetch_recent_orders(
        self,
        symbols: Optional[List[str]] = None,
        since_ms: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Fetch recent orders (closed orders) from exchange.

        获取最近的订单记录（包括已完成的订单），用于了解最近的交易活动。

        Args:
            symbols: 要查询的交易对列表
            since_ms: 起始时间戳（毫秒）
            limit: 每个交易对返回的最大记录数

        Returns:
            订单记录列表
        """
        exchange = await self._get_exchange()

        # 优先使用 fetchClosedOrders，如果不支持则用 fetchOrders
        has_closed = exchange.has.get("fetchClosedOrders")
        has_orders = exchange.has.get("fetchOrders")

        if not has_closed and not has_orders:
            cprint(f"{self.exchange_id} does not support fetchClosedOrders/fetchOrders", "yellow")
            return []

        all_orders: List[Dict] = []

        if not symbols:
            return []

        for symbol in symbols:
            try:
                normalized_symbol = self._normalize_symbol(symbol)

                if has_closed:
                    orders = await exchange.fetch_closed_orders(
                        symbol=normalized_symbol,
                        since=since_ms,
                        limit=limit,
                    )
                else:
                    orders = await exchange.fetch_orders(
                        symbol=normalized_symbol,
                        since=since_ms,
                        limit=limit,
                    )
                    # 过滤只保留已完成的
                    orders = [o for o in orders if o.get("status") in ("closed", "filled")]

                for order in orders:
                    order_info = {
                        "symbol": symbol,
                        "normalized_symbol": normalized_symbol,
                        "order_id": order.get("id"),
                        "type": order.get("type"),  # market/limit/stop_market/etc
                        "side": order.get("side"),  # buy/sell
                        "amount": order.get("amount"),
                        "filled": order.get("filled"),
                        "price": order.get("price"),
                        "average": order.get("average"),  # 平均成交价
                        "cost": order.get("cost"),
                        "fee": order.get("fee"),
                        "status": order.get("status"),
                        "timestamp": order.get("timestamp"),
                        "datetime": order.get("datetime"),
                        "reduce_only": order.get("reduceOnly"),
                        "info": order.get("info", {}),
                    }
                    all_orders.append(order_info)

            except Exception as e:
                cprint(f"Failed to fetch orders for {symbol}: {e}", "yellow")

        # 按时间倒序排列
        all_orders.sort(key=lambda o: o.get("timestamp", 0), reverse=True)

        return all_orders

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
