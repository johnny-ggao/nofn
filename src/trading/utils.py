"""Utility functions for trading operations."""

from typing import Dict, List, Optional, Tuple

import ccxt.pro as ccxtpro
from termcolor import cprint

from .constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from .models import FeatureVector


async def fetch_free_cash_from_gateway(
    execution_gateway, symbols: List[str]
) -> Tuple[float, float]:
    """Fetch exchange balance via `execution_gateway.fetch_balance()` and
    aggregate free cash for the given `symbols` (quote currencies).

    Returns aggregated free cash as float. Returns 0.0 on error or when
    balance shape cannot be parsed.
    """
    cprint("Fetching exchange balance for LIVE trading mode", "white")
    try:
        if not hasattr(execution_gateway, "fetch_balance"):
            return 0.0, 0.0
        balance = await execution_gateway.fetch_balance()
    except Exception:
        return 0.0, 0.0

    cprint(f"Raw balance response: {balance}", "white")
    free_map: Dict[str, float] = {}
    try:
        free_section = balance.get("free") if isinstance(balance, dict) else None
    except Exception:
        free_section = None

    if isinstance(free_section, dict):
        free_map = {str(k).upper(): float(v or 0.0) for k, v in free_section.items()}
    else:
        iterable = balance.items() if isinstance(balance, dict) else []
        for k, v in iterable:
            if isinstance(v, dict) and "free" in v:
                try:
                    free_map[str(k).upper()] = float(v.get("free") or 0.0)
                except Exception:
                    continue

    cprint(f"Parsed free balance map: {free_map}", "white")
    quotes: List[str] = []
    for sym in symbols or []:
        s = str(sym).upper()
        if "/" in s and len(s.split("/")) == 2:
            quotes.append(s.split("/")[1])
        elif "-" in s and len(s.split("-")) == 2:
            quotes.append(s.split("-")[1])

    quotes = list(dict.fromkeys(quotes))
    cprint(f"Quote currencies from symbols: {quotes}", "white")

    free_cash = 0.0
    total_cash = 0.0

    if quotes:
        for q in quotes:
            free_cash += float(free_map.get(q, 0.0) or 0.0)
            q_data = balance.get(q)
            if isinstance(q_data, dict):
                total_cash += float(q_data.get("total", 0.0) or 0.0)
            else:
                total_cash += float(free_map.get(q, 0.0) or 0.0)
    else:
        for q in ("USDT", "USD", "USDC"):
            free_cash += float(free_map.get(q, 0.0) or 0.0)
            q_data = balance.get(q)
            if isinstance(q_data, dict):
                total_cash += float(q_data.get("total", 0.0) or 0.0)
            else:
                total_cash += float(free_map.get(q, 0.0) or 0.0)

    cprint(
        f"Synced balance from exchange: free_cash={free_cash}, total_cash={total_cash}, quotes={quotes}",
        "white"
    )

    return float(free_cash), float(total_cash)


def extract_market_snapshot_features(
    features: List[FeatureVector],
) -> List[FeatureVector]:
    """Extract market snapshot feature vectors."""
    snapshot_features: List[FeatureVector] = []

    for item in features:
        if not isinstance(item, FeatureVector):
            continue

        meta = item.meta or {}
        group_key = meta.get(FEATURE_GROUP_BY_KEY)
        if group_key != FEATURE_GROUP_BY_MARKET_SNAPSHOT:
            continue

        snapshot_features.append(item)

    return snapshot_features


def extract_price_map(features: List[FeatureVector]) -> Dict[str, float]:
    """Extract symbol -> price map from market snapshot feature vectors."""
    price_map: Dict[str, float] = {}

    for item in features:
        if not isinstance(item, FeatureVector):
            continue

        meta = item.meta or {}
        group_key = meta.get(FEATURE_GROUP_BY_KEY)
        if group_key != FEATURE_GROUP_BY_MARKET_SNAPSHOT:
            continue

        instrument = getattr(item, "instrument", None)
        symbol = getattr(instrument, "symbol", None)
        if not symbol:
            continue

        values = item.values or {}
        price = (
            values.get("price.last")
            or values.get("price.close")
            or values.get("price.mark")
            or values.get("funding.mark_price")
        )
        if price is None:
            continue

        try:
            price_map[symbol] = float(price)
        except (TypeError, ValueError):
            cprint(f"Failed to parse feature price for {symbol}", "yellow")

    return price_map


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format for CCXT.

    Examples:
        BTC-USD -> BTC/USD:USD
        BTC-USDT -> BTC/USDT:USDT
        BTC/USDT -> BTC/USDT:USDT
    """
    base_symbol = symbol.replace("-", "/")

    if ":" not in base_symbol:
        parts = base_symbol.split("/")
        if len(parts) == 2:
            base_symbol = f"{parts[0]}/{parts[1]}:{parts[1]}"

    return base_symbol


def get_exchange_cls(exchange_id: str):
    """Get CCXT exchange class by exchange ID."""
    exchange_cls = getattr(ccxtpro, exchange_id, None)
    if exchange_cls is None:
        raise RuntimeError(f"Exchange '{exchange_id}' not found in ccxt.pro")
    return exchange_cls


def prune_none(obj):
    """Recursively remove None, empty dict, and empty list values."""
    if isinstance(obj, dict):
        pruned = {k: prune_none(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in pruned.items() if v not in (None, {}, [])}
    if isinstance(obj, list):
        pruned = [prune_none(v) for v in obj]
        return [v for v in pruned if v not in (None, {}, [])]
    return obj


def extract_market_section(market_data: List[Dict]) -> Dict:
    """Extract decision-critical metrics from market feature entries."""
    compact: Dict[str, Dict] = {}
    for item in market_data:
        symbol = (item.get("instrument") or {}).get("symbol")
        if not symbol:
            continue

        values = item.get("values") or {}
        entry: Dict[str, float] = {}

        for feature_key, alias in (
            ("price.last", "last"),
            ("price.close", "close"),
            ("price.open", "open"),
            ("price.high", "high"),
            ("price.low", "low"),
            ("price.bid", "bid"),
            ("price.ask", "ask"),
            ("price.change_pct", "change_pct"),
            ("price.volume", "volume"),
        ):
            if feature_key in values and values[feature_key] is not None:
                entry[alias] = values[feature_key]

        if values.get("open_interest") is not None:
            entry["open_interest"] = values["open_interest"]

        if values.get("funding.rate") is not None:
            entry["funding_rate"] = values["funding.rate"]
        if values.get("funding.mark_price") is not None:
            entry["mark_price"] = values["funding.mark_price"]

        normalized = {k: v for k, v in entry.items() if v is not None}
        if normalized:
            compact[symbol] = normalized

    return compact


def group_features(features: List[FeatureVector]) -> Dict:
    """Organize features by grouping metadata."""
    grouped: Dict[str, List] = {}

    for fv in features:
        data = fv.model_dump(mode="json")
        meta = data.get("meta") or {}
        group_key = meta.get(FEATURE_GROUP_BY_KEY)

        if not group_key:
            continue

        grouped.setdefault(group_key, []).append(data)

    return grouped
