"""Market snapshot feature computation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..constants import (
    FEATURE_GROUP_BY_KEY,
    FEATURE_GROUP_BY_MARKET_SNAPSHOT,
)
from ..models import FeatureVector, InstrumentRef, get_current_timestamp_ms
from .feature_interfaces import MarketSnapshotFeatureComputer as BaseMarketSnapshotFeatureComputer


class SimpleMarketSnapshotFeatureComputer(BaseMarketSnapshotFeatureComputer):
    """Convert exchange market_snapshot structures into FeatureVector items.

    Extracts price, open interest, and funding rate data from raw exchange
    responses and normalizes them into FeatureVector format.
    """

    def __init__(self, exchange_id: Optional[str] = None) -> None:
        self._exchange_id = exchange_id or "binance"

    def compute_features(
        self,
        snapshot: Dict[str, Dict[str, Any]],
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        """Compute features from market snapshot.

        Args:
            snapshot: Dict mapping symbol to market data
            meta: Optional metadata to include in feature vectors

        Returns:
            List of FeatureVector objects
        """
        features: List[FeatureVector] = []
        now_ts = get_current_timestamp_ms()

        for symbol, data in (snapshot or {}).items():
            if not isinstance(data, dict):
                continue

            price_obj = data.get("price") if isinstance(data, dict) else None
            timestamp = None
            values: Dict[str, float] = {}

            if isinstance(price_obj, dict):
                timestamp = price_obj.get("timestamp") or price_obj.get("ts")
                for key in ("last", "close", "open", "high", "low", "bid", "ask"):
                    val = price_obj.get(key)
                    if val is not None:
                        try:
                            values[f"price.{key}"] = float(val)
                        except (TypeError, ValueError):
                            continue

                change = price_obj.get("percentage")
                if change is not None:
                    try:
                        values["price.change_pct"] = float(change)
                    except (TypeError, ValueError):
                        pass

                volume = price_obj.get("quoteVolume") or price_obj.get("baseVolume")
                if volume is not None:
                    try:
                        values["price.volume"] = float(volume)
                    except (TypeError, ValueError):
                        pass

            # Extract open interest
            if isinstance(data.get("open_interest"), dict):
                oi = data["open_interest"]
                for field in ("openInterest", "openInterestAmount", "baseVolume"):
                    val = oi.get(field)
                    if val is not None:
                        try:
                            values["open_interest"] = float(val)
                        except (TypeError, ValueError):
                            pass
                        break

            # Extract funding rate
            if isinstance(data.get("funding_rate"), dict):
                fr = data["funding_rate"]
                rate = fr.get("fundingRate") or fr.get("funding_rate")
                if rate is not None:
                    try:
                        values["funding.rate"] = float(rate)
                    except (TypeError, ValueError):
                        pass
                mark_price = fr.get("markPrice") or fr.get("mark_price")
                if mark_price is not None:
                    try:
                        values["funding.mark_price"] = float(mark_price)
                    except (TypeError, ValueError):
                        pass

            if not values:
                continue

            fv_ts = int(timestamp) if timestamp is not None else now_ts
            fv_meta = {FEATURE_GROUP_BY_KEY: FEATURE_GROUP_BY_MARKET_SNAPSHOT}
            if meta:
                for k, v in meta.items():
                    fv_meta.setdefault(k, v)

            feature = FeatureVector(
                ts=int(fv_ts),
                instrument=InstrumentRef(symbol=symbol, exchange_id=self._exchange_id),
                values=values,
                meta=fv_meta,
            )
            features.append(feature)

        return features
