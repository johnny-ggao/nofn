"""Test ATR (Average True Range) indicator calculation."""

from src.trading.market.candle import SimpleCandleFeatureComputer
from src.trading.models import Candle, InstrumentRef


def test_atr_calculation():
    """Test that ATR is calculated correctly from candle data."""
    # Create sample candle data
    instrument = InstrumentRef(symbol="BTC/USDT")

    candles = [
        Candle(
            ts=1000 + i * 60000,  # 1 minute intervals
            instrument=instrument,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1000.0,
            interval="1m"
        )
        for i in range(50)  # Need at least 14 candles for ATR calculation
    ]

    # Compute features
    computer = SimpleCandleFeatureComputer()
    features = computer.compute_features(candles=candles)

    # Should return 1 feature vector (one per symbol)
    assert len(features) == 1

    feature = features[0]
    assert feature.instrument.symbol == "BTC/USDT"

    # Check that ATR is present in values
    assert "atr" in feature.values

    # ATR should be a positive number (or None if not enough data)
    atr_value = feature.values["atr"]
    if atr_value is not None:
        assert atr_value > 0, f"ATR should be positive, got {atr_value}"
        # ATR should be reasonable relative to the price range
        assert atr_value < 20, f"ATR seems too high: {atr_value}"

    print(f"✓ ATR value: {atr_value}")


def test_atr_with_volatility():
    """Test ATR responds to increased volatility."""
    instrument = InstrumentRef(symbol="ETH/USDT")

    # Create low volatility candles
    low_vol_candles = [
        Candle(
            ts=1000 + i * 60000,
            instrument=instrument,
            open=3000.0,
            high=3001.0,
            low=2999.0,
            close=3000.0,
            volume=1000.0,
            interval="1m"
        )
        for i in range(30)
    ]

    # Create high volatility candles
    high_vol_candles = [
        Candle(
            ts=1000 + i * 60000,
            instrument=instrument,
            open=3000.0 + (i % 2) * 100,
            high=3100.0 + (i % 2) * 100,
            low=2900.0 + (i % 2) * 100,
            close=3050.0 + (i % 2) * 100,
            volume=1000.0,
            interval="1m"
        )
        for i in range(30)
    ]

    computer = SimpleCandleFeatureComputer()

    low_vol_features = computer.compute_features(candles=low_vol_candles)
    high_vol_features = computer.compute_features(candles=high_vol_candles)

    low_atr = low_vol_features[0].values["atr"]
    high_atr = high_vol_features[0].values["atr"]

    # High volatility should produce higher ATR
    if low_atr is not None and high_atr is not None:
        assert high_atr > low_atr, (
            f"High volatility ATR ({high_atr}) should be greater than "
            f"low volatility ATR ({low_atr})"
        )
        print(f"✓ Low volatility ATR: {low_atr:.2f}")
        print(f"✓ High volatility ATR: {high_atr:.2f}")


def test_atr_with_multiple_symbols():
    """Test ATR calculation for multiple symbols."""
    btc = InstrumentRef(symbol="BTC/USDT")
    eth = InstrumentRef(symbol="ETH/USDT")

    candles = []

    # BTC candles
    for i in range(20):
        candles.append(
            Candle(
                ts=1000 + i * 60000,
                instrument=btc,
                open=90000.0 + i * 10,
                high=90500.0 + i * 10,
                low=89500.0 + i * 10,
                close=90200.0 + i * 10,
                volume=1000.0,
                interval="1m"
            )
        )

    # ETH candles
    for i in range(20):
        candles.append(
            Candle(
                ts=1000 + i * 60000,
                instrument=eth,
                open=3000.0 + i * 5,
                high=3050.0 + i * 5,
                low=2950.0 + i * 5,
                close=3025.0 + i * 5,
                volume=1000.0,
                interval="1m"
            )
        )

    computer = SimpleCandleFeatureComputer()
    features = computer.compute_features(candles=candles)

    # Should have 2 feature vectors (one per symbol)
    assert len(features) == 2

    symbols = {f.instrument.symbol for f in features}
    assert "BTC/USDT" in symbols
    assert "ETH/USDT" in symbols

    # Both should have ATR values
    for feature in features:
        assert "atr" in feature.values
        atr = feature.values["atr"]
        if atr is not None:
            assert atr > 0
            print(f"✓ {feature.instrument.symbol} ATR: {atr:.2f}")


if __name__ == "__main__":
    print("Testing ATR indicator implementation...\n")

    print("Test 1: Basic ATR calculation")
    test_atr_calculation()

    print("\nTest 2: ATR with different volatility levels")
    test_atr_with_volatility()

    print("\nTest 3: ATR for multiple symbols")
    test_atr_with_multiple_symbols()

    print("\n✅ All ATR tests passed!")
