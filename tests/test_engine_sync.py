import pandas as pd
import pytest

from stat_arb_engine.engine import DataStreamer, SynchronizedMarketBuffer
from stat_arb_engine.strategies import PairsTradingStrategy


def test_strategy_waits_for_same_timestamp_pair_before_signal() -> None:
    strategy = PairsTradingStrategy(
        "A",
        "B",
        hedge_ratio=1.0,
        window=2,
        entry_z=0.5,
        exit_z=0.1,
    )
    event_a_1 = {
        "type": "MARKET",
        "symbol": "A",
        "price": 100.0,
        "timestamp": pd.Timestamp("2024-01-01"),
    }
    event_b_1 = {
        "type": "MARKET",
        "symbol": "B",
        "price": 100.0,
        "timestamp": pd.Timestamp("2024-01-01"),
    }
    event_a_2 = {
        "type": "MARKET",
        "symbol": "A",
        "price": 95.0,
        "timestamp": pd.Timestamp("2024-01-02"),
    }
    event_b_2 = {
        "type": "MARKET",
        "symbol": "B",
        "price": 105.0,
        "timestamp": pd.Timestamp("2024-01-02"),
    }

    assert strategy.calculate_signals(event_a_1) == []
    assert strategy.calculate_signals(event_b_1) == []
    assert strategy.calculate_signals(event_a_2) == []
    orders = strategy.calculate_signals(event_b_2)
    assert orders
    assert {order["timestamp"] for order in orders} == {pd.Timestamp("2024-01-02")}



def test_duplicate_timestamp_validation() -> None:
    buffer = SynchronizedMarketBuffer(["A", "B"])
    first_event = {
        "type": "MARKET",
        "symbol": "A",
        "price": 100.0,
        "timestamp": pd.Timestamp("2024-01-01"),
    }
    duplicate_event = {
        "type": "MARKET",
        "symbol": "A",
        "price": 101.0,
        "timestamp": pd.Timestamp("2024-01-01"),
    }
    assert buffer.push(first_event) is None
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        buffer.push(duplicate_event)



def test_data_streamer_rejects_duplicate_index() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        DataStreamer({"A": df, "B": df})
