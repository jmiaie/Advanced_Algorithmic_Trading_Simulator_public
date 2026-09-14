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
        "price": 99.0,
        "timestamp": pd.Timestamp("2024-01-02"),
    }
    event_b_2 = {
        "type": "MARKET",
        "symbol": "B",
        "price": 100.0,
        "timestamp": pd.Timestamp("2024-01-02"),
    }
    event_a_3 = {
        "type": "MARKET",
        "symbol": "A",
        "price": 95.0,
        "timestamp": pd.Timestamp("2024-01-03"),
    }
    event_b_3 = {
        "type": "MARKET",
        "symbol": "B",
        "price": 100.0,
        "timestamp": pd.Timestamp("2024-01-03"),
    }

    assert strategy.calculate_signals(event_a_1) == []
    assert strategy.calculate_signals(event_b_1) == []
    assert strategy.calculate_signals(event_a_2) == []
    assert strategy.calculate_signals(event_b_2) == []
    assert strategy.calculate_signals(event_a_3) == []
    orders = strategy.calculate_signals(event_b_3)
    assert orders
    assert {order["timestamp"] for order in orders} == {pd.Timestamp("2024-01-03")}



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


def test_duplicate_timestamp_after_emitted_batch_is_rejected() -> None:
    buffer = SynchronizedMarketBuffer(["A", "B"])
    timestamp = pd.Timestamp("2024-01-01")
    assert buffer.push({"type": "MARKET", "symbol": "A", "price": 100.0, "timestamp": timestamp}) is None
    batch = buffer.push({"type": "MARKET", "symbol": "B", "price": 100.0, "timestamp": timestamp})
    assert batch is not None
    with pytest.raises(ValueError, match="Duplicate synchronized timestamp"):
        buffer.push({"type": "MARKET", "symbol": "A", "price": 101.0, "timestamp": timestamp})


def test_missing_bar_drop_policy_discards_stale_leg_instead_of_signaling() -> None:
    buffer = SynchronizedMarketBuffer(["A", "B"], missing_bar_policy="drop")
    t1 = pd.Timestamp("2024-01-01")
    t2 = pd.Timestamp("2024-01-02")
    t3 = pd.Timestamp("2024-01-03")

    assert buffer.push({"type": "MARKET", "symbol": "A", "price": 100.0, "timestamp": t1}) is None
    batch = buffer.push({"type": "MARKET", "symbol": "B", "price": 200.0, "timestamp": t1})
    assert batch is not None
    assert batch["timestamp"] == t1

    assert buffer.push({"type": "MARKET", "symbol": "A", "price": 101.0, "timestamp": t2}) is None
    assert buffer.push({"type": "MARKET", "symbol": "B", "price": 203.0, "timestamp": t3}) is None
    batch = buffer.push({"type": "MARKET", "symbol": "A", "price": 102.0, "timestamp": t3})
    assert batch is not None
    assert batch["timestamp"] == t3
    assert batch["prices"] == {"A": 102.0, "B": 203.0}


def test_strategy_uses_only_trailing_spreads_for_signal_normalization() -> None:
    strategy = PairsTradingStrategy("A", "B", hedge_ratio=1.0, window=2, entry_z=0.5, exit_z=0.1)
    events = [
        {"type": "MARKET", "symbol": "A", "price": 100.0, "timestamp": pd.Timestamp("2024-01-01")},
        {"type": "MARKET", "symbol": "B", "price": 100.0, "timestamp": pd.Timestamp("2024-01-01")},
        {"type": "MARKET", "symbol": "A", "price": 100.0, "timestamp": pd.Timestamp("2024-01-02")},
        {"type": "MARKET", "symbol": "B", "price": 100.0, "timestamp": pd.Timestamp("2024-01-02")},
        {"type": "MARKET", "symbol": "A", "price": 110.0, "timestamp": pd.Timestamp("2024-01-03")},
        {"type": "MARKET", "symbol": "B", "price": 100.0, "timestamp": pd.Timestamp("2024-01-03")},
    ]

    for event in events[:-1]:
        assert strategy.calculate_signals(event) == []
    assert strategy.calculate_signals(events[-1]) == []


def test_data_streamer_rejects_duplicate_index() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        DataStreamer({"A": df, "B": df})
