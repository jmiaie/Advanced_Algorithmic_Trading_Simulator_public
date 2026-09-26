"""Research-correctness regression tests.

Each group below guards one correctness property of the research engine:
1. Same-timestamp pair signals
2. Marked-to-market NAV
3. Decision-time pair research
4. Static-vs-sequential-Kalman distinction
5. Correct analytics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb_engine.analytics import (
    calculate_conventional_sharpe,
    calculate_max_drawdown,
    compute_strategy_analytics,
)
from stat_arb_engine.portfolio import PortfolioLedger
from stat_arb_engine.research import (
    as_of_frame,
    compare_hedge_models,
    fit_dynamic_kalman_hedge_ratio,
    fit_static_ols_hedge_ratio,
    run_walk_forward,
    select_pairs,
)
from stat_arb_engine.strategies import PairsTradingStrategy


def _make_fill(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    commission: float = 0.0,
    execution_cost: float = 0.0,
) -> dict:
    signed_qty = qty if side == "buy" else -qty
    gross_cash_flow = -signed_qty * price
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "commission": commission,
        "execution_cost": execution_cost,
        "net_cash_flow": gross_cash_flow - commission - execution_cost,
    }


# ---------------------------------------------------------------------------
# Gate 1: Same-timestamp pair signals
# ---------------------------------------------------------------------------


def test_gate1_no_signal_until_both_legs_share_timestamp() -> None:
    strategy = PairsTradingStrategy(
        "A",
        "B",
        hedge_ratio=1.0,
        window=2,
        entry_z=0.5,
        exit_z=0.1,
    )
    # Warm-up pair bars with non-zero trailing dispersion so day-3 can enter.
    for day, price_a, price_b in [
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 99.0, 100.0),
    ]:
        assert strategy.calculate_signals(
            {"type": "MARKET", "symbol": "A", "price": price_a, "timestamp": pd.Timestamp(day)}
        ) == []
        assert strategy.calculate_signals(
            {"type": "MARKET", "symbol": "B", "price": price_b, "timestamp": pd.Timestamp(day)}
        ) == []

    # Only leg A arrives for day 3: must not emit a pair signal.
    assert (
        strategy.calculate_signals(
            {
                "type": "MARKET",
                "symbol": "A",
                "price": 95.0,
                "timestamp": pd.Timestamp("2024-01-03"),
            }
        )
        == []
    )
    orders = strategy.calculate_signals(
        {
            "type": "MARKET",
            "symbol": "B",
            "price": 100.0,
            "timestamp": pd.Timestamp("2024-01-03"),
        }
    )
    assert orders
    assert {order["timestamp"] for order in orders} == {pd.Timestamp("2024-01-03")}
    assert {order["symbol"] for order in orders} == {"A", "B"}


def test_gate1_market_batch_rejects_incomplete_pair() -> None:
    strategy = PairsTradingStrategy("A", "B", hedge_ratio=1.0, window=2)
    with pytest.raises(ValueError, match="missing same-timestamp pair legs"):
        strategy.calculate_signals(
            {
                "type": "MARKET_BATCH",
                "timestamp": pd.Timestamp("2024-01-01"),
                "prices": {"A": 100.0},
            }
        )


# ---------------------------------------------------------------------------
# Gate 2: Marked-to-market NAV
# ---------------------------------------------------------------------------


def test_gate2_nav_equals_cash_plus_marked_positions() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(_make_fill("A", "buy", 100, 100.0, commission=1.0, execution_cost=4.0))
    ledger.process_fill(_make_fill("B", "sell", 50, 200.0, commission=1.0, execution_cost=4.0))
    snapshot = ledger.mark_to_market({"A": 105.0, "B": 190.0}, timestamp="2024-01-02")

    expected_nav = snapshot["cash"] + snapshot["net_market_value"]
    assert snapshot["nav"] == pytest.approx(expected_nav)
    assert snapshot["equity"] == pytest.approx(snapshot["nav"])
    assert snapshot["nav"] - ledger.starting_cash == pytest.approx(snapshot["net_pnl"])
    assert snapshot["unrealized_pnl"] == pytest.approx(500.0 + 500.0)


def test_gate2_nav_updates_when_marks_move_without_new_fills() -> None:
    ledger = PortfolioLedger(starting_cash=10_000.0)
    ledger.process_fill(_make_fill("A", "buy", 10, 100.0))
    first = ledger.mark_to_market({"A": 100.0}, timestamp="2024-01-01")
    second = ledger.mark_to_market({"A": 110.0}, timestamp="2024-01-02")
    assert first["nav"] == pytest.approx(10_000.0)
    assert second["nav"] == pytest.approx(10_100.0)
    assert second["unrealized_pnl"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Gate 3: Decision-time pair research
# ---------------------------------------------------------------------------


def test_gate3_as_of_frame_excludes_future_bars() -> None:
    frame = pd.DataFrame(
        {"A": range(5), "B": range(5, 10)},
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    sliced = as_of_frame(frame, "2024-01-03")
    assert sliced.index.max() == pd.Timestamp("2024-01-03")
    assert len(sliced) == 3


def test_gate3_pair_selection_ignores_post_decision_mutation() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=24, freq="D")
    a = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, len(index))), index=index)
    b = 1.2 * a + pd.Series(rng.normal(0.0, 0.05, len(index)), index=index)
    c = pd.Series(40.0 + np.cumsum(rng.normal(0.0, 3.0, len(index))), index=index)
    d = pd.Series(20.0 + np.cumsum(rng.normal(0.0, 3.0, len(index))), index=index)
    base = pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=index)

    decision_asof = index[11]
    baseline = select_pairs(as_of_frame(base, decision_asof))

    mutated = base.copy()
    # Corrupt only post-decision bars so a leaky selector would change rankings.
    mutated.loc[index[12]:, "C"] = mutated.loc[index[12]:, "A"] * 0.9
    mutated.loc[index[12]:, "D"] = mutated.loc[index[12]:, "A"] * 0.9 + 0.1
    future_safe = select_pairs(as_of_frame(mutated, decision_asof))

    assert not baseline.empty
    assert list(baseline["symbol_a"]) == list(future_safe["symbol_a"])
    assert list(baseline["symbol_b"]) == list(future_safe["symbol_b"])
    assert list(baseline["hedge_ratio"]) == pytest.approx(list(future_safe["hedge_ratio"]))


def test_gate3_walk_forward_records_decision_time_cutoffs() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range("2024-01-01", periods=18, freq="D")
    a = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, len(index))), index=index)
    b = 1.5 * a + pd.Series(rng.normal(0.0, 0.05, len(index)), index=index)
    frame = pd.DataFrame({"A": a, "B": b}, index=index)
    result = run_walk_forward(frame, formation_size=8, validation_size=3, test_size=2)
    assert not result.empty
    assert all(result["decision_time_as_of"] == result["formation_end"])
    assert all(result["pair_selection_end"] <= result["trained_through"])
    assert all(result["hedge_fit_end"] < result["oos_start"])
    assert all(result["selection_from_formation_only"])
    assert result["formation_hedge_ratio"].notna().all()


# ---------------------------------------------------------------------------
# Gate 4: Static-vs-sequential-Kalman distinction
# ---------------------------------------------------------------------------


def test_gate4_estimation_modes_are_explicitly_distinct() -> None:
    x = pd.Series(np.linspace(1.0, 20.0, 20))
    y = 0.5 + 1.5 * x
    static = fit_static_ols_hedge_ratio(y, x)
    kalman = fit_dynamic_kalman_hedge_ratio(
        y,
        x,
        process_variance=1e-6,
        observation_variance=1e-6,
    )
    assert static.estimation_mode == "static_batch_ols"
    assert kalman.estimation_mode == "sequential_filtered_kalman"
    assert static.estimation_mode != kalman.estimation_mode


def test_gate4_kalman_path_is_causal_unlike_batch_static_refit() -> None:
    x = pd.Series(np.arange(1.0, 16.0))
    y = 1.0 + 2.0 * x
    early = fit_dynamic_kalman_hedge_ratio(
        y.iloc[:8],
        x.iloc[:8],
        process_variance=1e-5,
        observation_variance=1e-3,
    )
    full = fit_dynamic_kalman_hedge_ratio(
        y,
        x,
        process_variance=1e-5,
        observation_variance=1e-3,
    )
    # Sequential filter: early path equals prefix of full path.
    assert list(early.beta_path.values) == pytest.approx(
        list(full.beta_path.iloc[:8].values),
        abs=1e-10,
    )

    # Static batch OLS on a longer sample is a new full-sample estimate, not a
    # causal path extension of the shorter-sample fit.
    static_early = fit_static_ols_hedge_ratio(y.iloc[:8], x.iloc[:8])
    static_full = fit_static_ols_hedge_ratio(y, x)
    assert static_early.slope == pytest.approx(static_full.slope, abs=1e-6)
    # Modes remain distinct even when slopes numerically agree on a clean toy.
    comparison = compare_hedge_models(y, x)
    assert set(comparison["static_estimation_mode"]) == {"static_batch_ols"}
    assert set(comparison["dynamic_estimation_mode"]) == {"sequential_filtered_kalman"}


def test_gate4_compare_rejects_mismatched_test_index() -> None:
    x = pd.Series(np.arange(1.0, 6.0))
    y = 1.0 + 2.0 * x
    with pytest.raises(ValueError, match="test_index length must match"):
        compare_hedge_models(y, x, test_index=x.index[:2])


# ---------------------------------------------------------------------------
# Gate 5: Correct analytics
# ---------------------------------------------------------------------------


def test_gate5_sharpe_and_drawdown_match_explicit_formulas() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01])
    expected_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert calculate_conventional_sharpe(returns, periods_per_year=252) == pytest.approx(
        expected_sharpe
    )

    nav = pd.concat([pd.Series([1.0]), (1.0 + returns).cumprod()], ignore_index=True)
    expected_mdd = float((nav / nav.cummax() - 1.0).min())
    assert calculate_max_drawdown(returns) == pytest.approx(expected_mdd)


def test_gate5_net_return_uses_nav_and_gross_adds_back_costs() -> None:
    equity = [10_000.0, 9_900.0, 10_050.0]
    returns = pd.Series(equity).pct_change().fillna(0.0)
    metrics = compute_strategy_analytics(
        equity_curve=equity,
        returns=returns,
        transaction_costs=[0.0, 25.0, 25.0],
    )
    assert metrics["net_return"] == pytest.approx((10_050.0 / 10_000.0) - 1.0)
    assert metrics["cost_drag"] == pytest.approx(50.0)
    assert metrics["gross_return"] == pytest.approx(((10_050.0 + 50.0) / 10_000.0) - 1.0)
    assert metrics["gross_return"] > metrics["net_return"]


def test_gate5_zero_cost_keeps_gross_equal_to_net() -> None:
    equity = [100.0, 110.0]
    metrics = compute_strategy_analytics(
        equity_curve=equity,
        returns=[0.0, 0.10],
        transaction_costs=[0.0, 0.0],
    )
    assert metrics["gross_return"] == pytest.approx(metrics["net_return"])
