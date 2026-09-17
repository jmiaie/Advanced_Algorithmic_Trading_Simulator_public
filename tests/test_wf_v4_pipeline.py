"""Offline tests for the D9-B v4 walk-forward pipeline -- specifically the
two fixes an independent review found necessary in v3: the execution-
timing look-ahead (decide-and-fill at the same bar-t close) and the
omitted turnover tie-break criterion.

All fixtures here are SYNTHETIC (seeded random walks / manufactured
series) -- for software verification only. No real market data is used or
implied; none of these numbers are research claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb_engine.wf_v3_backtest import (
    SpreadSeries,
    causal_zscore,
    generate_positions,
)
from stat_arb_engine.wf_v3_costs import BASE
from stat_arb_engine.wf_v4_backtest import lag_for_execution, simulate_pair_backtest
from stat_arb_engine.wf_v4_orch import _select_best_by_tiebreak


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-02", periods=n)


# --------------------------------------------------------- lag mechanics ---


def test_lag_for_execution_shifts_positions_and_hedge_ratio_by_one_bar():
    idx = _idx(6)
    positions = pd.Series([0, 1, 1, -1, -1, 0], index=idx)
    hedge_ratio = pd.Series([1.00, 1.01, 1.02, 1.03, 1.04, 1.05], index=idx)

    lagged_positions, lagged_hedge_ratio = lag_for_execution(positions, hedge_ratio)

    # Bar t's lagged value equals bar (t-1)'s decision-time value.
    for i in range(1, len(idx)):
        assert lagged_positions.iloc[i] == positions.iloc[i - 1]
        assert lagged_hedge_ratio.iloc[i] == pytest.approx(hedge_ratio.iloc[i - 1])
    # No prior decision exists before the first bar: flat by construction.
    assert lagged_positions.iloc[0] == 0


def test_t_signal_cannot_alter_t_holdings():
    """A position decided at bar t (using information through t) must not
    affect the state held/executed AT bar t -- only at t+1 onward."""
    idx = _idx(5)
    prices_y = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
    prices_x = pd.Series([50.0, 50.5, 51.0, 51.5, 52.0], index=idx)
    hedge_ratio = pd.Series([1.0] * 5, index=idx)
    spread = SpreadSeries(
        spread=prices_y - hedge_ratio * prices_x,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )

    # Two decision series identical everywhere except bar index 2, where
    # one decides "enter long" one bar earlier than the other.
    decisions_a = pd.Series([0, 0, 0, 1, 1], index=idx)
    decisions_b = pd.Series([0, 0, 1, 1, 1], index=idx)

    execution_index = idx[2:4]  # bars where the two decision series first differ + the next bar
    result_a = simulate_pair_backtest(
        prices_y=prices_y,
        prices_x=prices_x,
        spread=spread,
        decision_positions=decisions_a,
        execution_index=execution_index,
        allocated_nav=1_000_000.0,
        gross_notional_multiple=1.0,
        cost=BASE,
        pair_label="Y/X",
    )
    result_b = simulate_pair_backtest(
        prices_y=prices_y,
        prices_x=prices_x,
        spread=spread,
        decision_positions=decisions_b,
        execution_index=execution_index,
        allocated_nav=1_000_000.0,
        gross_notional_multiple=1.0,
        cost=BASE,
        pair_label="Y/X",
    )

    # At execution bar idx[2] (the bar where the raw decisions first
    # differ), both must hold the SAME state: the decision made "at" that
    # bar cannot affect that bar's own holdings -- only the bar before it
    # (idx[1], where both decision series agree: flat) can. Neither result
    # has entered a position by this first execution bar.
    assert result_a.equity_curve.iloc[0] == pytest.approx(result_b.equity_curve.iloc[0])
    assert all(f.date != idx[2] for f in result_a.fills)
    assert all(f.date != idx[2] for f in result_b.fills)
    # At execution bar idx[3], the two decision series now differ in their
    # t-1 value (decisions_a[2]=0 vs decisions_b[2]=1), so holdings DO
    # differ -- proving the lag, not same-bar leakage, is what matters:
    # result_b's decision made "at" raw index 2 only shows up executed one
    # bar later, at idx[3], never at idx[2] itself.
    assert result_a.equity_curve.iloc[1] != pytest.approx(result_b.equity_curve.iloc[1])
    assert len(result_b.fills) == 1
    assert result_b.fills[0].date == idx[3]
    assert len(result_a.fills) == 0


def test_kalman_style_hedge_ratio_execution_uses_prior_estimate_not_current():
    """The hedge ratio used to size a trade executing at bar t+1 must be
    the estimate as of t, not any later (t+1) update -- proven directly on
    lag_for_execution with a hedge ratio series that changes at every bar,
    as a sequential Kalman filter's beta path would."""
    idx = _idx(5)
    positions = pd.Series([0, 1, 1, 1, 1], index=idx)
    # Distinct value at every bar, as a forward-only Kalman filter would
    # produce (beta_t incorporates observation t).
    hedge_ratio = pd.Series([0.90, 0.95, 1.00, 1.05, 1.10], index=idx)

    _, lagged_hedge_ratio = lag_for_execution(positions, hedge_ratio)

    for i in range(1, len(idx)):
        # Executed at bar i, the hedge ratio must equal bar (i-1)'s
        # decision-time estimate -- never bar i's own (later) estimate.
        assert lagged_hedge_ratio.iloc[i] == pytest.approx(hedge_ratio.iloc[i - 1])
        assert lagged_hedge_ratio.iloc[i] != pytest.approx(hedge_ratio.iloc[i])


def test_no_future_bar_affects_prior_decisions():
    """Corrupting many bars near the end of a price series must not change
    any decision (z-score / position) computed at an earlier bar -- not
    just the immediately-next bar, but any later one."""
    rng = np.random.default_rng(0)
    idx = _idx(200)
    base = 100 + np.cumsum(rng.normal(0, 0.5, 200))

    prices_a = pd.Series(base, index=idx)
    prices_b = prices_a.copy()
    # Corrupt every bar in the back half of the series.
    corruption = rng.normal(0, 50.0, 100)
    prices_b.iloc[100:] = prices_b.iloc[100:].to_numpy() + corruption

    hedge_ratio = pd.Series(1.0, index=idx)
    spread_a = SpreadSeries(
        spread=prices_a,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )
    spread_b = SpreadSeries(
        spread=prices_b,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )

    z_a = causal_zscore(spread_a.spread, window=20)
    z_b = causal_zscore(spread_b.spread, window=20)
    positions_a = generate_positions(z_a, entry_z=2.0, exit_abs_z=0.5)
    positions_b = generate_positions(z_b, entry_z=2.0, exit_abs_z=0.5)

    # Every bar strictly before the corrupted region must be bit-identical.
    pd.testing.assert_series_equal(z_a.iloc[:100], z_b.iloc[:100])
    pd.testing.assert_series_equal(positions_a.iloc[:100], positions_b.iloc[:100])


def test_changing_price_at_t_plus_1_cannot_change_signal_decided_at_t():
    rng = np.random.default_rng(1)
    idx = _idx(100)
    base = 100 + np.cumsum(rng.normal(0, 0.5, 100))
    prices_a = pd.Series(base, index=idx)
    prices_b = prices_a.copy()
    prices_b.iloc[50:] = prices_b.iloc[50:] + 5.0  # perturb bar 50 onward only

    hedge_ratio = pd.Series(1.0, index=idx)
    spread_a = SpreadSeries(
        spread=prices_a,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )
    spread_b = SpreadSeries(
        spread=prices_b,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )
    z_a = causal_zscore(spread_a.spread, window=20)
    z_b = causal_zscore(spread_b.spread, window=20)

    # Bar 49 (the last bar strictly before the perturbation) must be
    # identical: its z-score/decision used information only through bar 49.
    pd.testing.assert_series_equal(z_a.iloc[:50], z_b.iloc[:50])


# ------------------------------------------------- drawdown calculation ---


def test_max_drawdown_reflects_loss_from_starting_capital_not_just_within_executed_bars():
    """A second, independently-flagged defect (distinct from the tie-break
    sign inversion covered below): the old running-peak calculation only
    ever looked at mark-to-market values starting from the first EXECUTED
    bar, with no baseline observation for NAV = allocated_nav "before" that
    bar. With only one executed bar, that meant `cummax()` over a
    single-point series trivially equals its own value, so ANY single-bar
    loss -- including the guaranteed nonzero entry cost of opening a
    position -- was reported as exactly 0% drawdown, regardless of the real
    loss versus starting capital. Entering a position costs money
    (commission/spread/slippage/impact), so a single executed bar with a
    fresh entry must show a strictly negative drawdown, not zero."""
    idx = _idx(2)
    prices_y = pd.Series([100.0, 100.0], index=idx)
    prices_x = pd.Series([50.0, 50.0], index=idx)
    hedge_ratio = pd.Series([1.0, 1.0], index=idx)
    spread = SpreadSeries(
        spread=prices_y - hedge_ratio * prices_x,
        hedge_ratio=hedge_ratio,
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )
    # Decided long at bar 0 so the lagged (executed) position at bar 1 is 1.
    decision_positions = pd.Series([1, 1], index=idx)

    result = simulate_pair_backtest(
        prices_y=prices_y,
        prices_x=prices_x,
        spread=spread,
        decision_positions=decision_positions,
        execution_index=idx[1:],  # a single executed bar
        allocated_nav=1_000_000.0,
        gross_notional_multiple=1.0,
        cost=BASE,
        pair_label="Y/X",
    )

    assert len(result.equity_curve) == 1
    assert result.equity_curve.iloc[0] < 0.0  # entry costs are strictly positive
    assert result.max_drawdown is not None
    assert result.max_drawdown == pytest.approx(
        result.equity_curve.iloc[0] / 1_000_000.0
    )
    assert result.max_drawdown < 0.0  # the old code always returned exactly 0.0 here


# -------------------------------------------------------- tie-break ---


def test_turnover_is_the_third_tiebreak_criterion():
    """When two candidates tie on Sharpe AND max drawdown, the one with
    lower turnover must win -- v3 explicitly omitted this; v4 must not."""
    from stat_arb_engine.wf_v4_backtest import BacktestResult

    tied_sharpe_and_dd_high_turnover = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=4,
        sharpe=1.0,
        max_drawdown=-0.05,
        turnover=2.0,
    )
    tied_sharpe_and_dd_low_turnover = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=2,
        sharpe=1.0,
        max_drawdown=-0.05,
        turnover=0.5,
    )
    candidates = [
        ({"label": "high_turnover"}, tied_sharpe_and_dd_high_turnover),
        ({"label": "low_turnover"}, tied_sharpe_and_dd_low_turnover),
    ]
    best = _select_best_by_tiebreak(candidates)
    assert best is not None
    assert best[0]["label"] == "low_turnover"


def test_tiebreak_falls_back_to_grid_order_on_exact_triple_tie():
    from stat_arb_engine.wf_v4_backtest import BacktestResult

    result_first = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=4,
        sharpe=1.0,
        max_drawdown=-0.05,
        turnover=1.0,
    )
    result_second = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=4,
        sharpe=1.0,
        max_drawdown=-0.05,
        turnover=1.0,
    )
    candidates = [
        ({"label": "first_in_grid_order"}, result_first),
        ({"label": "second_in_grid_order"}, result_second),
    ]
    best = _select_best_by_tiebreak(candidates)
    assert best is not None
    assert best[0]["label"] == "first_in_grid_order"


def test_drawdown_is_the_second_tiebreak_criterion_before_turnover():
    from stat_arb_engine.wf_v4_backtest import BacktestResult

    better_dd_worse_turnover = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=4,
        sharpe=1.0,
        max_drawdown=-0.02,
        turnover=5.0,
    )
    worse_dd_better_turnover = BacktestResult(
        estimation_mode="x",
        pair="A/B",
        n_bars=10,
        n_trades=4,
        sharpe=1.0,
        max_drawdown=-0.10,
        turnover=0.1,
    )
    candidates = [
        ({"label": "worse_dd_better_turnover"}, worse_dd_better_turnover),
        ({"label": "better_dd_worse_turnover"}, better_dd_worse_turnover),
    ]
    best = _select_best_by_tiebreak(candidates)
    assert best is not None
    assert best[0]["label"] == "better_dd_worse_turnover"
