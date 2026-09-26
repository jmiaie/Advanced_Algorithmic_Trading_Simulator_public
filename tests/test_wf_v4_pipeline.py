"""Offline tests for the v4 walk-forward pipeline -- specifically the
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


# ------------------------------------------------ validation grid search ---


def test_validation_grid_search_returns_none_when_no_candidate_clears_min_trades():
    """Regression/coverage test: nothing in the test suite exercised either
    branch of _validation_grid_search_signal_params before this (flagged by
    independent review). This is also the ACTUAL, currently-observed path in
    production: all 4 qualifying dev_formation windows in the real study
    have used_fallback=True, meaning this function returned None for every
    one of them. A flat spread with zero variance can never cross any
    entry_z threshold, so it must return (None, None) rather than silently
    selecting an arbitrary candidate."""
    from stat_arb_engine.wf_v3_costs import BASE
    from stat_arb_engine.wf_v4_orch import _validation_grid_search_signal_params

    idx = _idx(300)
    n_formation = 200
    # Formation: a small deterministic wiggle (not perfectly flat -- a
    # zero-variance regressor makes statsmodels' add_constant skip adding a
    # constant column at all, an unrelated edge case this test isn't about)
    # so the static OLS hedge-ratio fit behaves normally, landing close to
    # slope=1 on a Y that tracks X plus a tiny deterministic offset.
    wiggle = 0.01 * np.sin(np.arange(n_formation) / 5.0)
    prices_x_formation = 50.0 + wiggle
    prices_y_formation = 100.0 + wiggle
    # Validation: perfectly flat, so the spread has zero variance and its
    # z-score is undefined (zero rolling std) -> no entry ever fires.
    prices_x_validation = np.full(100, 50.0)
    prices_y_validation = np.full(100, 100.0)

    prices_x = np.concatenate([prices_x_formation, prices_x_validation])
    prices_y = np.concatenate([prices_y_formation, prices_y_validation])
    combined = pd.DataFrame({"Y": prices_y, "X": prices_x}, index=idx)
    formation_prices = combined.iloc[:n_formation]
    validation_index = idx[n_formation : n_formation + 80]

    params, sharpe = _validation_grid_search_signal_params(
        formation_prices=formation_prices,
        combined_for_spread=combined,
        validation_index=validation_index,
        sym_a="Y",
        sym_b="X",
        grid_entry_z=(1.5, 2.0, 2.5),
        grid_exit_abs_z=(0.25, 0.50, 0.75),
        grid_z_window=(20, 40, 60),
        allocated_nav=1_000_000.0,
        gross_notional_multiple=1.0,
        cost=BASE,
    )
    assert params is None
    assert sharpe is None


def test_validation_grid_search_selects_a_candidate_when_trades_clear_the_threshold():
    """Complements the no-candidate test above: an oscillating spread that
    reliably crosses entry/exit thresholds many times over the validation
    window must return a real (non-None) selected parameter set, proving
    the grid-search success path is not dead code."""
    from stat_arb_engine.wf_v3_costs import BASE
    from stat_arb_engine.wf_v4_orch import _validation_grid_search_signal_params

    idx = _idx(500)
    # Formation: a small deterministic wiggle (not perfectly flat -- see the
    # no-candidate test above for why), so the static OLS hedge ratio is a
    # clean fit (slope ~= 1) with no large spikes to contaminate it.
    n_formation = 200
    n_validation = 300
    wiggle = 0.01 * np.sin(np.arange(n_formation) / 5.0)
    prices_y = np.concatenate([100.0 + wiggle, np.zeros(n_validation)])
    prices_x = np.concatenate([50.0 + wiggle, np.zeros(n_validation)])
    # Validation: mostly-flat spread with a sign-alternating spike every 10
    # bars. A rolling std dominated by the mostly-flat majority stays small,
    # so each spike's z-score comfortably clears any grid entry_z, and the
    # very next (flat, z~0) bar comfortably clears any grid exit_abs_z --
    # 30 clean round trips over the window, verified empirically to clear
    # MIN_VALIDATION_TRADES for the (entry_z=1.5, exit_abs_z=0.25) cell.
    spikes = np.zeros(n_validation)
    spikes[0::10] = [30.0 if (i // 10) % 2 == 0 else -30.0 for i in range(0, n_validation, 10)]
    prices_y[n_formation:] = 100.0 + spikes
    prices_x[n_formation:] = 50.0

    combined = pd.DataFrame({"Y": prices_y, "X": prices_x}, index=idx)
    formation_prices = combined.iloc[:n_formation]
    validation_index = idx[n_formation:]

    params, sharpe = _validation_grid_search_signal_params(
        formation_prices=formation_prices,
        combined_for_spread=combined,
        validation_index=validation_index,
        sym_a="Y",
        sym_b="X",
        grid_entry_z=(1.5, 2.0, 2.5),
        grid_exit_abs_z=(0.25, 0.50, 0.75),
        grid_z_window=(20,),
        allocated_nav=1_000_000.0,
        gross_notional_multiple=1.0,
        cost=BASE,
    )
    assert params is not None
    assert set(params) == {"entry_z", "exit_abs_z", "trailing_z_window"}


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


# ------------------------------------------- frozen holdout hyperparameters ---


def test_frozen_signal_params_bypasses_both_selection_functions():
    """Regression test for a real, independently-flagged defect: the
    holdout path must NOT call _validation_grid_search_signal_params or
    _select_kalman_process_variance at all -- this config's own
    no_retune_after_freeze / no_2025_access_before_final_configuration_frozen
    constraints require entry_z/exit_abs_z/trailing_z_window/
    kalman_process_variance to be selected ONCE from pre-2025 DEV/2024-
    validation data and then held fixed, not reselected inside each 2025
    window from that window's own preceding (partly-2025) validation slice.
    pair_rediscovery_per_window governs pair discovery only, not this.

    Proven here by monkeypatching both selection functions to raise if
    called, using data deliberately engineered so that if either function
    DID run, it would find candidates clearing MIN_VALIDATION_TRADES (i.e.
    the test doesn't pass merely because nothing qualifies) -- confirmed by
    running the same fixture with frozen_signal_params=None first."""
    import stat_arb_engine.wf_v4_orch as orch
    from stat_arb_engine.wf_v3_pairs import PairSelectionResult

    idx = _idx(700)
    n_formation = 200
    n_validation_and_test = 500
    n_total = n_formation + n_validation_and_test
    # A persistent small wiggle across the FULL series (not just formation)
    # keeps prices_x from ever being exactly flat -- with 3 rolling windows
    # here (run_walk_forward_study walks forward, unlike the single-window
    # grid-search tests above), a LATER window's formation slice can land
    # entirely past n_formation, and a flat regressor there degenerates
    # statsmodels' add_constant (it skips adding a constant column at all),
    # which is an unrelated edge case this test isn't about.
    wiggle = 0.01 * np.sin(np.arange(n_total) / 5.0)
    prices_y = 100.0 + wiggle
    prices_x = 50.0 + wiggle
    spikes = np.zeros(n_validation_and_test)
    spikes[0::10] = [
        30.0 if (i // 10) % 2 == 0 else -30.0 for i in range(0, n_validation_and_test, 10)
    ]
    prices_y[n_formation:] += spikes
    combined_prices = pd.DataFrame({"Y": prices_y, "X": prices_x}, index=idx)
    group_membership = {"Y": "GROUP_A", "X": "GROUP_A"}

    def _fake_select_pair(formation_prices, group_membership, **kwargs):
        return PairSelectionResult(
            n_candidate_tests=1,
            n_fdr_survivors=1,
            selected={"symbol_a": "Y", "symbol_b": "X"},
            all_candidates=pd.DataFrame(),
        )

    common_kwargs = dict(
        combined_prices=combined_prices,
        group_membership=group_membership,
        formation_size=n_formation,
        validation_size=200,
        test_size=100,
        step_size=100,
        grid_entry_z=(1.5, 2.0, 2.5),
        grid_exit_abs_z=(0.25, 0.50, 0.75),
        grid_z_window=(20,),
        grid_kalman_process_variance=(1e-4,),
    )

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(orch, "select_pair_within_groups", _fake_select_pair)

        # Sanity: with selection ALLOWED (frozen_signal_params=None), at
        # least one window must actually reach a non-fallback or fallback
        # selection without erroring, proving the fixture is well-formed.
        baseline_study = orch.run_walk_forward_study(**common_kwargs)
        assert any(not w.no_trade for w in baseline_study.windows)

        # Now the actual regression check: force both selection functions
        # to raise, and confirm frozen_signal_params avoids calling them.
        def _boom(*args, **kwargs):
            raise AssertionError(
                "holdout path must not call this -- hyperparameters must be frozen"
            )

        mp.setattr(orch, "_validation_grid_search_signal_params", _boom)
        mp.setattr(orch, "_select_kalman_process_variance", _boom)

        frozen = dict(orch.FALLBACK_PARAMS)
        study = orch.run_walk_forward_study(frozen_signal_params=frozen, **common_kwargs)

    qualifying = [w for w in study.windows if not w.no_trade]
    assert qualifying  # the fixture must actually reach a qualifying window
    for w in qualifying:
        assert w.used_fallback is True
        assert w.validation_ols_sharpe is None
        assert w.validation_kalman_sharpe is None
        assert w.frozen_params == {
            "entry_z": frozen["entry_z"],
            "exit_abs_z": frozen["exit_abs_z"],
            "trailing_z_window": frozen["trailing_z_window"],
            "kalman_process_variance": frozen["kalman_process_variance"],
        }


# --------------------- Holdout requirements 6 and 7 (pre-registered spec) ---------------------


def test_holdout_pair_selection_receives_formation_only_data():
    """Pre-registered spec requirement 6 ('pair selection remains
    formation-only'): a dedicated regression proving the ORCHESTRATOR --
    not select_pair_within_groups's own cointegration logic, which is
    outside this module's scope -- only ever passes each window's own
    formation-period slice into pair selection, in frozen_signal_params
    (holdout) mode. No date from that window's own validation or test
    (2025) period may appear in the frame handed to selection."""
    import stat_arb_engine.wf_v4_orch as orch
    from stat_arb_engine.research import walk_forward_windows

    idx = _idx(700)
    rng = np.random.default_rng(2)
    combined_prices = pd.DataFrame(
        {
            "Y": 100 + np.cumsum(rng.normal(0, 0.5, len(idx))),
            "X": 50 + np.cumsum(rng.normal(0, 0.5, len(idx))),
        },
        index=idx,
    )
    group_membership = {"Y": "GROUP_A", "X": "GROUP_A"}
    window_kwargs = dict(formation_size=200, validation_size=200, test_size=100, step_size=100)
    expected_windows = walk_forward_windows(combined_prices, **window_kwargs)
    assert len(expected_windows) >= 2  # the fixture must exercise more than one window

    captured_formation_indices: list[pd.Index] = []
    real_select = orch.select_pair_within_groups

    def _spy_select(formation_prices, *args, **kwargs):
        captured_formation_indices.append(formation_prices.index)
        return real_select(formation_prices, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orch, "select_pair_within_groups", _spy_select)
        orch.run_walk_forward_study(
            combined_prices=combined_prices,
            group_membership=group_membership,
            formation_size=200,
            validation_size=200,
            test_size=100,
            step_size=100,
            grid_entry_z=(1.5,),
            grid_exit_abs_z=(0.25,),
            grid_z_window=(20,),
            grid_kalman_process_variance=(1e-4,),
            frozen_signal_params=dict(orch.FALLBACK_PARAMS),
        )

    assert len(captured_formation_indices) == len(expected_windows)
    for captured_index, expected_window in zip(
        captured_formation_indices, expected_windows, strict=True
    ):
        assert captured_index.equals(expected_window["formation"])
        assert captured_index.max() < expected_window["validation"].min()
        assert captured_index.max() < expected_window["test"].min()


def test_holdout_signal_unaffected_by_perturbing_that_windows_own_later_test_bars():
    """Pre-registered spec requirement 7 ('no test-window future information
    affects its own decision'): perturbing the LATTER portion of a
    window's own 2025 test period must not change the causal z-score fed
    into position generation for the EARLIER portion of that same test
    period. This exercises the actual orchestrator wiring (formation-fit
    static spread + continuous forward-only Kalman filter run across
    formation+validation+test, per build_static_spread's and
    build_kalman_spread's own docstrings) end-to-end, complementing the
    lower-level causal_zscore/generate_positions unit tests above with a
    holdout-mode-specific integration proof."""
    import stat_arb_engine.wf_v4_orch as orch
    from stat_arb_engine.wf_v3_pairs import PairSelectionResult

    idx = _idx(500)
    n_formation = 200
    wiggle = 0.01 * np.sin(np.arange(len(idx)) / 5.0)
    base_y = 100.0 + wiggle
    base_x = 50.0 + wiggle
    tail = len(idx) - n_formation
    spikes = np.zeros(tail)
    spikes[0::10] = [30.0 if (i // 10) % 2 == 0 else -30.0 for i in range(0, tail, 10)]
    base_y[n_formation:] += spikes

    combined_baseline = pd.DataFrame({"Y": base_y.copy(), "X": base_x.copy()}, index=idx)
    combined_perturbed = combined_baseline.copy()
    # Window 1 (formation=idx[100:300], validation=idx[300:400],
    # test=idx[400:500]) is the only qualifying window below. Perturb only
    # the back half of ITS OWN test period -- strictly after the point
    # whose earlier z-scores/decisions must remain untouched.
    perturb_from = idx[460]
    combined_perturbed.loc[perturb_from:, "Y"] += 500.0

    group_membership = {"Y": "GROUP_A", "X": "GROUP_A"}

    def _fake_select_pair(formation_prices, group_membership, **kwargs):
        return PairSelectionResult(
            n_candidate_tests=1,
            n_fdr_survivors=1,
            selected={"symbol_a": "Y", "symbol_b": "X"},
            all_candidates=pd.DataFrame(),
        )

    def _run_capturing_z(combined_prices: pd.DataFrame) -> list[pd.Series]:
        captured_z: list[pd.Series] = []
        real_causal_zscore = orch.causal_zscore

        def _spy_causal_zscore(spread, window, *args, **kwargs):
            z = real_causal_zscore(spread, window, *args, **kwargs)
            captured_z.append(z)
            return z

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(orch, "select_pair_within_groups", _fake_select_pair)
            mp.setattr(orch, "causal_zscore", _spy_causal_zscore)
            study = orch.run_walk_forward_study(
                combined_prices=combined_prices,
                group_membership=group_membership,
                formation_size=200,
                validation_size=100,
                test_size=100,
                step_size=100,
                grid_entry_z=(1.5,),
                grid_exit_abs_z=(0.25,),
                grid_z_window=(20,),
                grid_kalman_process_variance=(1e-4,),
                frozen_signal_params=dict(orch.FALLBACK_PARAMS),
            )
        assert any(not w.no_trade for w in study.windows)  # fixture must qualify
        return captured_z

    baseline_z_series = _run_capturing_z(combined_baseline)
    perturbed_z_series = _run_capturing_z(combined_perturbed)

    assert len(baseline_z_series) == len(perturbed_z_series) > 0
    for baseline_z, perturbed_z in zip(baseline_z_series, perturbed_z_series, strict=True):
        prefix = baseline_z.index[baseline_z.index < perturb_from]
        assert len(prefix) > 0
        pd.testing.assert_series_equal(baseline_z.loc[prefix], perturbed_z.loc[prefix])
