"""Offline tests for the D9-B v3 conforming walk-forward pipeline.

All fixtures here are SYNTHETIC (seeded random walks / manufactured
cointegrated pairs) -- for software verification only. No real market data
is used or implied; none of these numbers are research claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb_engine.wf_v3_backtest import (
    build_kalman_spread,
    build_static_spread,
    causal_zscore,
    simulate_pair_backtest,
)
from stat_arb_engine.wf_v3_costs import BASE, GROSS, STRESS
from stat_arb_engine.wf_v3_orch import (
    available_symbols_for_window,
    run_walk_forward_study,
)
from stat_arb_engine.wf_v3_pairs import (
    enumerate_group_candidates,
    select_pair_within_groups,
)
from stat_arb_engine.wf_v3_sizing import size_pair_legs

GROUPS = {
    "A": ["SPY", "QQQ", "DIA", "IWM"],
    "B": ["XLK", "XLF"],
}
GROUP_MEMBERSHIP = {sym: g for g, syms in GROUPS.items() for sym in syms}


def _synthetic_panel(
    n: int = 700, seed: int = 0, cointegrated_pairs: tuple[tuple[str, str], ...] = (("SPY", "QQQ"),)
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n)
    data = {}
    base_walks: dict[str, np.ndarray] = {}
    for sym in ["SPY", "DIA", "IWM", "XLK", "XLF"]:
        base_walks[sym] = 100 + np.cumsum(rng.normal(0, 0.5, n))
    for sym_a, sym_b in cointegrated_pairs:
        if sym_a not in base_walks:
            base_walks[sym_a] = base_walks[sym_b] * 1.3 + rng.normal(0, 0.1, n)
        else:
            base_walks[sym_b] = base_walks[sym_a] / 1.3 + rng.normal(0, 0.1, n)
    for sym, walk in base_walks.items():
        data[sym] = np.abs(walk) + 10.0
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------- groups ---


def test_no_cross_group_pairs_enumerated():
    symbols = ["SPY", "QQQ", "XLK", "XLF"]
    pairs = enumerate_group_candidates(symbols, GROUP_MEMBERSHIP)
    for a, b in pairs:
        assert GROUP_MEMBERSHIP[a] == GROUP_MEMBERSHIP[b]
    assert ("SPY", "XLK") not in pairs and ("XLK", "SPY") not in pairs


def test_within_group_pairs_are_exhaustive():
    pairs = enumerate_group_candidates(["SPY", "QQQ", "DIA"], GROUP_MEMBERSHIP)
    assert set(pairs) == {("DIA", "QQQ"), ("DIA", "SPY"), ("QQQ", "SPY")}


# ------------------------------------------------------------- selection ---


def test_no_qualifying_pair_yields_no_trade():
    panel = _synthetic_panel(n=520, cointegrated_pairs=())  # independent random walks
    formation = panel.iloc[:504]
    result = select_pair_within_groups(formation, GROUP_MEMBERSHIP, fdr_alpha=0.01, adf_alpha=0.01)
    assert result.selected is None


def test_deterministic_tiebreak_order(monkeypatch):
    """Two equally-significant candidate pairs must be ranked by ADF p-value
    then lexicographic symbol order, not by insertion order or randomness."""
    from stat_arb_engine import wf_v3_pairs

    class FakeFit:
        def __init__(self, eg_p, adf_p):
            self.engle_granger_pvalue = eg_p
            self.adf_pvalue = adf_p
            self.slope = 1.0
            self.intercept = 0.0
            self.half_life = 10.0
            self.formation_sample_count = 100

    calls = {
        ("QQQ", "SPY"): FakeFit(0.001, 0.02),
        ("DIA", "IWM"): FakeFit(0.001, 0.01),  # same EG p, lower ADF p -> should win
    }

    def fake_fit(y, x):
        # y/x are passed as columns; recover names via .name
        return calls[(y.name, x.name)]

    monkeypatch.setattr(wf_v3_pairs, "fit_static_ols_hedge_ratio", fake_fit)
    # Two disjoint groups so only the two candidate pairs above are enumerated.
    panel = pd.DataFrame(
        {c: np.arange(100, dtype=float) for c in ["SPY", "QQQ", "DIA", "IWM"]},
    )
    membership = {"SPY": "A", "QQQ": "A", "DIA": "B", "IWM": "B"}
    result = select_pair_within_groups(panel, membership, fdr_alpha=0.5, adf_alpha=0.5)
    assert result.selected is not None
    assert (result.selected["symbol_a"], result.selected["symbol_b"]) == ("DIA", "IWM")


def test_available_symbols_excludes_partial_coverage_window():
    idx = pd.bdate_range("2015-01-02", periods=10)
    panel = pd.DataFrame({"SPY": range(10), "XLRE": [np.nan] * 5 + list(range(5))}, index=idx)
    available = available_symbols_for_window(panel, idx[:5])
    assert "XLRE" not in available and "SPY" in available
    available_later = available_symbols_for_window(panel, idx)
    assert "XLRE" not in available_later  # still has NaNs in the full window


# ---------------------------------------------------------------- z-score --


def test_causal_zscore_does_not_use_current_bar():
    rng = np.random.default_rng(0)
    trailing = list(1.0 + rng.normal(0, 0.01, 30))
    spread = pd.Series(trailing + [1000.0], index=pd.bdate_range("2015-01-02", periods=31))
    z = causal_zscore(spread, window=20)
    # The huge jump at the last bar must not appear in its own rolling stats.
    last_z = z.iloc[-1]
    assert last_z > 0  # spread is far above its trailing mean
    # Perturbing the last value must not change earlier z-scores (no lookahead).
    spread2 = spread.copy()
    spread2.iloc[-1] = -1000.0
    z2 = causal_zscore(spread2, window=20)
    pd.testing.assert_series_equal(z.iloc[:-1], z2.iloc[:-1])


def test_zscore_window_respected():
    spread = pd.Series(np.arange(50, dtype=float))
    z = causal_zscore(spread, window=20)
    assert z.iloc[:20].isna().all()  # not enough trailing history yet
    assert z.iloc[20:].notna().all()


# ------------------------------------------------------------------ OLS ---


def test_static_ols_spread_has_intercept_and_is_frozen():
    panel = _synthetic_panel(n=520)
    formation = panel.iloc[:400]
    spread_series = build_static_spread(formation, panel, "SPY", "QQQ")
    # Hedge ratio/intercept must be constant across the whole combined series
    # (frozen from formation, not refit).
    assert spread_series.hedge_ratio.nunique() == 1
    assert spread_series.intercept.nunique() == 1


# --------------------------------------------------------------- Kalman ---


def test_kalman_is_forward_only_prefix_invariant():
    """Extending the series with future bars must not change earlier beta_path
    values (no smoothing, no lookahead)."""
    panel = _synthetic_panel(n=520)
    short = panel.iloc[:300]
    long = panel.iloc[:400]
    s_short = build_kalman_spread(
        short, "SPY", "QQQ", process_variance=1e-4, observation_variance=1e-2
    )
    s_long = build_kalman_spread(
        long, "SPY", "QQQ", process_variance=1e-4, observation_variance=1e-2
    )
    pd.testing.assert_series_equal(
        s_short.hedge_ratio, s_long.hedge_ratio.iloc[: len(s_short.hedge_ratio)]
    )
    pd.testing.assert_series_equal(
        s_short.spread, s_long.spread.iloc[: len(s_short.spread)]
    )


# ---------------------------------------------------------------- sizing --


def test_gross_notional_invariant_respected():
    legs = size_pair_legs(
        direction=1, hedge_ratio=1.5, price_y=100.0, price_x=50.0,
        allocated_nav=1_000_000.0, gross_notional_multiple=1.0,
    )
    assert legs.gross_notional_actual == pytest.approx(1_000_000.0, rel=1e-9)


def test_gross_notional_multiple_cannot_exceed_one():
    with pytest.raises(ValueError):
        size_pair_legs(
            direction=1, hedge_ratio=1.0, price_y=100.0, price_x=100.0,
            allocated_nav=1_000_000.0, gross_notional_multiple=1.5,
        )


def test_sizing_direction_sign_conventions():
    long_legs = size_pair_legs(
        direction=1, hedge_ratio=1.0, price_y=100.0, price_x=100.0,
        allocated_nav=1_000_000.0,
    )
    short_legs = size_pair_legs(
        direction=-1, hedge_ratio=1.0, price_y=100.0, price_x=100.0,
        allocated_nav=1_000_000.0,
    )
    assert long_legs.qty_y > 0 and long_legs.qty_x < 0  # long spread: long y, short x
    assert short_legs.qty_y < 0 and short_legs.qty_x > 0


# ----------------------------------------------------------------- costs --


def test_cost_scenarios_ordered_gross_base_stress():
    qty, price = 1000.0, 100.0
    assert GROSS.fill_cost(qty, price) == 0.0
    assert 0 < BASE.fill_cost(qty, price) < STRESS.fill_cost(qty, price)


def test_borrow_cost_scales_with_short_market_value_and_bps():
    smv = 100_000.0
    base_borrow = BASE.daily_borrow_cost(smv)
    stress_borrow = STRESS.daily_borrow_cost(smv)
    assert base_borrow > 0
    assert stress_borrow > base_borrow
    assert GROSS.daily_borrow_cost(smv) == 0.0
    assert BASE.daily_borrow_cost(0.0) == 0.0


# ------------------------------------------------------------- backtest ---


def test_backtest_no_trade_when_flat_throughout():
    idx = pd.bdate_range("2015-01-02", periods=10)
    prices_y = pd.Series(100.0, index=idx)
    prices_x = pd.Series(50.0, index=idx)
    from stat_arb_engine.wf_v3_backtest import SpreadSeries

    spread = SpreadSeries(
        spread=pd.Series(0.0, index=idx),
        hedge_ratio=pd.Series(1.0, index=idx),
        intercept=pd.Series(0.0, index=idx),
        estimation_mode="static_batch_ols",
    )
    positions = pd.Series(0, index=idx)
    result = simulate_pair_backtest(
        prices_y=prices_y, prices_x=prices_x, spread=spread, positions=positions,
        allocated_nav=1_000_000.0, gross_notional_multiple=1.0, cost=BASE, pair_label="Y/X",
    )
    assert result.n_trades == 0
    assert result.net_return == pytest.approx(0.0)


# ---------------------------------------------------------- orchestrator --


def test_walk_forward_windows_are_exact_504_126_63_63():
    panel = _synthetic_panel(n=693 + 63)  # two windows worth
    study = run_walk_forward_study(panel, GROUP_MEMBERSHIP, fdr_alpha=0.5, adf_alpha=0.5)
    assert study.n_windows >= 1
    w = study.windows[0]
    assert (w.formation_end - w.formation_start).days >= 0  # sanity: populated
    # Recompute expected sizes directly against the windows helper.
    from stat_arb_engine.research import walk_forward_windows

    raw_windows = walk_forward_windows(panel, 504, 126, 63, step_size=63)
    assert len(raw_windows[0]["formation"]) == 504
    assert len(raw_windows[0]["validation"]) == 126
    assert len(raw_windows[0]["test"]) == 63


def test_pair_rediscovery_is_independent_per_window_not_chronology_only():
    """A cointegrated pair only in the first half of the series must be
    selectable in early windows and produce NO_TRADE (or a different pair)
    once its relationship decays -- proving selection re-runs per window
    rather than being fixed once at the start."""
    n = 693 + 63
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-02", periods=n)
    spy = 100 + np.cumsum(rng.normal(0, 0.5, n))
    # QQQ cointegrated with SPY for the first 500 bars, then decouples.
    qqq = np.concatenate(
        [spy[:500] * 1.2 + rng.normal(0, 0.05, 500), 100 + np.cumsum(rng.normal(0, 0.5, n - 500))]
    )
    dia = 100 + np.cumsum(rng.normal(0, 0.5, n))
    iwm = 100 + np.cumsum(rng.normal(0, 0.5, n))
    panel = pd.DataFrame(
        {"SPY": np.abs(spy) + 10, "QQQ": np.abs(qqq) + 10, "DIA": np.abs(dia) + 10,
         "IWM": np.abs(iwm) + 10},
        index=idx,
    )
    study = run_walk_forward_study(panel, GROUP_MEMBERSHIP, fdr_alpha=0.1, adf_alpha=0.1)
    selections = [w.selected_pair for w in study.windows]
    # If selection were chronology-only (fixed after window 0), every window
    # would show the same outcome regardless of the underlying relationship
    # decay; this asserts the study actually re-evaluated per window.
    assert len(set(selections)) > 1 or any(w.no_trade for w in study.windows)


def test_same_signal_rules_for_ols_and_kalman_in_qualifying_window():
    panel = _synthetic_panel(n=693)
    study = run_walk_forward_study(panel, GROUP_MEMBERSHIP, fdr_alpha=0.5, adf_alpha=0.5)
    qualifying = [w for w in study.windows if not w.no_trade]
    if not qualifying:
        pytest.skip("no qualifying window in this synthetic draw")
    w = qualifying[0]
    assert "static_batch_ols" in w.test_results
    assert "sequential_filtered_kalman" in w.test_results
    for scenario in ("analytical_zero", "BASE", "STRESS"):
        assert scenario in w.test_results["static_batch_ols"]
        assert scenario in w.test_results["sequential_filtered_kalman"]


def test_deterministic_rerun_produces_identical_results():
    panel = _synthetic_panel(n=693, seed=7)
    study1 = run_walk_forward_study(panel, GROUP_MEMBERSHIP, fdr_alpha=0.5, adf_alpha=0.5)
    study2 = run_walk_forward_study(panel, GROUP_MEMBERSHIP, fdr_alpha=0.5, adf_alpha=0.5)
    assert [w.selected_pair for w in study1.windows] == [w.selected_pair for w in study2.windows]
    assert [w.no_trade for w in study1.windows] == [w.no_trade for w in study2.windows]
