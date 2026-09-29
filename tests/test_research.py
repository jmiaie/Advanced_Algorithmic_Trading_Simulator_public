import numpy as np
import pandas as pd
import pytest

from stat_arb_engine.analytics import calculate_conventional_sharpe, calculate_max_drawdown
from stat_arb_engine.execution import ExecutionHandler, LimitOrderBook
from stat_arb_engine.research import (
    benjamini_hochberg,
    chronological_split,
    fit_dynamic_kalman_hedge_ratio,
    fit_static_ols_hedge_ratio,
    run_walk_forward,
    walk_forward_windows,
)
from stat_arb_engine.strategies import PositionSizer


def test_static_ols_recovers_toy_slope() -> None:
    rng = np.random.default_rng(42)
    x = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, 50)))
    y = 2.0 + 3.0 * x + pd.Series(rng.normal(0.0, 0.05, 50))
    result = fit_static_ols_hedge_ratio(y, x)
    assert result.intercept == pytest.approx(2.0, abs=0.5)
    assert result.slope == pytest.approx(3.0, abs=0.01)



def test_kalman_is_sequential_and_unaffected_by_future_row() -> None:
    x = pd.Series(np.arange(1.0, 11.0))
    y = 1.0 + 2.0 * x
    early = fit_dynamic_kalman_hedge_ratio(
        y.iloc[:5],
        x.iloc[:5],
        process_variance=1e-6,
        observation_variance=1e-4,
    )
    full = fit_dynamic_kalman_hedge_ratio(
        y,
        x,
        process_variance=1e-6,
        observation_variance=1e-4,
    )
    assert list(early.beta_path.values) == pytest.approx(
        list(full.beta_path.iloc[:5].values),
        abs=1e-8,
    )
    assert list(early.alpha_path.values) == pytest.approx(
        list(full.alpha_path.iloc[:5].values),
        abs=1e-8,
    )


def test_kalman_recovers_known_slope_without_future_smoothing() -> None:
    x = pd.Series(np.linspace(1.0, 50.0, 50))
    y = 1.0 + 2.0 * x
    result = fit_dynamic_kalman_hedge_ratio(
        y,
        x,
        process_variance=1e-6,
        observation_variance=1e-6,
    )
    assert float(result.beta_path.iloc[-1]) == pytest.approx(2.0, abs=5e-3)
    assert float(result.alpha_path.iloc[-1]) == pytest.approx(1.0, abs=5e-2)
    assert float(result.fitted_spread.abs().max()) < 1e-2


def _toy_pair(n: int = 200, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.Series(50.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    y = 1.0 + 1.5 * x + pd.Series(rng.normal(0.0, 0.5, n))
    return y, x


def test_kalman_changing_next_observation_leaves_beta_and_innovation_at_t_unchanged() -> None:
    y, x = _toy_pair()
    t = 120
    base = fit_dynamic_kalman_hedge_ratio(y, x)
    y_shocked = y.copy()
    y_shocked.iloc[t + 1] += 25.0
    shocked = fit_dynamic_kalman_hedge_ratio(y_shocked, x)
    for field_name in (
        "beta_path",
        "alpha_path",
        "predictive_innovation",
        "innovation_variance",
        "prior_beta_path",
    ):
        before = getattr(base, field_name).iloc[: t + 1].to_numpy()
        after = getattr(shocked, field_name).iloc[: t + 1].to_numpy()
        np.testing.assert_array_equal(before, after)
    # Sanity: the shock is actually visible from t+1 onward.
    assert shocked.predictive_innovation.iloc[t + 1] != base.predictive_innovation.iloc[t + 1]
    assert shocked.beta_path.iloc[t + 1] != base.beta_path.iloc[t + 1]


def test_kalman_predictive_innovation_uses_prior_state_not_y_t() -> None:
    y, x = _toy_pair()
    t = 120
    base = fit_dynamic_kalman_hedge_ratio(y, x)
    # The innovation is built from the prior (pre-update) state.
    expected = y - (base.prior_alpha_path + base.prior_beta_path * x)
    np.testing.assert_allclose(base.predictive_innovation.to_numpy(), expected.to_numpy())
    # Under the random-walk state model, the prior at t is the filtered state at t-1.
    np.testing.assert_array_equal(
        base.prior_beta_path.iloc[1:].to_numpy(), base.beta_path.iloc[:-1].to_numpy()
    )

    # Perturbing y_t moves the innovation at t one-for-one (no feedback from
    # y_t into the state estimate used to form it) and leaves the prior state
    # and predictive variance at t untouched.
    bump = 3.0
    y_bumped = y.copy()
    y_bumped.iloc[t] += bump
    bumped = fit_dynamic_kalman_hedge_ratio(y_bumped, x)
    assert bumped.prior_alpha_path.iloc[t] == base.prior_alpha_path.iloc[t]
    assert bumped.prior_beta_path.iloc[t] == base.prior_beta_path.iloc[t]
    assert bumped.innovation_variance.iloc[t] == base.innovation_variance.iloc[t]
    assert bumped.predictive_innovation.iloc[t] - base.predictive_innovation.iloc[t] == (
        pytest.approx(bump, rel=1e-12)
    )
    # The post-update residual, by contrast, absorbs only part of the bump
    # because the filtered state at t has already moved toward y_t.
    residual_move = bumped.in_sample_residual.iloc[t] - base.in_sample_residual.iloc[t]
    assert 0.0 < residual_move < bump


def test_kalman_innovation_variance_exceeds_post_update_residual_variance() -> None:
    y, x = _toy_pair()
    result = fit_dynamic_kalman_hedge_ratio(y, x)
    burn_in = 20
    innovation = result.predictive_innovation.iloc[burn_in:]
    residual = result.in_sample_residual.iloc[burn_in:]
    assert float(innovation.var(ddof=1)) > float(residual.var(ddof=1))
    # Post-update residual is the innovation scaled by R / S_t in (0, 1).
    shrink = result.observation_variance / result.innovation_variance
    np.testing.assert_allclose(
        result.in_sample_residual.to_numpy(),
        (result.predictive_innovation * shrink).to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )
    assert bool(result.innovation_variance.gt(result.observation_variance).all())
    assert result.fitted_spread is result.in_sample_residual


def test_bh_fdr_toy_example() -> None:
    result = benjamini_hochberg([0.001, 0.01, 0.04, 0.20], alpha=0.05)
    assert result.loc[0, "rejected"]
    assert result.loc[1, "rejected"]
    assert not result.loc[3, "rejected"]
    assert result.loc[2, "adjusted_pvalue"] == pytest.approx(0.0533333333)
    assert all(result["test_count"] == 4)
    assert all(result["qvalue"].sort_values().diff().fillna(0.0) >= 0.0)



def test_formation_validation_test_non_overlap() -> None:
    frame = pd.DataFrame(
        {"x": range(12)},
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    split = chronological_split(frame, formation_size=4, validation_size=4, test_size=4)
    assert split.formation.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_chronological_split_rejects_non_monotonic_index() -> None:
    frame = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"]),
    )
    with pytest.raises(ValueError, match="Non-monotonic timestamps"):
        chronological_split(frame, formation_size=1, validation_size=1, test_size=1)


def test_walk_forward_training_excludes_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 15) + np.sin(np.arange(15)) * 0.01,
            "B": np.linspace(50, 55, 15) + np.cos(np.arange(15)) * 0.01,
        },
        index=pd.date_range("2024-01-01", periods=15, freq="D"),
    )
    result = run_walk_forward(frame, formation_size=5, validation_size=3, test_size=2)
    assert not result.empty
    assert all(result["trained_through"] < result["test_start"])
    assert all(result["trained_through"] == result["formation_end"])
    assert all(result["pair_selection_end"] == result["formation_end"])
    assert all(result["hedge_fit_end"] == result["formation_end"])
    assert all(result["normalization_end"] == result["formation_end"])
    assert all(result["oos_start"] == result["test_start"])
    windows = walk_forward_windows(frame, formation_size=5, validation_size=3, test_size=2)
    assert all(window["formation"].max() < window["validation"].min() for window in windows)


def test_walk_forward_pair_selection_ignores_future_mutation() -> None:
    rng = np.random.default_rng(123)
    index = pd.date_range("2024-01-01", periods=20, freq="D")
    a = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, len(index))), index=index)
    b = 1.5 * a + 2.0 + pd.Series(rng.normal(0.0, 0.05, len(index)), index=index)
    c = pd.Series(50.0 + np.cumsum(rng.normal(0.0, 2.0, len(index))), index=index)
    d = pd.Series(25.0 + np.cumsum(rng.normal(0.0, 2.0, len(index))), index=index)
    base = pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=index)
    mutated = base.copy()
    mutated.loc[index[10]:, "C"] = mutated.loc[index[10]:, "A"] * 0.5 + 5.0
    mutated.loc[index[10]:, "D"] = mutated.loc[index[10]:, "A"] * 0.5 + 5.1

    baseline = run_walk_forward(base, formation_size=10, validation_size=4, test_size=2)
    future_mutated = run_walk_forward(mutated, formation_size=10, validation_size=4, test_size=2)

    assert not baseline.empty
    assert baseline.loc[0, "selected_pair"] == future_mutated.loc[0, "selected_pair"]


def test_trailing_max_drawdown_includes_starting_nav() -> None:
    assert calculate_max_drawdown([-0.10, 0.0]) == pytest.approx(-0.10)



def test_conventional_sharpe_matches_explicit_formula() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    expected = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert calculate_conventional_sharpe(returns, periods_per_year=252) == pytest.approx(
        expected
    )



def test_position_sizer_fixed_gross_notional_bounds_gross_exposure() -> None:
    sizer = PositionSizer(mode="fixed_gross_notional", gross_notional=10_000.0)
    sizes = sizer.size({"A": 100.0, "B": 50.0}, "A", "B", hedge_ratio=1.0)
    gross = sizes["A"] * 100.0 + sizes["B"] * 50.0
    assert gross <= 10_000.0



def test_execution_vwap_partial_fill_and_zero_liquidity() -> None:
    lob = LimitOrderBook()
    lob.update(99.99, 100.01, depth_qty=50.0, levels=2)
    execution = ExecutionHandler(lob)
    fill = execution.submit_order(
        {
            "type": "MARKET",
            "symbol": "A",
            "side": "buy",
            "qty": 120,
            "timestamp": "2024-01-01",
        }
    )
    assert fill is not None
    assert fill["qty"] == 100.0
    assert fill["remaining_qty"] == 20.0

    empty = LimitOrderBook()
    execution_empty = ExecutionHandler(empty)
    assert (
        execution_empty.submit_order(
            {
                "type": "MARKET",
                "symbol": "A",
                "side": "buy",
                "qty": 10,
                "timestamp": "2024-01-01",
            }
        )
        is None
    )
