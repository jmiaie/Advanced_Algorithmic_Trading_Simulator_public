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
