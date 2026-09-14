from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class StaticOLSHedgeResult:
    slope: float
    intercept: float
    residual_spread: pd.Series
    adf_stat: float
    adf_pvalue: float
    engle_granger_pvalue: float
    half_life: float
    formation_sample_count: int


@dataclass(frozen=True)
class DynamicKalmanResult:
    alpha_path: pd.Series
    beta_path: pd.Series
    fitted_spread: pd.Series
    observation_variance: float
    process_variance: float


@dataclass(frozen=True)
class ChronologicalSplit:
    formation: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _validate_chronological_index(index: pd.Index) -> None:
    if index.has_duplicates:
        raise ValueError("Duplicate timestamps detected for chronological research input")
    if not index.is_monotonic_increasing:
        raise ValueError("Non-monotonic timestamps detected for chronological research input")



def _coerce_pair_inputs(
    y: Sequence[float] | pd.Series,
    x: Sequence[float] | pd.Series,
) -> tuple[pd.Series, pd.Series]:
    y_series = pd.Series(y, dtype=float)
    x_series = pd.Series(x, dtype=float)
    aligned = pd.concat([y_series, x_series], axis=1).dropna()
    aligned.columns = ["y", "x"]
    if len(aligned) < 5:
        raise ValueError("Need at least 5 aligned observations")
    return aligned["y"], aligned["x"]



def _estimate_half_life(spread: pd.Series) -> float:
    lagged = spread.shift(1)
    delta = spread - lagged
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    aligned.columns = ["lagged", "delta"]
    if aligned.empty:
        return float("inf")
    design = sm.add_constant(aligned["lagged"])
    result = sm.OLS(aligned["delta"], design).fit()
    beta = float(result.params["lagged"])
    return float(-np.log(2.0) / beta) if beta < 0 else float("inf")



def fit_static_ols_hedge_ratio(
    y: Sequence[float] | pd.Series,
    x: Sequence[float] | pd.Series,
) -> StaticOLSHedgeResult:
    y_series, x_series = _coerce_pair_inputs(y, x)
    design = sm.add_constant(x_series)
    result = sm.OLS(y_series, design).fit()
    intercept = float(result.params["const"])
    slope = float(result.params["x"])
    residual_spread = y_series - (intercept + slope * x_series)
    adf_stat, adf_pvalue, *_ = adfuller(residual_spread)
    _, eg_pvalue, _ = coint(y_series, x_series)
    return StaticOLSHedgeResult(
        slope=slope,
        intercept=intercept,
        residual_spread=residual_spread,
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_pvalue),
        engle_granger_pvalue=float(eg_pvalue),
        half_life=_estimate_half_life(residual_spread),
        formation_sample_count=int(len(y_series)),
    )



def fit_dynamic_kalman_hedge_ratio(
    y: Sequence[float] | pd.Series,
    x: Sequence[float] | pd.Series,
    process_variance: float = 1e-4,
    observation_variance: float = 1e-2,
    alpha_random_walk: bool = True,
) -> DynamicKalmanResult:
    y_series, x_series = _coerce_pair_inputs(y, x)
    index = y_series.index

    state = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float)
    if alpha_random_walk:
        transition_cov = np.diag([process_variance, process_variance])
    else:
        transition_cov = np.diag([0.0, process_variance])

    alpha_path: List[float] = []
    beta_path: List[float] = []
    residuals: List[float] = []

    for t in range(len(y_series)):
        predicted_state = state.copy()
        predicted_covariance = covariance + transition_cov
        observation = np.array([1.0, float(x_series.iloc[t])], dtype=float)
        innovation = float(y_series.iloc[t]) - float(observation @ predicted_state)
        innovation_variance = float(
            observation @ predicted_covariance @ observation.T + observation_variance
        )
        kalman_gain = predicted_covariance @ observation.T / innovation_variance
        state = predicted_state + kalman_gain * innovation
        covariance = (np.eye(2) - np.outer(kalman_gain, observation)) @ predicted_covariance
        alpha_path.append(float(state[0]))
        beta_path.append(float(state[1]))
        residuals.append(float(y_series.iloc[t] - (state[0] + state[1] * x_series.iloc[t])))

    return DynamicKalmanResult(
        alpha_path=pd.Series(alpha_path, index=index, name="alpha_t"),
        beta_path=pd.Series(beta_path, index=index, name="beta_t"),
        fitted_spread=pd.Series(residuals, index=index, name="spread_t"),
        observation_variance=observation_variance,
        process_variance=process_variance,
    )



def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    series = pd.Series(list(p_values), dtype=float)
    if series.empty:
        return pd.DataFrame(
            columns=[
                "raw_pvalue",
                "adjusted_pvalue",
                "qvalue",
                "rejected",
                "test_count",
                "hypotheses_count",
            ]
        )
    ordered = series.sort_values()
    m = len(ordered)
    adjusted = ordered * m / (np.arange(1, m + 1))
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    rejected = ordered <= (alpha * np.arange(1, m + 1) / m)
    result = pd.DataFrame(
        {
            "raw_pvalue": ordered,
            "adjusted_pvalue": adjusted,
            "qvalue": adjusted,
            "rejected": rejected,
            "test_count": m,
            "hypotheses_count": m,
        }
    )
    return result.reindex(series.index)



def chronological_split(
    data: pd.DataFrame,
    formation_size: int,
    validation_size: int,
    test_size: int,
) -> ChronologicalSplit:
    _validate_chronological_index(data.index)
    if formation_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("All split sizes must be positive")
    total = formation_size + validation_size + test_size
    if len(data) < total:
        raise ValueError("Not enough rows for requested split sizes")
    formation = data.iloc[:formation_size].copy()
    validation = data.iloc[formation_size : formation_size + validation_size].copy()
    test = data.iloc[formation_size + validation_size : total].copy()
    return ChronologicalSplit(formation=formation, validation=validation, test=test)



def walk_forward_windows(
    data: pd.DataFrame,
    formation_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
) -> List[Dict[str, pd.Index]]:
    _validate_chronological_index(data.index)
    if step_size is None:
        step_size = test_size
    windows: List[Dict[str, pd.Index]] = []
    total_window = formation_size + validation_size + test_size
    for start in range(0, len(data) - total_window + 1, step_size):
        formation = data.index[start : start + formation_size]
        validation = data.index[
            start + formation_size : start + formation_size + validation_size
        ]
        test = data.index[start + formation_size + validation_size : start + total_window]
        windows.append({"formation": formation, "validation": validation, "test": test})
    return windows



def select_pairs(
    price_frame: pd.DataFrame,
    universe: Sequence[str] | None = None,
    sector_grouping: Dict[str, str] | None = None,
    fdr_alpha: float = 0.05,
    adf_alpha: float = 0.05,
) -> pd.DataFrame:
    frame = price_frame.copy()
    _validate_chronological_index(frame.index)
    if universe is not None:
        frame = frame.loc[:, list(universe)]
    frame = frame.dropna(axis=0, how="any")
    results: List[Dict[str, Any]] = []
    for sym_a, sym_b in combinations(frame.columns, 2):
        try:
            pair_result = fit_static_ols_hedge_ratio(frame[sym_a], frame[sym_b])
        except ValueError:
            continue
        results.append(
            {
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "engle_granger_pvalue": pair_result.engle_granger_pvalue,
                "adf_pvalue": pair_result.adf_pvalue,
                "hedge_ratio": pair_result.slope,
                "intercept": pair_result.intercept,
                "half_life": pair_result.half_life,
                "formation_sample_count": pair_result.formation_sample_count,
                "sector_a": None if sector_grouping is None else sector_grouping.get(sym_a),
                "sector_b": None if sector_grouping is None else sector_grouping.get(sym_b),
                "coverage_start": frame.index.min(),
                "coverage_end": frame.index.max(),
            }
        )
    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        return result_frame
    bh = benjamini_hochberg(result_frame["engle_granger_pvalue"], alpha=fdr_alpha)
    result_frame["raw_pvalue"] = bh["raw_pvalue"].values
    result_frame["adjusted_pvalue"] = bh["adjusted_pvalue"].values
    result_frame["qvalue"] = bh["qvalue"].values
    result_frame["rejected"] = bh["rejected"].values & (
        result_frame["adf_pvalue"] < adf_alpha
    )
    result_frame["test_count"] = bh["test_count"].values
    result_frame["hypotheses_count"] = bh["hypotheses_count"].values
    result_frame["selection_logic"] = (
        "BH on Engle-Granger p-values; residual ADF as secondary diagnostic filter"
    )
    return result_frame.sort_values(
        ["rejected", "qvalue", "engle_granger_pvalue"],
        ascending=[False, True, True],
    ).reset_index(drop=True)



def run_walk_forward(
    price_frame: pd.DataFrame,
    formation_size: int,
    validation_size: int,
    test_size: int,
    model: str = "static_ols",
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    windows = walk_forward_windows(price_frame, formation_size, validation_size, test_size)
    rows: List[Dict[str, Any]] = []
    for idx, window in enumerate(windows):
        formation = price_frame.loc[window["formation"]]
        validation = price_frame.loc[window["validation"]]
        test = price_frame.loc[window["test"]]
        selected = select_pairs(formation, fdr_alpha=fdr_alpha)
        top_pair = selected.iloc[0] if not selected.empty else None
        rows.append(
            {
                "window": idx,
                "formation_start": formation.index.min(),
                "formation_end": formation.index.max(),
                "validation_start": validation.index.min(),
                "validation_end": validation.index.max(),
                "test_start": test.index.min(),
                "test_end": test.index.max(),
                "trained_through": formation.index.max(),
                "pair_selection_end": formation.index.max(),
                "hedge_fit_end": formation.index.max(),
                "normalization_end": formation.index.max(),
                "validation_reserved_for_tuning": True,
                "oos_start": test.index.min(),
                "model": model,
                "selected_pair": (
                    None
                    if top_pair is None
                    else f"{top_pair['symbol_a']}/{top_pair['symbol_b']}"
                ),
                "selection_from_formation_only": True,
                "test_rows": len(test),
            }
        )
    return pd.DataFrame(rows)



def compare_hedge_models(
    y: Sequence[float] | pd.Series,
    x: Sequence[float] | pd.Series,
    test_index: Sequence[Any] | None = None,
    process_variance: float = 1e-4,
    observation_variance: float = 1e-2,
) -> pd.DataFrame:
    static_result = fit_static_ols_hedge_ratio(y, x)
    dynamic_result = fit_dynamic_kalman_hedge_ratio(
        y,
        x,
        process_variance=process_variance,
        observation_variance=observation_variance,
    )
    index = pd.Series(y).index if test_index is None else pd.Index(test_index)
    return pd.DataFrame(
        {
            "static_spread": static_result.residual_spread.values,
            "dynamic_spread": dynamic_result.fitted_spread.values,
            "dynamic_beta": dynamic_result.beta_path.values,
            "dynamic_alpha": dynamic_result.alpha_path.values,
        },
        index=index,
    )
