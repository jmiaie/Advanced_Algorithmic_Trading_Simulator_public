"""D9-B v4 walk-forward orchestration -- corrects two defects an
independent review found in v3 (both verified against v3's actual code
before this fix was written, not accepted on the review's word alone):

(1) execution timing: v3 decided a position using information through bar
    t (the z-score/hedge-ratio computed FROM bar t's own close) and then
    filled that trade AT bar t's own price -- an unrealistic same-close
    assumption, most consequential for the sequential Kalman hedge ratio
    (whose bar-t filtered estimate itself incorporates bar t's own
    observation). v4 lags the executable position and the hedge ratio
    used to size it by one bar via wf_v4_backtest.simulate_pair_backtest /
    lag_for_execution: a decision made with information through t
    transacts no earlier than t+1's observed price, sized with the hedge
    ratio estimate as of t.
(2) validation tie-break: v3's tie-break for the entry/exit/z-window grid
    stopped at (Sharpe, max drawdown) and its own docstring explicitly
    said turnover was "omitted... not expected to bind" -- contradicting
    the pre-registered spec's tie_break: [lower_max_drawdown,
    lower_turnover]. v4 implements the full three-level tie-break: higher
    BASE net Sharpe, then lower max drawdown, then lower turnover, with
    ties beyond that broken by the fixed grid-iteration order (never by
    performance).

Pair selection, sizing, costs, universe, and walk-forward windowing are
unchanged from v3 and reused directly (not implicated by either defect).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .research import walk_forward_windows
from .wf_v3_backtest import (
    build_kalman_spread,
    build_static_spread,
    causal_zscore,
    generate_positions,
)
from .wf_v3_costs import BASE, GROSS, STRESS, CostScenario
from .wf_v3_pairs import PairSelectionResult, select_pair_within_groups
from .wf_v4_backtest import BacktestResult, simulate_pair_backtest

MIN_VALIDATION_TRADES = 10

FALLBACK_PARAMS: Dict[str, float] = {
    "entry_z": 2.0,
    "exit_abs_z": 0.5,
    "trailing_z_window": 40,
    "kalman_process_variance": 1e-4,
}


@dataclass
class WindowResult:
    window_index: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selection: PairSelectionResult
    no_trade: bool
    selected_pair: str | None = None
    frozen_params: Dict[str, float] | None = None
    used_fallback: bool = False
    validation_ols_sharpe: float | None = None
    validation_kalman_sharpe: float | None = None
    test_results: Dict[str, Dict[str, BacktestResult]] = field(default_factory=dict)
    # test_results[model_name][scenario_label] -> BacktestResult


@dataclass
class WalkForwardStudyResult:
    windows: List[WindowResult]
    n_windows: int
    n_no_trade_windows: int
    n_qualifying_windows: int


def available_symbols_for_window(prices: pd.DataFrame, window_index: pd.Index) -> List[str]:
    sliced = prices.loc[window_index]
    return [c for c in sliced.columns if sliced[c].notna().all()]


def _select_best_by_tiebreak(
    candidates: List[tuple[Any, BacktestResult]],
) -> tuple[Any, BacktestResult] | None:
    """Three-level tie-break, in fixed candidate (grid) order: higher
    Sharpe, then lower max drawdown, then lower turnover. A candidate only
    replaces the current best on a strict improvement, so an exact tie on
    all three keeps whichever candidate was encountered first in the
    pre-registered grid order -- never broken by any further performance
    criterion.

    max_drawdown is always <= 0 (it is a loss-from-peak fraction), so a
    SHALLOWER/better drawdown is the value CLOSER TO ZERO, i.e. the
    numerically LARGER (less negative) one -- e.g. -0.02 is a better
    drawdown than -0.10, and -0.02 > -0.10 is already true on the raw
    signed value with no negation needed. Using max_drawdown directly
    (not its negation) is what "prefer higher key" must compare against
    to prefer the shallower drawdown; negating it, as v3's original
    tie-break helper did, inverts this comparison and ends up preferring
    the DEEPER drawdown on a Sharpe tie -- verified as a third,
    independently-found defect in v3 while porting this logic to v4 (see
    the config/ledger v3-invalidation notes), not present in v3's own
    docstring claim but present in its actual comparison. Turnover is
    always >= 0, and a lower turnover is better, so it IS negated (prefer
    the candidate whose turnover is numerically smallest, i.e. whose
    negated turnover is largest)."""
    best: tuple[Any, BacktestResult] | None = None
    best_key: tuple[float, float, float] | None = None
    for params, result in candidates:
        if result.sharpe is None or result.max_drawdown is None or result.turnover is None:
            continue
        key = (result.sharpe, result.max_drawdown, -result.turnover)
        if best_key is None or key > best_key:
            best_key = key
            best = (params, result)
    return best


def _validation_grid_search_signal_params(
    formation_prices: pd.DataFrame,
    combined_for_spread: pd.DataFrame,
    validation_index: pd.Index,
    sym_a: str,
    sym_b: str,
    grid_entry_z: Sequence[float],
    grid_exit_abs_z: Sequence[float],
    grid_z_window: Sequence[int],
    allocated_nav: float,
    gross_notional_multiple: float,
    cost: CostScenario,
) -> tuple[Dict[str, float] | None, float | None]:
    """Grid-search entry_z/exit_abs_z/trailing_z_window using the Static OLS
    model's BASE-cost net Sharpe on the validation block, subject to
    min-trade-count >= 10, with the full three-level tie-break."""
    static_spread = build_static_spread(formation_prices, combined_for_spread, sym_a, sym_b)
    candidates: List[tuple[Dict[str, float], BacktestResult]] = []

    for entry_z, exit_abs_z, z_window in product(grid_entry_z, grid_exit_abs_z, grid_z_window):
        if exit_abs_z >= entry_z:
            continue  # exit band must be inside the entry band
        z = causal_zscore(static_spread.spread, z_window)
        decision_positions = generate_positions(z, entry_z, exit_abs_z)
        result = simulate_pair_backtest(
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            spread=static_spread,
            decision_positions=decision_positions,
            execution_index=validation_index,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            cost=cost,
            pair_label=f"{sym_a}/{sym_b}",
        )
        if result.n_trades < MIN_VALIDATION_TRADES or result.sharpe is None:
            continue
        candidates.append(
            ({"entry_z": entry_z, "exit_abs_z": exit_abs_z, "trailing_z_window": z_window}, result)
        )

    best = _select_best_by_tiebreak(candidates)
    if best is None:
        return None, None
    return best[0], best[1].sharpe


def _select_kalman_process_variance(
    combined_for_spread: pd.DataFrame,
    validation_index: pd.Index,
    sym_a: str,
    sym_b: str,
    signal_params: Dict[str, float],
    grid_process_variance: Sequence[float],
    observation_variance: float,
    allocated_nav: float,
    gross_notional_multiple: float,
    cost: CostScenario,
) -> tuple[float, float | None]:
    candidates: List[tuple[float, BacktestResult]] = []
    for q in grid_process_variance:
        kalman_spread = build_kalman_spread(
            combined_for_spread,
            sym_a,
            sym_b,
            process_variance=q,
            observation_variance=observation_variance,
        )
        z = causal_zscore(kalman_spread.spread, int(signal_params["trailing_z_window"]))
        decision_positions = generate_positions(
            z, signal_params["entry_z"], signal_params["exit_abs_z"]
        )
        result = simulate_pair_backtest(
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            spread=kalman_spread,
            decision_positions=decision_positions,
            execution_index=validation_index,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            cost=cost,
            pair_label=f"{sym_a}/{sym_b}",
        )
        if result.n_trades < MIN_VALIDATION_TRADES or result.sharpe is None:
            continue
        candidates.append((q, result))

    best = _select_best_by_tiebreak(candidates)
    if best is None:
        return FALLBACK_PARAMS["kalman_process_variance"], None
    return best[0], best[1].sharpe


def _run_all_cost_scenarios(
    *,
    spread: Any,
    prices_y: pd.Series,
    prices_x: pd.Series,
    decision_positions: pd.Series,
    execution_index: pd.Index,
    allocated_nav: float,
    gross_notional_multiple: float,
    pair_label: str,
) -> Dict[str, BacktestResult]:
    results: Dict[str, BacktestResult] = {}
    gross_net_return: float | None = None
    for scenario in (GROSS, BASE, STRESS):
        res = simulate_pair_backtest(
            prices_y=prices_y,
            prices_x=prices_x,
            spread=spread,
            decision_positions=decision_positions,
            execution_index=execution_index,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            cost=scenario,
            pair_label=pair_label,
        )
        if scenario.label == "analytical_zero":
            gross_net_return = res.net_return
            res.gross_return = res.net_return
        results[scenario.label] = res
    for label, res in results.items():
        if label == "analytical_zero" or res.net_return is None or gross_net_return is None:
            continue
        res.gross_return = gross_net_return
        res.cost_drag = gross_net_return - res.net_return
    return results


def run_walk_forward_study(
    combined_prices: pd.DataFrame,
    group_membership: Mapping[str, str],
    *,
    formation_size: int = 504,
    validation_size: int = 126,
    test_size: int = 63,
    step_size: int = 63,
    grid_entry_z: Sequence[float] = (1.5, 2.0, 2.5),
    grid_exit_abs_z: Sequence[float] = (0.25, 0.50, 0.75),
    grid_z_window: Sequence[int] = (20, 40, 60),
    grid_kalman_process_variance: Sequence[float] = (1e-5, 1e-4, 1e-3),
    kalman_observation_variance: float = 1e-2,
    fdr_alpha: float = 0.05,
    adf_alpha: float = 0.05,
    allocated_nav: float = 1_000_000.0,
    gross_notional_multiple: float = 1.0,
    validation_cost: CostScenario = BASE,
    frozen_signal_params: Dict[str, float] | None = None,
    fallback_params: Dict[str, float] | None = None,
) -> WalkForwardStudyResult:
    """frozen_signal_params, when provided, disables per-window hyperparameter
    re-selection entirely: every qualifying window uses these exact
    entry_z/exit_abs_z/trailing_z_window/kalman_process_variance values
    instead of calling _validation_grid_search_signal_params /
    _select_kalman_process_variance. This is required for the one-time 2025
    holdout run (config constraints no_retune_after_freeze /
    no_2025_access_before_final_configuration_frozen): the signal/Kalman
    grid must be selected once from pre-2025 DEV/2024-validation data and
    then held fixed, not reopened inside 2025 windows. Per-window PAIR
    rediscovery (select_pair_within_groups, above) and the per-window
    static-OLS/causal-Kalman spread fits below are UNCHANGED and continue
    exactly as in DEV/validation mode -- pair_rediscovery_per_window governs
    those, not the signal/Kalman hyperparameters this flag freezes. When
    None (the default, used for DEV/2024-validation), behavior is
    unchanged: each window grid-searches independently, falling back to
    fallback_params only when no candidate clears MIN_VALIDATION_TRADES.

    fallback_params supplies the same four values used both as the
    DEV-path insufficient-trades fallback (above) and, when the caller also
    passes them as frozen_signal_params, as the holdout freeze -- one
    caller-supplied source of truth (the pre-registered config's own
    selection_objective.insufficient_trades_fallback block) rather than two
    independently-maintained copies of the same numbers. Defaults to the
    module-level FALLBACK_PARAMS only when the caller passes None (e.g. a
    config that predates this field, or a test that doesn't set it)."""
    effective_fallback = dict(FALLBACK_PARAMS if fallback_params is None else fallback_params)
    windows = walk_forward_windows(
        combined_prices, formation_size, validation_size, test_size, step_size=step_size
    )
    results: List[WindowResult] = []

    for i, w in enumerate(windows):
        formation_idx, validation_idx, test_idx = w["formation"], w["validation"], w["test"]
        available = available_symbols_for_window(combined_prices, formation_idx)
        formation_prices = combined_prices.loc[formation_idx, available]

        selection = select_pair_within_groups(
            formation_prices, group_membership, fdr_alpha=fdr_alpha, adf_alpha=adf_alpha
        )
        window_result = WindowResult(
            window_index=i,
            formation_start=formation_idx.min(),
            formation_end=formation_idx.max(),
            validation_start=validation_idx.min(),
            validation_end=validation_idx.max(),
            test_start=test_idx.min(),
            test_end=test_idx.max(),
            selection=selection,
            no_trade=selection.selected is None,
        )
        if selection.selected is None:
            results.append(window_result)
            continue

        sym_a = str(selection.selected["symbol_a"])
        sym_b = str(selection.selected["symbol_b"])
        window_result.selected_pair = f"{sym_a}/{sym_b}"

        combined_for_spread = combined_prices.loc[
            formation_idx.union(validation_idx).union(test_idx), [sym_a, sym_b]
        ].dropna()

        signal_params: Dict[str, float]
        if frozen_signal_params is not None:
            # Holdout mode: hyperparameters are globally frozen, not
            # re-selected from this window's own preceding validation
            # slice. Neither selection function is called at all.
            signal_params = {
                "entry_z": frozen_signal_params["entry_z"],
                "exit_abs_z": frozen_signal_params["exit_abs_z"],
                "trailing_z_window": frozen_signal_params["trailing_z_window"],
            }
            best_q = frozen_signal_params["kalman_process_variance"]
            window_result.validation_ols_sharpe = None
            window_result.validation_kalman_sharpe = None
            window_result.used_fallback = True
        else:
            selected_signal_params, ols_val_sharpe = _validation_grid_search_signal_params(
                formation_prices=formation_prices,
                combined_for_spread=combined_for_spread,
                validation_index=validation_idx,
                sym_a=sym_a,
                sym_b=sym_b,
                grid_entry_z=grid_entry_z,
                grid_exit_abs_z=grid_exit_abs_z,
                grid_z_window=grid_z_window,
                allocated_nav=allocated_nav,
                gross_notional_multiple=gross_notional_multiple,
                cost=validation_cost,
            )
            used_fallback = selected_signal_params is None
            signal_params = (
                dict(effective_fallback)
                if selected_signal_params is None
                else selected_signal_params
            )
            window_result.validation_ols_sharpe = ols_val_sharpe
            window_result.used_fallback = used_fallback

            best_q, kalman_val_sharpe = _select_kalman_process_variance(
                combined_for_spread=combined_for_spread,
                validation_index=validation_idx,
                sym_a=sym_a,
                sym_b=sym_b,
                signal_params=signal_params,
                grid_process_variance=grid_kalman_process_variance,
                observation_variance=kalman_observation_variance,
                allocated_nav=allocated_nav,
                gross_notional_multiple=gross_notional_multiple,
                cost=validation_cost,
            )
            window_result.validation_kalman_sharpe = kalman_val_sharpe
        frozen_params = dict(signal_params)
        frozen_params["kalman_process_variance"] = best_q
        window_result.frozen_params = frozen_params

        static_spread = build_static_spread(formation_prices, combined_for_spread, sym_a, sym_b)
        kalman_spread = build_kalman_spread(
            combined_for_spread,
            sym_a,
            sym_b,
            process_variance=best_q,
            observation_variance=kalman_observation_variance,
        )
        static_z = causal_zscore(static_spread.spread, int(frozen_params["trailing_z_window"]))
        kalman_z = causal_zscore(kalman_spread.spread, int(frozen_params["trailing_z_window"]))
        static_decision_positions = generate_positions(
            static_z, frozen_params["entry_z"], frozen_params["exit_abs_z"]
        )
        kalman_decision_positions = generate_positions(
            kalman_z, frozen_params["entry_z"], frozen_params["exit_abs_z"]
        )

        window_result.test_results["static_batch_ols"] = _run_all_cost_scenarios(
            spread=static_spread,
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            decision_positions=static_decision_positions,
            execution_index=test_idx,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            pair_label=window_result.selected_pair,
        )
        window_result.test_results["sequential_filtered_kalman"] = _run_all_cost_scenarios(
            spread=kalman_spread,
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            decision_positions=kalman_decision_positions,
            execution_index=test_idx,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            pair_label=window_result.selected_pair,
        )
        results.append(window_result)

    n_no_trade = sum(1 for r in results if r.no_trade)
    return WalkForwardStudyResult(
        windows=results,
        n_windows=len(results),
        n_no_trade_windows=n_no_trade,
        n_qualifying_windows=len(results) - n_no_trade,
    )
