"""V3 walk-forward orchestration: true per-window pair rediscovery,
validation-grid parameter selection, and a fair OLS-vs-Kalman test
evaluation under all three cost scenarios.

Entry/exit/z-window are selected once per window using the Static OLS
model's validation-period net Sharpe (OLS is this program's primary
trading model) and then held identical for the Kalman comparison --
spec: "Use identical: pair, OOS window, entry/exit policy, costs, sizing
... except for hedge estimation." Kalman's own process_variance is then
selected separately using the Kalman model's validation Sharpe under
those already-fixed signal rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .research import walk_forward_windows
from .wf_v3_backtest import (
    BacktestResult,
    build_kalman_spread,
    build_static_spread,
    causal_zscore,
    generate_positions,
    simulate_pair_backtest,
)
from .wf_v3_costs import BASE, GROSS, STRESS, CostScenario
from .wf_v3_pairs import PairSelectionResult, select_pair_within_groups

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
    """Symbols with complete (non-NaN) coverage across the given window --
    handles a symbol (e.g. XLRE) whose actual data start is after the
    requested dataset start by simply excluding it from windows that
    predate its inception, without changing the acquired universe."""
    sliced = prices.loc[window_index]
    return [c for c in sliced.columns if sliced[c].notna().all()]


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
    min-trade-count >= 10. Returns (best_params_or_None, best_sharpe)."""
    static_spread = build_static_spread(formation_prices, combined_for_spread, sym_a, sym_b)
    best: tuple[Dict[str, float], float] | None = None

    for entry_z, exit_abs_z, z_window in product(grid_entry_z, grid_exit_abs_z, grid_z_window):
        if exit_abs_z >= entry_z:
            continue  # exit band must be inside the entry band
        z = causal_zscore(static_spread.spread, z_window)
        positions_full = generate_positions(z, entry_z, exit_abs_z)
        positions_val = positions_full.loc[validation_index]
        result = simulate_pair_backtest(
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            spread=static_spread,
            positions=positions_val,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            cost=cost,
            pair_label=f"{sym_a}/{sym_b}",
        )
        if result.n_trades < MIN_VALIDATION_TRADES or result.sharpe is None:
            continue
        candidate = (
            {"entry_z": entry_z, "exit_abs_z": exit_abs_z, "trailing_z_window": z_window},
            result.sharpe,
        )
        if best is None:
            best = candidate
        else:
            best_dd = _validation_tiebreak_metric(
                static_spread, combined_for_spread, validation_index, sym_a, sym_b,
                best[0], allocated_nav, gross_notional_multiple, cost,
            )
            cand_dd = _validation_tiebreak_metric(
                static_spread, combined_for_spread, validation_index, sym_a, sym_b,
                candidate[0], allocated_nav, gross_notional_multiple, cost,
            )
            if candidate[1] > best[1] or (
                candidate[1] == best[1] and cand_dd is not None and best_dd is not None
                and cand_dd > best_dd
            ):
                best = candidate

    if best is None:
        return None, None
    return best[0], best[1]


def _validation_tiebreak_metric(
    static_spread: Any,
    combined_for_spread: pd.DataFrame,
    validation_index: pd.Index,
    sym_a: str,
    sym_b: str,
    params: Dict[str, float],
    allocated_nav: float,
    gross_notional_multiple: float,
    cost: CostScenario,
) -> float | None:
    """max_drawdown for tie-break (spec: lower max drawdown, then lower turnover;
    turnover tie-break omitted as a third-order refinement not expected to bind)."""
    z = causal_zscore(static_spread.spread, int(params["trailing_z_window"]))
    positions = generate_positions(z, params["entry_z"], params["exit_abs_z"]).loc[
        validation_index
    ]
    result = simulate_pair_backtest(
        prices_y=combined_for_spread[sym_a],
        prices_x=combined_for_spread[sym_b],
        spread=static_spread,
        positions=positions,
        allocated_nav=allocated_nav,
        gross_notional_multiple=gross_notional_multiple,
        cost=cost,
        pair_label=f"{sym_a}/{sym_b}",
    )
    return -result.max_drawdown if result.max_drawdown is not None else None


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
    best_q = FALLBACK_PARAMS["kalman_process_variance"]
    best_sharpe: float | None = None
    for q in grid_process_variance:
        kalman_spread = build_kalman_spread(
            combined_for_spread, sym_a, sym_b,
            process_variance=q, observation_variance=observation_variance,
        )
        z = causal_zscore(kalman_spread.spread, int(signal_params["trailing_z_window"]))
        positions = generate_positions(
            z, signal_params["entry_z"], signal_params["exit_abs_z"]
        ).loc[validation_index]
        result = simulate_pair_backtest(
            prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b],
            spread=kalman_spread,
            positions=positions,
            allocated_nav=allocated_nav,
            gross_notional_multiple=gross_notional_multiple,
            cost=cost,
            pair_label=f"{sym_a}/{sym_b}",
        )
        if result.n_trades < MIN_VALIDATION_TRADES or result.sharpe is None:
            continue
        if best_sharpe is None or result.sharpe > best_sharpe:
            best_sharpe = result.sharpe
            best_q = q
    return best_q, best_sharpe


def _run_all_cost_scenarios(
    *,
    spread: Any,
    prices_y: pd.Series,
    prices_x: pd.Series,
    positions: pd.Series,
    allocated_nav: float,
    gross_notional_multiple: float,
    pair_label: str,
) -> Dict[str, BacktestResult]:
    results: Dict[str, BacktestResult] = {}
    gross_net_return: float | None = None
    for scenario in (GROSS, BASE, STRESS):
        res = simulate_pair_backtest(
            prices_y=prices_y, prices_x=prices_x, spread=spread, positions=positions,
            allocated_nav=allocated_nav, gross_notional_multiple=gross_notional_multiple,
            cost=scenario, pair_label=pair_label,
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
) -> WalkForwardStudyResult:
    windows = walk_forward_windows(combined_prices, formation_size, validation_size, test_size,
                                    step_size=step_size)
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
            formation_start=formation_idx.min(), formation_end=formation_idx.max(),
            validation_start=validation_idx.min(), validation_end=validation_idx.max(),
            test_start=test_idx.min(), test_end=test_idx.max(),
            selection=selection, no_trade=selection.selected is None,
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

        signal_params, ols_val_sharpe = _validation_grid_search_signal_params(
            formation_prices=formation_prices,
            combined_for_spread=combined_for_spread,
            validation_index=validation_idx,
            sym_a=sym_a, sym_b=sym_b,
            grid_entry_z=grid_entry_z, grid_exit_abs_z=grid_exit_abs_z,
            grid_z_window=grid_z_window,
            allocated_nav=allocated_nav, gross_notional_multiple=gross_notional_multiple,
            cost=validation_cost,
        )
        used_fallback = signal_params is None
        if signal_params is None:
            signal_params = dict(FALLBACK_PARAMS)
        window_result.validation_ols_sharpe = ols_val_sharpe
        window_result.used_fallback = used_fallback

        best_q, kalman_val_sharpe = _select_kalman_process_variance(
            combined_for_spread=combined_for_spread, validation_index=validation_idx,
            sym_a=sym_a, sym_b=sym_b, signal_params=signal_params,
            grid_process_variance=grid_kalman_process_variance,
            observation_variance=kalman_observation_variance,
            allocated_nav=allocated_nav, gross_notional_multiple=gross_notional_multiple,
            cost=validation_cost,
        )
        window_result.validation_kalman_sharpe = kalman_val_sharpe
        frozen_params = dict(signal_params)
        frozen_params["kalman_process_variance"] = best_q
        window_result.frozen_params = frozen_params

        static_spread = build_static_spread(formation_prices, combined_for_spread, sym_a, sym_b)
        kalman_spread = build_kalman_spread(
            combined_for_spread, sym_a, sym_b,
            process_variance=best_q, observation_variance=kalman_observation_variance,
        )
        static_z = causal_zscore(static_spread.spread, int(frozen_params["trailing_z_window"]))
        kalman_z = causal_zscore(kalman_spread.spread, int(frozen_params["trailing_z_window"]))
        static_positions = generate_positions(
            static_z, frozen_params["entry_z"], frozen_params["exit_abs_z"]
        ).loc[test_idx]
        kalman_positions = generate_positions(
            kalman_z, frozen_params["entry_z"], frozen_params["exit_abs_z"]
        ).loc[test_idx]

        window_result.test_results["static_batch_ols"] = _run_all_cost_scenarios(
            spread=static_spread, prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b], positions=static_positions,
            allocated_nav=allocated_nav, gross_notional_multiple=gross_notional_multiple,
            pair_label=window_result.selected_pair,
        )
        window_result.test_results["sequential_filtered_kalman"] = _run_all_cost_scenarios(
            spread=kalman_spread, prices_y=combined_for_spread[sym_a],
            prices_x=combined_for_spread[sym_b], positions=kalman_positions,
            allocated_nav=allocated_nav, gross_notional_multiple=gross_notional_multiple,
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
