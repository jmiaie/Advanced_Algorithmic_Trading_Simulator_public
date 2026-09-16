"""Period study orchestration for D9 historical OOS."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd

from .execution import CostModel
from .research import as_of_frame, walk_forward_windows
from .strategies import PositionSizer
from .historical_oos_panel import (
    PeriodSpec,
    select_pairs_within_sectors,
    slice_period,
)
from .historical_oos_backtest import (
    aggregate_pair_analytics,
    backtest_static_pair,
    kalman_diagnostics,
    summarize_selection,
)

def run_period_study(
    *,
    full_panel: pd.DataFrame,
    formation: PeriodSpec,
    eval_period: PeriodSpec,
    sector_grouping: Mapping[str, str],
    config: Mapping[str, Any],
    allow_holdout: bool,
) -> Dict[str, Any]:
    status = str(config.get("status", ""))
    if eval_period.name == "holdout" and status != "frozen-for-holdout":
        raise RuntimeError(
            "Holdout evaluation blocked until experiment status is frozen-for-holdout"
        )
    if eval_period.name == "holdout" and not allow_holdout:
        raise RuntimeError("Pass allow_holdout=True after FINAL CONFIGURATION FROZEN")

    # Decision-time: pair selection and hedge fit on formation only.
    formation_panel = slice_period(full_panel, formation).dropna(axis=0, how="any")
    decision = as_of_frame(formation_panel, formation_panel.index.max())
    selection = select_pairs_within_sectors(
        decision,
        sector_grouping,
        fdr_alpha=float(config["selection"]["fdr_alpha"]),
        adf_alpha=float(config["selection"]["adf_alpha"]),
    )
    top_n = int(config["selection"]["top_n_pairs"])
    selection_summary = summarize_selection(selection, top_n)
    top_pairs = selection_summary["top_pairs"]

    eval_panel = slice_period(full_panel, eval_period)
    costs = config["costs"]
    cost_model = CostModel(
        commission_per_share=float(costs["commission_per_share"]),
        min_commission=float(costs["min_commission"]),
        spread_bps=float(costs["spread_bps"]),
        slippage_bps=float(costs["slippage_bps"]),
        impact_bps=float(costs["impact_bps"]),
        borrow_bps_per_day=float(costs.get("borrow_bps_per_day", 0.0)),
        scenario_label=str(costs.get("scenario_label", "stylized")),
    )
    sizer = PositionSizer(
        mode=str(config["execution"]["sizing_mode"]),
        quantity=int(config["execution"]["fixed_shares"]),
        gross_notional=float(config["execution"]["gross_notional"]),
    )

    pair_results: List[Dict[str, Any]] = []
    for pair in top_pairs:
        # Hedge ratio frozen from formation OLS (static_batch_ols).
        result = backtest_static_pair(
            eval_panel,
            symbol_a=pair["symbol_a"],
            symbol_b=pair["symbol_b"],
            hedge_ratio=float(pair["hedge_ratio"]),
            entry_z=float(config["signals"]["entry_z"]),
            exit_z=float(config["signals"]["exit_z"]),
            z_window=int(config["signals"]["z_window"]),
            starting_cash=float(config["execution"]["starting_cash"]),
            cost_model=cost_model,
            position_sizer=sizer,
        )
        pair_results.append(result)

    kalman = kalman_diagnostics(
        formation_panel,
        top_pairs,
        process_variance=float(config["kalman"]["process_variance"]),
        observation_variance=float(config["kalman"]["observation_variance"]),
    )

    # Lightweight walk-forward chronology check (window indices only; no per-window
    # Engle-Granger rescan — that would retune on overlapping slices and is O(windows*pairs)).
    wf_cfg = config.get("walk_forward", {})
    walk_forward_summary: Dict[str, Any]
    try:
        windows = walk_forward_windows(
            formation_panel,
            formation_size=int(wf_cfg.get("formation_size", 504)),
            validation_size=int(wf_cfg.get("validation_size", 126)),
            test_size=int(wf_cfg.get("test_size", 126)),
        )
        chronology_ok = True
        for window in windows:
            if not (
                window["formation"].max() < window["validation"].min()
                <= window["validation"].max()
                < window["test"].min()
            ):
                chronology_ok = False
                break
        walk_forward_summary = {
            "n_windows": int(len(windows)),
            "chronology_ok": chronology_ok,
            "selection_rescanned_per_window": False,
            "note": "Window chronology only; pair selection remains single formation-fit.",
        }
    except ValueError as exc:
        walk_forward_summary = {"n_windows": 0, "error": str(exc)}

    key_metrics = aggregate_pair_analytics(pair_results)
    key_metrics.update(
        {
            "n_tests_within_sector": selection_summary["n_tests"],
            "n_rejected_within_sector": selection_summary["n_rejected"],
            "n_top_pairs": len(top_pairs),
            "n_eval_bars": int(len(eval_panel)),
            "n_formation_bars": int(len(formation_panel)),
        }
    )

    return {
        "period_name": eval_period.name,
        "period_start": eval_period.start,
        "period_end": eval_period.end_inclusive,
        "formation_start": formation.start,
        "formation_end": formation.end_inclusive,
        "selection": selection_summary,
        "pair_results": pair_results,
        "kalman_diagnostics": kalman,
        "walk_forward": walk_forward_summary,
        "key_metrics": key_metrics,
        "same_timestamp_signals": True,
        "decision_time_pair_research": True,
        "hedge_models": ["static_batch_ols", "sequential_filtered_kalman"],
        "primary_trading_model": "static_batch_ols",
        "notes": (
            "Primary trading uses formation-fit static OLS hedge on eval window. "
            "Kalman reported as sequential diagnostic (estimation_mode labels), "
            "not as a separate live beta-trading path in this study."
        ),
    }
