"""Pair backtest + Kalman diagnostics for D9 historical OOS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .analytics import compute_strategy_analytics
from .engine import DataStreamer, EventDrivenBacktester
from .execution import CostModel, ExecutionHandler, LimitOrderBook
from .portfolio import PortfolioLedger
from .research import (
    compare_hedge_models,
    fit_dynamic_kalman_hedge_ratio,
    fit_static_ols_hedge_ratio,
)
from .strategies import PairsTradingStrategy, PositionSizer
from .historical_oos_panel import _pair_ohlc_dict

def backtest_static_pair(
    panel: pd.DataFrame,
    *,
    symbol_a: str,
    symbol_b: str,
    hedge_ratio: float,
    entry_z: float,
    exit_z: float,
    z_window: int,
    starting_cash: float,
    cost_model: CostModel,
    position_sizer: PositionSizer,
) -> Dict[str, Any]:
    data = _pair_ohlc_dict(panel, symbol_a, symbol_b)
    if min(len(df) for df in data.values()) < z_window + 5:
        return {
            "pair": f"{symbol_a}/{symbol_b}",
            "n_bars": int(min(len(df) for df in data.values())),
            "n_fills": 0,
            "analytics": None,
            "notes": "insufficient_bars",
        }
    streamer = DataStreamer(data, missing_bar_policy="error")
    strategy = PairsTradingStrategy(
        ticker_a=symbol_a,
        ticker_b=symbol_b,
        hedge_ratio=hedge_ratio,
        window=z_window,
        entry_z=entry_z,
        exit_z=exit_z,
        position_sizer=position_sizer,
    )
    execution = ExecutionHandler(LimitOrderBook(), cost_model=cost_model)
    portfolio = PortfolioLedger(starting_cash=starting_cash)
    bt = EventDrivenBacktester(streamer, strategy, execution, portfolio)
    # Suppress noisy prints from engine during batch study
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        bt.run()

    history = pd.DataFrame(bt.portfolio_history)
    if history.empty:
        analytics = None
    else:
        history = history.copy()
        history["returns"] = history["nav"].pct_change().fillna(0.0)
        per_bar_costs = pd.Series(
            [
                float(history["commissions"].iloc[i] + history["transaction_costs"].iloc[i])
                for i in range(len(history))
            ],
            dtype=float,
        )
        first_cost = (
            float(history["commissions"].iloc[0] + history["transaction_costs"].iloc[0])
            if len(history)
            else 0.0
        )
        analytics = compute_strategy_analytics(
            equity_curve=history["nav"],
            returns=history["returns"],
            gross_exposure=history.get("gross_exposure", pd.Series(dtype=float)),
            transaction_costs=per_bar_costs.diff().fillna(first_cost),
        )

    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "hedge_ratio": float(hedge_ratio),
        "estimation_mode": "static_batch_ols",
        "n_bars": int(len(history)) if not history.empty else 0,
        "n_fills": int(len(execution.fills)),
        "final_nav": None if history.empty else float(history["nav"].iloc[-1]),
        "starting_cash": float(starting_cash),
        "analytics": analytics,
        "cost_scenario": cost_model.scenario_label,
    }

def summarize_selection(selection: pd.DataFrame, top_n: int) -> Dict[str, Any]:
    if selection.empty:
        return {
            "n_tests": 0,
            "n_rejected": 0,
            "top_pairs": [],
            "diagnostic_ranked_by_qvalue": [],
            "min_qvalue": None,
            "n_raw_eg_lt_alpha": 0,
            "n_raw_adf_lt_alpha": 0,
        }

    def _row_dict(row: Any) -> Dict[str, Any]:
        return {
            "symbol_a": row.symbol_a,
            "symbol_b": row.symbol_b,
            "sector": getattr(row, "sector", None),
            "qvalue": float(row.qvalue),
            "engle_granger_pvalue": float(row.engle_granger_pvalue),
            "adf_pvalue": float(row.adf_pvalue),
            "hedge_ratio": float(row.hedge_ratio),
            "half_life": float(row.half_life),
            "rejected": bool(row.rejected),
            "estimation_mode": getattr(row, "estimation_mode", "static_batch_ols"),
        }

    rejected = selection.loc[selection["rejected"]].head(top_n)
    top_pairs = [_row_dict(row) for row in rejected.itertuples(index=False)]
    diagnostic = [_row_dict(row) for row in selection.head(top_n).itertuples(index=False)]
    return {
        "n_tests": (
            int(selection["test_count"].iloc[0])
            if "test_count" in selection
            else len(selection)
        ),
        "n_rejected": int(selection["rejected"].sum()),
        "top_pairs": top_pairs,
        "diagnostic_ranked_by_qvalue": diagnostic,
        "min_qvalue": float(selection["qvalue"].min()),
        "n_raw_eg_lt_alpha": int((selection["engle_granger_pvalue"] < 0.05).sum()),
        "n_raw_adf_lt_alpha": int((selection["adf_pvalue"] < 0.05).sum()),
    }

def kalman_diagnostics(
    panel: pd.DataFrame,
    pairs: Sequence[Mapping[str, Any]],
    *,
    process_variance: float,
    observation_variance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        sym_a = str(pair["symbol_a"])
        sym_b = str(pair["symbol_b"])
        if sym_a not in panel.columns or sym_b not in panel.columns:
            continue
        y = panel[sym_a].dropna()
        x = panel[sym_b].reindex(y.index).dropna()
        aligned = pd.concat([y, x], axis=1).dropna()
        if len(aligned) < 30:
            rows.append({"pair": f"{sym_a}/{sym_b}", "status": "insufficient_bars"})
            continue
        static = fit_static_ols_hedge_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])
        dyn = fit_dynamic_kalman_hedge_ratio(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
            process_variance=process_variance,
            observation_variance=observation_variance,
        )
        comparison = compare_hedge_models(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
            process_variance=process_variance,
            observation_variance=observation_variance,
        )
        rows.append(
            {
                "pair": f"{sym_a}/{sym_b}",
                "static_estimation_mode": static.estimation_mode,
                "dynamic_estimation_mode": dyn.estimation_mode,
                "static_hedge_ratio": float(static.slope),
                "kalman_beta_last": float(dyn.beta_path.iloc[-1]),
                "kalman_beta_mean": float(dyn.beta_path.mean()),
                "kalman_beta_std": (
                    float(dyn.beta_path.std(ddof=1)) if len(dyn.beta_path) > 1 else 0.0
                ),
                "n_obs": int(len(aligned)),
                "comparison_rows": int(len(comparison)),
            }
        )
    return rows

def aggregate_pair_analytics(pair_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sharpes: List[float] = []
    net_returns: List[float] = []
    max_dds: List[float] = []
    traded = 0
    for row in pair_results:
        analytics = row.get("analytics")
        if not analytics:
            continue
        traded += 1
        sharpes.append(float(analytics["sharpe_ratio"]))
        net_returns.append(float(analytics["net_return"]))
        max_dds.append(float(analytics["max_drawdown"]))
    if not traded:
        return {
            "n_pairs_traded": 0,
            "mean_sharpe": None,
            "median_sharpe": None,
            "mean_net_return": None,
            "median_net_return": None,
            "mean_max_drawdown": None,
            "total_fills": int(sum(int(r.get("n_fills", 0)) for r in pair_results)),
        }
    s = pd.Series(sharpes, dtype=float)
    r = pd.Series(net_returns, dtype=float)
    d = pd.Series(max_dds, dtype=float)
    return {
        "n_pairs_traded": traded,
        "mean_sharpe": float(s.mean()),
        "median_sharpe": float(s.median()),
        "mean_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "mean_max_drawdown": float(d.mean()),
        "total_fills": int(sum(int(x.get("n_fills", 0)) for x in pair_results)),
    }

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_text(text)
