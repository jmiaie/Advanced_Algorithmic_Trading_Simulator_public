"""Study A2: end-to-end positive control through the real v4 walk-forward runner.

A2a writes a synthetic 19-symbol close panel (one planted cointegrated pair,
everything else independent geometric random walks) as per-symbol CSVs, then
runs ``scripts/run_statarb_wf_v4_study.py`` -- the unmodified production runner,
reading the frozen v4 config -- in DEV mode and in ``--allow-holdout`` mode,
writing its artifacts and ledger rows to a temporary directory (never to
results/historical_oos/ or research/experiment-ledger.csv).

A2b isolates the step that populates ``validation_ols_sharpe``: the production
``wf_v4_orch._validation_grid_search_signal_params`` (27-point grid, >= 10
validation trades required) on one planted pair, as a function of half-life.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

from stat_arb_engine.wf_v3_costs import BASE
from stat_arb_engine.wf_v4_orch import MIN_VALIDATION_TRADES, _validation_grid_search_signal_params

from .screen_power import FROZEN_CONFIG, MASTER_SEED, REPO_ROOT, START_PRICE, ar1_spread

PLANTED_A2 = ("XLB", "XLE")  # y = beta * x + s ; XLB is the production regressand


def _runner_module() -> Any:
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("run_statarb_wf_v4_study")


def synthetic_panel(
    *,
    half_life: float,
    seed: int,
    start: str = "2015-01-02",
    end: str = "2025-12-31",
    sigma: float = 0.01,
    beta: float = 1.0,
    spread_sd: float = 1.0,
) -> pd.DataFrame:
    cfg = yaml.safe_load(FROZEN_CONFIG.read_text(encoding="utf-8"))
    symbols = sorted(s for syms in cfg["universe"]["groups"].values() for s in syms)
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 2, seed]))
    logs = np.cumsum(rng.normal(0.0, sigma, size=(len(index), len(symbols))), axis=0)
    panel = pd.DataFrame(START_PRICE * np.exp(logs), index=index, columns=symbols)
    eps = spread_sd * rng.normal(0.0, 1.0, size=len(index))
    sym_y, sym_x = PLANTED_A2
    panel[sym_y] = beta * panel[sym_x].to_numpy() + ar1_spread(eps, half_life)
    return panel


def write_raw_csvs(panel: pd.DataFrame, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sym in panel.columns:
        frame = pd.DataFrame({"Close": panel[sym]})
        frame.index.name = "Date"
        frame.to_csv(raw_dir / f"{sym}.csv")


def _summarise_bucket(payload: Dict[str, Any]) -> Dict[str, Any]:
    planted_key = "/".join(PLANTED_A2)
    windows = payload["windows"]
    rows = []
    for w in windows:
        ols = w.get("static_batch_ols", {}).get("BASE", {})
        rows.append(
            {
                "window_index": w["window_index"],
                "test_start": w["test_start"],
                "n_candidate_tests": w["n_candidate_tests"],
                "n_fdr_survivors": w["n_fdr_survivors"],
                "selected_pair": w["selected_pair"],
                "used_fallback": w["used_fallback"],
                "validation_ols_sharpe": w["validation_ols_sharpe"],
                "validation_kalman_sharpe": w["validation_kalman_sharpe"],
                "frozen_params": w["frozen_params"],
                "test_static_ols_base_trades": ols.get("n_trades"),
                "test_static_ols_base_net_return": ols.get("net_return"),
            }
        )
    return {
        "n_windows": len(windows),
        "n_planted_selected": sum(r["selected_pair"] == planted_key for r in rows),
        "n_other_selected": sum(
            r["selected_pair"] not in (None, planted_key) for r in rows
        ),
        "n_validation_ols_sharpe_populated": sum(
            r["validation_ols_sharpe"] is not None for r in rows
        ),
        "n_validation_kalman_sharpe_populated": sum(
            r["validation_kalman_sharpe"] is not None for r in rows
        ),
        "n_used_fallback": sum(bool(r["used_fallback"]) for r in rows),
        "n_windows_with_test_trades": sum(
            (r["test_static_ols_base_trades"] or 0) > 0 for r in rows
        ),
        "windows": rows,
    }


def run_end_to_end(
    *, half_life: float, seed: int, start: str = "2015-01-02", end: str = "2025-12-31"
) -> Dict[str, Any]:
    """Run the production runner (DEV and holdout modes) on a synthetic panel."""
    runner = _runner_module()
    panel = synthetic_panel(half_life=half_life, seed=seed, start=start, end=end)
    out: Dict[str, Any] = {"half_life": half_life, "seed": seed, "start": start, "end": end}
    with tempfile.TemporaryDirectory(prefix="power_a2_") as tmp:
        tmp_path = Path(tmp)
        raw = tmp_path / "raw"
        write_raw_csvs(panel, raw)
        for mode, extra in (("dev", []), ("holdout", ["--allow-holdout"])):
            res_dir = tmp_path / f"results_{mode}"
            argv = [
                "--config",
                str(FROZEN_CONFIG),
                "--raw-dir",
                str(raw),
                "--results-dir",
                str(res_dir),
                "--ledger",
                str(tmp_path / f"ledger_{mode}.csv"),
                *extra,
            ]
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                warnings.simplefilter("ignore")
                code = runner.main(argv)
            buckets = {}
            for path in sorted(res_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                buckets[payload["period_name"]] = _summarise_bucket(payload)
            out[mode] = {"exit_code": code, "buckets": buckets}
    return out


def grid_fire_probability(
    half_lives: Sequence[float], reps: int, *, sigma: float = 0.01, beta: float = 1.0
) -> List[Dict[str, Any]]:
    """P(the production validation grid returns a selection) on one planted
    pair, formation 504 / validation 126 / test 63 bars, BASE costs."""
    rows: List[Dict[str, Any]] = []
    n = 504 + 126 + 63
    index = pd.bdate_range("2001-01-01", periods=n)
    for h in half_lives:
        fired = 0
        max_trades: List[int] = []
        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 3, rep]))
            x = START_PRICE * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
            eps = rng.normal(0.0, 1.0, n)
            y = beta * x + ar1_spread(eps, h)
            df = pd.DataFrame({"A": y, "B": x}, index=index)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, sharpe = _validation_grid_search_signal_params(
                    formation_prices=df.iloc[:504],
                    combined_for_spread=df,
                    validation_index=index[504:630],
                    sym_a="A",
                    sym_b="B",
                    grid_entry_z=(1.5, 2.0, 2.5),
                    grid_exit_abs_z=(0.25, 0.50, 0.75),
                    grid_z_window=(20, 40, 60),
                    allocated_nav=1_000_000.0,
                    gross_notional_multiple=1.0,
                    cost=BASE,
                )
            fired += params is not None
            max_trades.append(_max_validation_trades(df, index[504:630]))
        rows.append(
            {
                "half_life": h,
                "reps": reps,
                "grid_fired": fired,
                "p_grid_fires": fired / reps,
                "max_validation_trades_median": float(np.median(max_trades)),
                "max_validation_trades_p10": float(np.percentile(max_trades, 10)),
                "max_validation_trades_p90": float(np.percentile(max_trades, 90)),
                "min_validation_trades_required": MIN_VALIDATION_TRADES,
            }
        )
    return rows


def _max_validation_trades(df: pd.DataFrame, validation_index: pd.Index) -> int:
    """Largest validation trade count over the 27-point grid (production
    spread, z-score, position and backtest functions)."""
    from itertools import product

    from stat_arb_engine.wf_v3_backtest import (
        build_static_spread,
        causal_zscore,
        generate_positions,
    )
    from stat_arb_engine.wf_v4_backtest import simulate_pair_backtest

    spread = build_static_spread(df.iloc[:504], df, "A", "B")
    best = 0
    for ez, xz, zw in product((1.5, 2.0, 2.5), (0.25, 0.50, 0.75), (20, 40, 60)):
        pos = generate_positions(causal_zscore(spread.spread, zw), ez, xz)
        res = simulate_pair_backtest(
            prices_y=df["A"],
            prices_x=df["B"],
            spread=spread,
            decision_positions=pos,
            execution_index=validation_index,
            allocated_nav=1_000_000.0,
            gross_notional_multiple=1.0,
            cost=BASE,
            pair_label="A/B",
        )
        best = max(best, res.n_trades)
    return best
