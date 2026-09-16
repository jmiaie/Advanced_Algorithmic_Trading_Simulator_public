#!/usr/bin/env python3
"""Run Directive #9 D9-B v3 conforming walk-forward study on frozen local
data only.

Without --allow-holdout, 2025 rows are sliced out of the panel before the
study ever runs -- not just gated by a status check -- so the holdout period
is structurally unreachable during DEV/2024-validation execution.

Examples:
  # Development (2015-2023) + validation (2024); holdout structurally excluded
  python scripts/run_statarb_wf_v3_study.py

  # Holdout (2025) only after YAML status is frozen-for-holdout
  python scripts/run_statarb_wf_v3_study.py --allow-holdout
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from stat_arb_engine.wf_v3_orch import WindowResult, run_walk_forward_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_close_panel(raw_dir: Path, symbols: list[str]) -> pd.DataFrame:
    frames: Dict[str, pd.Series] = {}
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen raw CSV: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        rename = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=rename)
        if "close" not in df.columns:
            raise ValueError(f"{path} missing a close column; have {list(df.columns)}")
        frames[symbol] = df["close"].sort_index()
        frames[symbol] = frames[symbol][~frames[symbol].index.duplicated(keep="first")]
    panel = pd.DataFrame(frames)
    if not panel.index.is_monotonic_increasing:
        panel = panel.sort_index()
    return panel


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_artifact(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def window_to_dict(w: WindowResult) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "window_index": w.window_index,
        "formation_start": str(w.formation_start.date()),
        "formation_end": str(w.formation_end.date()),
        "validation_start": str(w.validation_start.date()),
        "validation_end": str(w.validation_end.date()),
        "test_start": str(w.test_start.date()),
        "test_end": str(w.test_end.date()),
        "n_candidate_tests": w.selection.n_candidate_tests,
        "n_fdr_survivors": w.selection.n_fdr_survivors,
        "no_trade": w.no_trade,
        "selected_pair": w.selected_pair,
        "frozen_params": w.frozen_params,
        "used_fallback": w.used_fallback,
        "validation_ols_sharpe": w.validation_ols_sharpe,
        "validation_kalman_sharpe": w.validation_kalman_sharpe,
    }
    for model_name, scenarios in w.test_results.items():
        out[model_name] = {}
        for scenario_label, result in scenarios.items():
            out[model_name][scenario_label] = {
                "n_trades": result.n_trades,
                "gross_return": result.gross_return,
                "net_return": result.net_return,
                "annualized_vol": result.annualized_vol,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "turnover": result.turnover,
                "cost_drag": result.cost_drag,
                "avg_holding_period_days": result.avg_holding_period_days,
                "gross_exposure_mean": result.gross_exposure_mean,
                "net_exposure_mean": result.net_exposure_mean,
            }
    return out


def summarize_bucket(windows: List[WindowResult], label: str) -> Dict[str, Any]:
    qualifying = [w for w in windows if not w.no_trade]
    param_distribution: Dict[str, List[float]] = {
        "entry_z": [], "exit_abs_z": [], "trailing_z_window": [], "kalman_process_variance": [],
    }
    fallback_count = 0
    for w in qualifying:
        if w.frozen_params:
            for k in param_distribution:
                param_distribution[k].append(w.frozen_params[k])
        if w.used_fallback:
            fallback_count += 1
    return {
        "period_name": label,
        "n_windows": len(windows),
        "n_no_trade_windows": sum(1 for w in windows if w.no_trade),
        "n_qualifying_windows": len(qualifying),
        "n_fallback_windows": fallback_count,
        "selected_pairs": [w.selected_pair for w in qualifying],
        "param_distribution": param_distribution,
        "windows": [window_to_dict(w) for w in windows],
    }


def _append_ledger(path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "experiment_id", "repo", "branch", "dataset_id", "config_path", "status",
        "period_name", "period_start", "period_end", "horizons", "seed", "primary_symbol",
        "artifact_path", "artifact_sha256", "key_metrics_json", "notes", "created_utc",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/statarb_historical_etf_wf_v3.yaml"),
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results/historical_oos"))
    parser.add_argument("--ledger", type=Path, default=Path("research/experiment-ledger.csv"))
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args(argv)

    root = _repo_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    status = str(experiment["status"])
    dataset_id = str(experiment["dataset_id"])

    groups = experiment["universe"]["groups"]
    group_membership = {sym: group for group, syms in groups.items() for sym in syms}
    symbols = sorted(group_membership)

    raw_dir = args.raw_dir or (root / "data" / "raw" / dataset_id)
    if not raw_dir.is_dir():
        print(f"ERROR: frozen raw dir missing: {raw_dir}", file=sys.stderr)
        print("Acquire locally with scripts/acquire_yf_stat_arb_etfs_daily.py.", file=sys.stderr)
        return 2

    panel = load_close_panel(raw_dir, symbols)

    if args.allow_holdout:
        if status not in {"frozen-for-holdout"}:
            print(
                f"ERROR: refuse holdout; experiment status is '{status}', "
                "need frozen-for-holdout",
                file=sys.stderr,
            )
            return 3
        eval_panel = panel
        print("Holdout authorized: full panel through 2025 in scope.")
    else:
        # Structural exclusion, not just a status check: 2025 rows never
        # reach the study at all for a DEV/validation-only run.
        eval_panel = panel.loc[panel.index < pd.Timestamp("2025-01-01")]
        print(f"DEV+2024-validation only: panel sliced to {eval_panel.index.max().date()} "
              "(2025 rows excluded).")

    wf_cfg = experiment["walk_forward"] if "walk_forward" in experiment else {}
    signals = experiment["signals"]
    kalman = experiment["kalman"]
    sizing = experiment["sizing"]

    study = run_walk_forward_study(
        eval_panel,
        group_membership,
        formation_size=int(wf_cfg.get("formation_size", 504)),
        validation_size=int(wf_cfg.get("validation_size", 126)),
        test_size=int(wf_cfg.get("test_size", 63)),
        step_size=int(wf_cfg.get("step_size", 63)),
        grid_entry_z=tuple(signals["entry_z_candidates"]),
        grid_exit_abs_z=tuple(signals["exit_abs_z_candidates"]),
        grid_z_window=tuple(signals["trailing_z_window_candidates"]),
        grid_kalman_process_variance=tuple(kalman["process_variance_candidates"]),
        kalman_observation_variance=float(kalman["observation_variance"]),
        fdr_alpha=float(experiment["selection"]["fdr_alpha"]),
        adf_alpha=float(experiment["selection"]["adf_alpha"]),
        allocated_nav=1_000_000.0,
        gross_notional_multiple=float(sizing["gross_notional_multiple_of_allocated_nav"]),
    )

    results_dir = args.results_dir if args.results_dir.is_absolute() else root / args.results_dir
    ledger_path = args.ledger if args.ledger.is_absolute() else root / args.ledger

    # Three-way, exhaustive, non-overlapping bucketing by test-block calendar
    # coverage. A window's test block can itself cross a calendar-year
    # boundary (several do, e.g. Oct-Jan windows every year); that alone
    # doesn't make it ambiguous -- only a window whose test days fall on
    # BOTH sides of the DEV/VAL-2024 reporting split (i.e. some 2023 and
    # some 2024 sessions in the same 63-session test block) is genuinely
    # mixed and gets its own bucket rather than being folded into either
    # "DEV 2015-2023" or "VAL 2024" as if it were purely one or the other.
    if args.allow_holdout:
        buckets = [("holdout_2025", [w for w in study.windows if w.test_start.year == 2025])]
    else:
        buckets = [
            ("dev_formation",
             [w for w in study.windows if w.test_start.year <= 2023 and w.test_end.year <= 2023]),
            ("boundary_2023_2024",
             [w for w in study.windows if w.test_start.year <= 2023 < w.test_end.year <= 2024]),
            ("val_2024",
             [w for w in study.windows if w.test_start.year >= 2024 and w.test_end.year <= 2024]),
        ]
    assert sum(len(ws) for _, ws in buckets) == len(study.windows), (
        "bucketing dropped or double-counted a window"
    )

    for period_name, windows in buckets:
        if not windows:
            print(f"  {period_name}: no windows in range, skipping")
            continue
        summary = summarize_bucket(windows, period_name)
        experiment_id = f"statarb_hist_etf_wf_v3_{period_name}"
        artifact_path = results_dir / f"{experiment_id}.json"
        artifact_sha = write_artifact(artifact_path, summary)
        print(
            f"{experiment_id}: n_windows={summary['n_windows']} "
            f"qualifying={summary['n_qualifying_windows']} "
            f"no_trade={summary['n_no_trade_windows']} "
            f"fallback={summary['n_fallback_windows']}"
        )
        _append_ledger(
            ledger_path,
            {
                "experiment_id": experiment_id,
                "repo": "Advanced_Algorithmic_Trading_Simulator_public",
                "branch": "research/historical-oos-study",
                "dataset_id": dataset_id,
                "config_path": str(config_path.relative_to(root)),
                "status": status,
                "period_name": period_name,
                "period_start": str(windows[0].test_start.date()),
                "period_end": str(windows[-1].test_end.date()),
                "horizons": "",
                "seed": "0",
                "primary_symbol": "",
                "artifact_path": str(artifact_path.relative_to(root)),
                "artifact_sha256": artifact_sha,
                "key_metrics_json": json.dumps(
                    {
                        "n_windows": summary["n_windows"],
                        "n_qualifying_windows": summary["n_qualifying_windows"],
                        "n_no_trade_windows": summary["n_no_trade_windows"],
                        "n_fallback_windows": summary["n_fallback_windows"],
                    },
                    sort_keys=True,
                ),
                "notes": (
                    "D9-B v3 conforming walk-forward (per-window pair rediscovery; "
                    "fair OLS-vs-Kalman; fixed_gross_notional sizing; GROSS/BASE/STRESS "
                    "costs)."
                ),
                "created_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        print(f"  wrote {artifact_path} sha256={artifact_sha}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
