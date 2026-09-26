"""Automated evidence for the v3 holdout gate -- as opposed to a
one-off manual CLI run, which isn't reproducible evidence on its own.

Exercises scripts/run_statarb_wf_v3_study.py's actual main(), not the
older stat_arb_engine.historical_oos.run_period_study path (a different,
v1-era module that a prior draft mistakenly cited as v3 gate evidence).

Synthetic fixtures only -- for gate-logic verification, not a research
claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_statarb_wf_v3_study as runner  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "DIA", "IWM", "XLB"]


def _write_synthetic_raw(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-02", "2025-12-31")
    for sym in SYMBOLS:
        close = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.001,
                "Low": close * 0.999,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        df.index.name = "Date"
        df.to_csv(raw_dir / f"{sym}.csv")


def _write_config(config_path: Path, *, status: str, dataset_id: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": "test_v3",
        "status": status,
        "dataset_id": dataset_id,
        "universe": {"groups": {"GROUP_A": SYMBOLS}},
        "walk_forward": {
            "formation_size": 100,
            "validation_size": 30,
            "test_size": 20,
            "step_size": 20,
        },
        "signals": {
            "entry_z_candidates": [2.0],
            "exit_abs_z_candidates": [0.5],
            "trailing_z_window_candidates": [20],
        },
        "kalman": {"process_variance_candidates": [1e-4], "observation_variance": 1e-2},
        "selection": {"fdr_alpha": 0.5, "adf_alpha": 0.5},
        "sizing": {"gross_notional_multiple_of_allocated_nav": 1.0},
    }
    config_path.write_text(yaml.safe_dump(config))


def test_v3_runner_refuses_holdout_when_not_frozen(tmp_path, capsys):
    dataset_id = "test_dataset_v3"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, status="pre-registered", dataset_id=dataset_id)

    exit_code = runner.main(
        [
            "--config",
            str(config_path),
            "--raw-dir",
            str(raw_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--ledger",
            str(tmp_path / "ledger.csv"),
            "--allow-holdout",
        ]
    )

    assert exit_code == 3
    err = capsys.readouterr().err
    assert "refuse holdout" in err
    assert "pre-registered" in err
    # No 2025 artifact must be written when the gate refuses.
    assert not (tmp_path / "results" / "statarb_hist_etf_wf_v3_holdout_2025.json").exists()


def test_v3_runner_dev_val_run_never_scores_2025_rows(tmp_path):
    """The panel read from disk includes all rows through 2025-12-31 (the
    raw CSVs are not truncated), but the eval_panel passed into the study
    is sliced to exclude every 2025 row before any pair selection or
    backtest runs. This is the precise claim: 2025 values are read into
    memory as part of loading the full file, then structurally excluded
    before they can be used in any computation -- not that the file on
    disk somehow lacks 2025 rows."""
    dataset_id = "test_dataset_v3"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, status="pre-registered", dataset_id=dataset_id)

    full_panel = runner.load_close_panel(raw_dir, SYMBOLS)
    assert full_panel.index.max() >= pd.Timestamp("2025-12-31")  # 2025 rows ARE on disk/loaded

    exit_code = runner.main(
        [
            "--config",
            str(config_path),
            "--raw-dir",
            str(raw_dir),
            "--results-dir",
            str(tmp_path / "results"),
            "--ledger",
            str(tmp_path / "ledger.csv"),
        ]
    )
    assert exit_code == 0

    import json

    for artifact in (tmp_path / "results").glob("*.json"):
        payload = json.loads(artifact.read_text())
        for window in payload.get("windows", []):
            assert pd.Timestamp(window["test_end"]) < pd.Timestamp("2025-01-01"), (
                f"{artifact.name} scored a window extending into 2025"
            )
