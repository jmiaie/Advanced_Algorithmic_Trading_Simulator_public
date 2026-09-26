"""Automated evidence for the v4 holdout gate -- as opposed to a
one-off manual CLI run, which isn't reproducible evidence on its own.

Exercises scripts/run_statarb_wf_v4_study.py's actual main(). Synthetic
fixtures only -- for gate-logic verification, not a research claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_statarb_wf_v4_study as runner  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "DIA", "IWM", "XLB"]


def _write_synthetic_raw(raw_dir: Path, *, end: str = "2025-12-31") -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-02", end)
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


def _write_config(
    config_path: Path,
    *,
    status: str,
    dataset_id: str,
    insufficient_trades_fallback: dict | None = None,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": "test_v4",
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
    if insufficient_trades_fallback is not None:
        config["selection_objective"] = {
            "insufficient_trades_fallback": insufficient_trades_fallback
        }
    config_path.write_text(yaml.safe_dump(config))


def test_v4_runner_refuses_holdout_when_not_frozen(tmp_path, capsys):
    dataset_id = "test_dataset_v4"
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
    assert not (tmp_path / "results" / "statarb_hist_etf_wf_v4_holdout_2025.json").exists()


def test_v4_runner_dev_val_run_never_scores_2025_rows(tmp_path):
    dataset_id = "test_dataset_v4"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, status="pre-registered", dataset_id=dataset_id)

    full_panel = runner.load_close_panel(raw_dir, SYMBOLS)
    assert full_panel.index.max() >= pd.Timestamp("2025-12-31")

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
            assert pd.Timestamp(window["test_end"]) < pd.Timestamp(
                "2025-01-01"
            ), f"{artifact.name} scored a window extending into 2025"


def test_v4_runner_frozen_holdout_happy_path_no_reselection(tmp_path):
    """End-to-end synthetic happy path for the frozen-config holdout run:
    a --allow-holdout invocation against a frozen-for-holdout config must
    (a) not crash on the bucketing assertion (a real, independently-flagged
    bug: the old assertion compared the 2025-only bucket's size against
    len(study.windows) over the FULL 2015-2025 panel, which can never be
    equal), and (b) produce a 2025-only artifact whose every window's
    frozen_params are exactly the pre-registered fallback, never a
    grid-selected value -- proving the holdout path never reopens the
    signal/Kalman hyperparameter grid inside 2025, per this config's own
    no_retune_after_freeze / no_2025_access_before_final_configuration_frozen
    constraints. No real market data is used or touched."""
    import json

    from stat_arb_engine.wf_v4_orch import FALLBACK_PARAMS

    dataset_id = "test_dataset_v4"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, status="frozen-for-holdout", dataset_id=dataset_id)

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
    assert exit_code == 0

    artifacts = list((tmp_path / "results").glob("*holdout_2025.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())

    for window in payload["windows"]:
        assert pd.Timestamp(window["test_start"]).year == 2025
        if window["no_trade"]:
            continue
        assert window["used_fallback"] is True
        assert window["validation_ols_sharpe"] is None
        assert window["validation_kalman_sharpe"] is None
        assert window["frozen_params"] == {
            "entry_z": FALLBACK_PARAMS["entry_z"],
            "exit_abs_z": FALLBACK_PARAMS["exit_abs_z"],
            "trailing_z_window": FALLBACK_PARAMS["trailing_z_window"],
            "kalman_process_variance": FALLBACK_PARAMS["kalman_process_variance"],
        }


def test_v4_runner_holdout_empty_bucket_guard_actually_fails(tmp_path):
    """Regression for a real, independently-flagged defect distinct from
    the bucketing bug fixed above: the previous --allow-holdout guard
    compared the holdout_2025 bucket's size against a count built from the
    exact same test_start.year == 2025 predicate used to build that
    bucket -- tautologically equal, so it could never fail. Consequence: a
    --allow-holdout run whose panel produces ZERO 2025 test windows exited
    0, printed only a skip message, and wrote no artifact -- a false
    success on the one-time evaluation. This proves the replacement guard
    (holdout bucket must be non-empty) actually raises in that scenario,
    and that no artifact is written when it does. Synthetic data ending
    before 2025 entirely -- no real 2025 data is used, per the pre-registered spec."""
    dataset_id = "test_dataset_v4"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir, end="2023-12-31")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, status="frozen-for-holdout", dataset_id=dataset_id)

    with pytest.raises(AssertionError, match="holdout_2025 bucket is empty"):
        runner.main(
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

    assert not list((tmp_path / "results").glob("*.json"))


def test_v4_runner_holdout_uses_config_fallback_not_python_constant(tmp_path):
    """Regression for a real, independently-flagged provenance gap: the
    holdout path used wf_v4_orch.FALLBACK_PARAMS (a Python constant)
    directly, not the pre-registered config's own
    selection_objective.insufficient_trades_fallback block whose sha256 is
    what's actually recorded in the ledger's config_sha256 field. Both
    happened to hold the same four numbers, but nothing wired them
    together -- a future config edit to that block (a legitimate,
    pre-freeze action) would have silently kept using the OLD Python
    values instead. This end-to-end test uses a config whose fallback
    values DIFFER from FALLBACK_PARAMS and confirms the produced holdout
    artifact carries the YAML values, not the Python constant. Synthetic
    data only, per the pre-registered spec."""
    import json

    from stat_arb_engine.wf_v4_orch import FALLBACK_PARAMS

    custom_fallback = {
        "entry_z": 3.0,
        "exit_abs_z": 0.75,
        "trailing_z_window": 55,
        "kalman_process_variance": 2.5e-4,
    }
    assert custom_fallback != FALLBACK_PARAMS  # the test is meaningless otherwise

    dataset_id = "test_dataset_v4"
    raw_dir = tmp_path / "data" / "raw" / dataset_id
    _write_synthetic_raw(raw_dir)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        status="frozen-for-holdout",
        dataset_id=dataset_id,
        insufficient_trades_fallback=custom_fallback,
    )

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
    assert exit_code == 0

    artifacts = list((tmp_path / "results").glob("*holdout_2025.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())

    qualifying = [w for w in payload["windows"] if not w["no_trade"]]
    assert qualifying  # the fixture must actually reach a qualifying window
    for window in qualifying:
        assert window["frozen_params"] == {
            "entry_z": custom_fallback["entry_z"],
            "exit_abs_z": custom_fallback["exit_abs_z"],
            "trailing_z_window": custom_fallback["trailing_z_window"],
            "kalman_process_variance": custom_fallback["kalman_process_variance"],
        }


def test_v4_orch_fallback_params_constant_matches_pre_registered_config():
    """Drift guard: the pre-registered production config
    (configs/experiments/statarb_historical_etf_wf_v4.yaml) is the actual
    source of truth for the insufficient-trades fallback / holdout freeze
    (see the two tests above); wf_v4_orch.FALLBACK_PARAMS is now only the
    default used when a config omits that block entirely. This test fails
    loudly if the two are ever allowed to silently diverge -- e.g. someone
    edits the production config's fallback block without also updating (or
    deliberately leaving stale, with a comment) the Python default, or vice
    versa -- rather than that divergence going unnoticed until it changes
    which artifact a real run produces."""
    from stat_arb_engine.wf_v4_orch import FALLBACK_PARAMS

    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "experiments" / "statarb_historical_etf_wf_v4.yaml"
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_fallback = experiment["selection_objective"]["insufficient_trades_fallback"]

    assert {
        "entry_z": float(config_fallback["entry_z"]),
        "exit_abs_z": float(config_fallback["exit_abs_z"]),
        "trailing_z_window": float(config_fallback["trailing_z_window"]),
        "kalman_process_variance": float(config_fallback["kalman_process_variance"]),
    } == {k: float(v) for k, v in FALLBACK_PARAMS.items()}
