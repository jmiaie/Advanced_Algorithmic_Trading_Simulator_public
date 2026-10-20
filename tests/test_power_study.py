"""Tests for the power / null-calibration study (research/power/, Study A)."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.power import screen_power as sp  # noqa: E402
from research.power.positive_control import run_end_to_end  # noqa: E402
from research.power.run_power_study import main as run_study  # noqa: E402
from stat_arb_engine.research import benjamini_hochberg  # noqa: E402


def test_design_matches_frozen_v4_config() -> None:
    design = sp.load_frozen_design()
    assert design.formation_size == 504
    assert design.n_candidate_tests == 55
    assert design.fdr_alpha == 0.05
    assert design.adf_alpha == 0.05
    assert len(design.symbols) == 19


def test_prefix_rule_is_the_per_rank_rule_and_under_rejects() -> None:
    p = [0.01, 0.03, 0.035, 0.9]
    # thresholds alpha*k/m = 0.0125, 0.025, 0.0375, 0.05
    old = sp.bh_reject_prefix_per_rank(p, 0.05)
    assert old.tolist() == [True, False, True, False]
    new = benjamini_hochberg(p, alpha=0.05)["rejected"].to_numpy(dtype=bool)
    assert new.tolist() == [True, True, True, False]


def test_prefix_rejections_are_a_subset_of_production_rejections() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = np.concatenate([rng.uniform(size=50), rng.uniform(0, 0.01, size=5)])
        old = sp.bh_reject_prefix_per_rank(p, 0.05)
        new = benjamini_hochberg(p, alpha=0.05)["rejected"].to_numpy(dtype=bool)
        assert not (old & ~new).any()


def test_ar1_spread_uses_half_life_parameterisation() -> None:
    eps = np.zeros(50)
    eps[0] = 1.0
    s = sp.ar1_spread(eps, half_life=10)
    phi = 0.5 ** (1 / 10)
    assert np.isclose(s[10] / s[0], 0.5)
    assert np.isclose(s[1] / s[0], phi)


def test_planting_changes_only_the_regressand_legs_and_uses_common_random_numbers() -> None:
    design = sp.load_frozen_design()
    legs = sp.null_legs(design, 0.01, 7)
    a = sp.plant_pairs(legs, 0.01, 7, k=3, half_life=10, beta=2.0)
    b = sp.plant_pairs(legs, 0.01, 7, k=3, half_life=40, beta=2.0)
    changed = [c for c in legs.columns if not np.allclose(a[c], legs[c])]
    assert sorted(changed) == sorted(y for y, _ in sp.PLANTED_PAIRS)
    # same innovations, different persistence
    eps = sp.spread_innovations(0.01, 7, 0, len(legs))
    assert np.allclose(a["XLB"] - 2.0 * legs["XLE"], sp.ar1_spread(eps, 10))
    assert np.allclose(b["XLB"] - 2.0 * legs["XLE"], sp.ar1_spread(eps, 40))


def test_memoized_fit_returns_identical_screen_results() -> None:
    design = sp.load_frozen_design()
    panel = sp.plant_pairs(sp.null_legs(design, 0.01, 3), 0.01, 3, 3, 10, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.screen_counts(panel, design, sp.PLANTED_PAIRS)
        with sp.memoized_production_fit():
            cached_first = sp.screen_counts(panel, design, sp.PLANTED_PAIRS)
            cached_again = sp.screen_counts(panel, design, sp.PLANTED_PAIRS)
    assert plain == cached_first == cached_again
    assert plain["n_tests"] == 55


def test_aggregate_and_mde() -> None:
    def rec(rep: int, h: float, tp: int) -> dict:
        rule = {"true_survivors": tp, "false_survivors": 0, "fdp": 0.0, "any_survivor": tp > 0}
        return {
            "sigma": 0.01, "k": 1, "half_life": h, "beta": 1.0, "rep": rep, "n_tests": 55,
            "current": rule, "prefix": rule, "bh_only_true": tp, "adf_only_true": tp,
            "selected_is_planted": tp > 0,
        }

    records = [rec(r, 5, 1) for r in range(10)] + [rec(r, 20, int(r < 9)) for r in range(10)]
    records += [rec(r, 40, int(r < 5)) for r in range(10)]
    cells = sp.aggregate(records)
    assert [c["current"]["power"] for c in cells] == [1.0, 0.9, 0.5]
    mde = [m for m in sp.mde_table(cells) if m["rule"] == "current"][0]
    assert mde["mde_half_life_days"] == 20.0


def test_smoke_simulation_under_30_seconds(tmp_path: Path) -> None:
    out = tmp_path / "smoke.json"
    t0 = time.time()
    assert run_study(["--smoke", "--out", str(out)]) == 0
    assert time.time() - t0 < 30
    payload = json.loads(out.read_text())
    assert (tmp_path / "smoke.png").exists()
    cells = {(c["k"], c["half_life"]): c for c in payload["a1"]["cells"]}
    assert cells[(0, None)]["n_reps"] == 2
    assert cells[(1, 5)]["current"]["planted_survivor_count"] == 2  # h=5: always found
    assert cells[(1, 120)]["current"]["power"] <= 0.5
    assert payload["a2"]["a2b_grid_fire_probability"][0]["reps"] == 2


def test_end_to_end_runner_positive_control_short_panel() -> None:
    """Strong planted pair through the unmodified v4 runner on a short panel:
    DEV mode selects it and populates validation_ols_sharpe; holdout mode
    selects it but leaves validation_ols_sharpe null by construction."""
    res = run_end_to_end(half_life=2, seed=0, start="2022-03-01")
    assert res["dev"]["exit_code"] == 0 and res["holdout"]["exit_code"] == 0
    dev = res["dev"]["buckets"]["val_2024"]
    assert dev["n_planted_selected"] == dev["n_windows"] == 1
    assert dev["n_validation_ols_sharpe_populated"] == 1
    hold = res["holdout"]["buckets"]["holdout_2025"]
    assert hold["n_planted_selected"] == hold["n_windows"] >= 1
    assert hold["n_validation_ols_sharpe_populated"] == 0
    assert hold["n_windows_with_test_trades"] >= 1


def test_runner_artifacts_never_touch_frozen_results() -> None:
    frozen = REPO_ROOT / "results" / "historical_oos"
    before = {p.name: p.stat().st_mtime_ns for p in frozen.iterdir()}
    ledger = REPO_ROOT / "research" / "experiment-ledger.csv"
    ledger_before = ledger.read_bytes()
    run_end_to_end(half_life=2, seed=1, start="2023-01-03")
    assert {p.name: p.stat().st_mtime_ns for p in frozen.iterdir()} == before
    assert ledger.read_bytes() == ledger_before
    assert isinstance(pd.Timestamp("2023-01-03"), pd.Timestamp)
