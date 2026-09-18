#!/usr/bin/env python3
"""Independent claim cross-check for the D10-B publication pack.

Deliberately NOT the pack generator: no shared code, no shared helpers, no
reuse of its bucket table. This script re-reads the accepted artifacts with its
own parsing and asserts the CLAIM-REGISTER values directly, so that a defect in
the generator cannot mask a false claim (and vice versa).

Pure stdlib, zero network, no writes. Any failed assertion exits non-zero.

    python3 publication/stat-arb-study/scripts/claim_crosscheck.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]

V4 = "results/historical_oos/statarb_hist_etf_wf_v4_{}.json"
V3 = "results/historical_oos/statarb_hist_etf_wf_v3_{}.json"
HASHES = {
    "holdout_2025": "b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66",
    "dev_formation": "52a45f3b8bbd59ba0795442b3cc4ed58d1c2b2465b0b89f99d012f715e7e3146",
    "boundary_2023_2024": "e311a0e27b3be19ff66c496849a87717852f027c03fa4210aa641b67709b25f2",
    "val_2024": "d59c9747044262a2b736f455a5924fde13dedadb5ff8567d3964fa9b919dfbb8",
}
REQUIRED_DOCS = [
    "SOURCE-GATE.md",
    "TECHNICAL-PAPER.md",
    "RESULT-SOURCE-MAP.md",
    "reproducibility.json",
    "CASE-STUDY.md",
    "CLAIM-REGISTER.md",
    "QUANT-RED-TEAM.md",
    "CLAIM-RED-TEAM.md",
    "CITATION-RED-TEAM.md",
    "D10-STATUS.md",
]

CHECKS = 0


def ok(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(f"FAILED: {label}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    docs = {d for d in REQUIRED_DOCS if (PACK / d).exists()}
    ok(
        docs == set(REQUIRED_DOCS),
        f"all 10 deliverable docs present (missing {sorted(set(REQUIRED_DOCS) - docs)})",
    )

    b = {name: load(V4.format(name)) for name in HASHES}
    for name, digest in HASHES.items():
        ok(sha(ROOT / V4.format(name)) == digest, f"{name} artifact hash (C-01/C-12/C-17/C-18)")

    # --- 2025 bucket: the primary result -------------------------------------
    h = b["holdout_2025"]
    ok(h["period_name"] == "holdout_2025", "C-01 period_name")
    ok(h["n_windows"] == 3 and len(h["windows"]) == 3, "C-01 3 windows")
    ok(h["n_qualifying_windows"] == 0, "C-02 zero qualifying windows")
    ok(h["selected_pairs"] == [], "C-02 empty selected_pairs")
    ok(h["n_no_trade_windows"] == 3, "C-03 all windows no-trade")
    ok(h["n_fallback_windows"] == 0, "C-06 no fallback windows")
    ok(all(v == [] for v in h["param_distribution"].values()), "C-08 empty param distribution")
    ok(
        sorted(w["window_index"] for w in h["windows"]) == [30, 31, 32],
        "C-09 window indices 30/31/32",
    )
    expect_tests = {
        30: ("2025-01-08", "2025-04-09", "2024-07-10", "2025-01-07"),
        31: ("2025-04-10", "2025-07-11", "2024-10-08", "2025-04-09"),
        32: ("2025-07-14", "2025-10-09", "2025-01-08", "2025-07-11"),
    }
    for w in h["windows"]:
        e = expect_tests[w["window_index"]]
        ok(
            (w["test_start"], w["test_end"], w["validation_start"], w["validation_end"]) == e,
            f"C-09/C-10 window {w['window_index']} dates",
        )
        ok(w["validation_end"] < w["test_start"], f"C-10 causal order window {w['window_index']}")
        ok(w["n_candidate_tests"] == 55, f"C-04 55 candidate tests (window {w['window_index']})")
        ok(w["n_fdr_survivors"] == 0, f"C-05 zero survivors (window {w['window_index']})")
        ok(w["no_trade"] is True, f"C-03 no_trade true (window {w['window_index']})")
        ok(w["used_fallback"] is False, f"C-06 used_fallback false (window {w['window_index']})")
        ok(
            w["selected_pair"] is None and w["frozen_params"] is None,
            f"C-07 null pair/params (window {w['window_index']})",
        )
        ok(
            w["validation_ols_sharpe"] is None and w["validation_kalman_sharpe"] is None,
            "C-33 no hedge metrics in 2025",
        )
        banned = [
            k
            for k in w
            if any(t in k for t in ("sharpe", "return", "pnl", "drawdown", "turnover"))
            and not k.startswith("validation_")
        ]
        ok(
            banned == [],
            f"C-11 no return/risk metrics in 2025 window {w['window_index']} (found {banned})",
        )

    # --- pre-2025 buckets ----------------------------------------------------
    d = b["dev_formation"]
    ok(
        (
            d["n_windows"],
            d["n_qualifying_windows"],
            d["n_no_trade_windows"],
            d["n_fallback_windows"],
        )
        == (25, 4, 21, 4),
        "C-12/C-13 DEV counts",
    )
    traded = [w for w in d["windows"] if w["no_trade"] is False]
    ok(
        len(traded) == 4 and all(w["used_fallback"] is True for w in traded),
        "C-13 all traded DEV windows used fallback",
    )
    ok(sum(w["n_candidate_tests"] for w in d["windows"]) == 1339, "C-16 DEV candidate tests 1,339")
    ok(sum(w["n_fdr_survivors"] for w in d["windows"]) == 4, "C-16 DEV survivors 4")
    sp = sorted(d["selected_pairs"])
    ok(sp == sorted(["XLP/XLU", "XLP/XLU", "XLP/XLU", "XLU/XLV"]), f"C-15 DEV pairs {sp}")
    fb = d["windows"][1]["frozen_params"]
    ok(
        fb
        == {
            "entry_z": 2.0,
            "exit_abs_z": 0.5,
            "kalman_process_variance": 0.0001,
            "trailing_z_window": 40,
        },
        "C-14 fallback params",
    )
    ok(all(w["validation_ols_sharpe"] is None for w in d["windows"]), "C-31 OLS Sharpe null in DEV")
    kal = [
        w["validation_kalman_sharpe"]
        for w in d["windows"]
        if w["validation_kalman_sharpe"] is not None
    ]
    ok(
        len(kal) == 3
        and all(
            abs(a - b_) < 1e-12
            for a, b_ in zip(
                sorted(kal),
                sorted([-0.14987589534869653, 0.8398060100944159, -2.117478227370739]),
                strict=True,
            )
        ),
        f"C-32 DEV Kalman Sharpe trio {kal}",
    )

    bd, vl = b["boundary_2023_2024"], b["val_2024"]
    ok(
        bd["n_windows"] == 1 and bd["windows"][0]["window_index"] == 25,
        "C-17 boundary window index 25",
    )
    ok(
        (bd["windows"][0]["test_start"], bd["windows"][0]["test_end"])
        == ("2023-10-06", "2024-01-05"),
        "C-17 boundary dates",
    )
    ok(
        bd["n_no_trade_windows"] == 1 and bd["windows"][0]["n_candidate_tests"] == 55,
        "C-17 boundary 55 tests, no-trade",
    )
    ok(
        vl["n_windows"] == 3 and sorted(w["window_index"] for w in vl["windows"]) == [26, 27, 28],
        "C-18 val windows 26/27/28",
    )
    ok(
        vl["n_no_trade_windows"] == 3 and all(w["n_candidate_tests"] == 55 for w in vl["windows"]),
        "C-18 val all 55-test no-trade",
    )

    # --- totals --------------------------------------------------------------
    every = [w for name in HASHES for w in b[name]["windows"]]
    ok(len(every) == 32, "C-19 32 windows total")
    ok(sum(w["n_candidate_tests"] for w in every) == 1724, "C-19 1,724 candidate tests total")
    ok(sum(w["n_fdr_survivors"] for w in every) == 4, "C-19 4 survivors total")
    ok(sum(1 for w in every if w["no_trade"] is False) == 4, "C-19 4 traded windows total")
    ok(
        all(
            not any(t in k for t in ("pair_persistence", "hedge_variation"))
            for w in every
            for k in w
        ),
        "C-39 no persistence/hedge diagnostics",
    )

    # --- superseded lineage --------------------------------------------------
    for bucket in ("boundary_2023_2024", "val_2024"):
        ok(
            sha(ROOT / V3.format(bucket)) == sha(ROOT / V4.format(bucket)),
            f"C-20 v3==v4 bytes for {bucket}",
        )
    ok(
        sha(ROOT / V3.format("dev_formation")) != sha(ROOT / V4.format("dev_formation")),
        "C-20 v3!=v4 bytes for dev_formation",
    )
    v1 = load("results/historical_oos/statarb_hist_oos_v1_holdout_2025.json")
    ok(
        sha(ROOT / "results/historical_oos/statarb_hist_oos_v1_holdout_2025.json")
        == "4fd88a8204ecb4b92cc5017764fcaf0e159b20b202d971b9edc85d4128d6f271",
        "C-38 superseded v1 artifact hash",
    )
    ok(
        v1["n_symbols"] == 56 and v1["experiment_id"] == "statarb_hist_oos_v1_holdout_2025",
        "C-38 v1 is the different-universe (56-symbol) study",
    )
    ok(
        v1["result"]["period_name"] == "holdout" and v1["result"]["pair_results"] == [],
        "C-38 v1 label intact and 0 pairs selected",
    )
    ok(v1["result"]["selection"]["n_rejected"] == 0, "C-38 v1 zero FDR rejections")
    ok(
        len(
            sorted(
                p.name for p in (ROOT / "results/historical_oos").glob("statarb_hist_oos_v1_*.json")
            )
        )
        == 3,
        "C-20 v1 result artifacts all retained",
    )
    ok(
        len(
            sorted(
                p.name for p in (ROOT / "results/historical_oos").glob("statarb_hist_etf_wf_v3_*")
            )
        )
        == 4,
        "C-20 v3 result artifacts all retained",
    )
    ok(
        not list((ROOT / "results/historical_oos").glob("*v2*")),
        "C-40 v2 produced no result artifacts (config-only lineage)",
    )
    ok(
        sha(ROOT / "configs/experiments/statarb_historical_etf_wf_v2.yaml")
        == "0c6b8405011205bdce0fd49d50230b584e4faaa345ed4d3fbb1432cea142db11",
        "C-40 v2 config retained",
    )
    sup = list(csv.DictReader((ROOT / "research/experiment-ledger.csv").open(encoding="utf-8")))
    sids = [r["experiment_id"] for r in sup]
    ok(len(sup) == 23, f"C-40 ledger rows {len(sup)} == 23")
    ok(
        sids.count("statarb_hist_etf_wf_v2_INVALIDATED") == 1
        and sum(i.startswith("statarb_hist_oos_v1_") for i in sids) == 4,
        "C-40 superseded lineage rows retained (1 v2_INVALIDATED + 4 v1 rows)",
    )
    ok(
        sum(1 for i in sids if "SUPERSEDED_" in i) == 3,
        "C-40 explicit SUPERSEDED_* ledger rows retained",
    )

    # --- frozen config -------------------------------------------------------
    import re

    cfg = (ROOT / "configs/experiments/statarb_historical_etf_wf_v4.yaml").read_text(
        encoding="utf-8"
    )
    ok(
        sha(ROOT / "configs/experiments/statarb_historical_etf_wf_v4.yaml")
        == "5768fd10b63c0436f3ff5aa1863374b348cca5bf72ccfb7d1b01a7ae0d95376d",
        "C-21..C-27 frozen config hash",
    )
    ok(
        re.search(
            r"entry_z: 2\.0\s*\n\s*exit_abs_z: 0\.5\s*\n\s*"
            r"trailing_z_window: 40\s*\n\s*kalman_process_variance: 1\.0e-4",
            cfg,
        )
        is not None,
        "C-14/C-24 fallback block in config",
    )
    ok(re.search(r"min_trade_count: 10", cfg) is not None, "C-24 min_trade_count 10")
    ok(
        re.search(r"n_grid_selected_windows_total: 0", cfg) is not None,
        "C-13 zero grid-selected windows in freeze record",
    )
    ok(
        re.search(
            r"formation_size: 504\n\s*validation_size: 126\n\s*test_size: 63\n\s*step_size: 63", cfg
        )
        is not None,
        "C-22 walk-forward geometry",
    )
    for bps in (
        "spread_bps_per_side: 1.0",
        "spread_bps_per_side: 3.0",
        "borrow_bps_annualized: 50.0",
        "borrow_bps_annualized: 200.0",
    ):
        ok(bps in cfg, f"C-25 cost scenario value {bps}")
    ok(
        "Scenario assumptions, not measured historical execution costs" in cfg,
        "C-26 costs are assumptions",
    )
    ok("observation_variance: 1.0e-2" in cfg, "C-23 observation variance 1e-2")

    # --- dataset manifest ----------------------------------------------------
    man = load("data/manifests/yf_stat_arb_etfs_daily_2015_2025_v1.json")
    ok(
        man["sha256"]["dataset_canonical"]
        == "a4082398991400c171062ee4ff2b12dba87e84c4f0133462ed9ee2fbb13f9fa8",
        "C-30 canonical hash",
    )
    ok(man["sha256"]["dataset_canonical"] in cfg, "C-30 config freeze_record agrees with manifest")
    ok(sum(man["row_counts"].values()) == 52361, "C-29 52,361 rows")
    ok(sum(man["missing_counts"].values()) == 0, "C-29 zero missing rows")
    ok(len(man["row_counts"]) == 19, "C-21 19 symbols")
    ok(man.get("status") == "DATA FROZEN", "C-28 dataset frozen")

    # --- ledger --------------------------------------------------------------
    rows = list(csv.DictReader((ROOT / "research/experiment-ledger.csv").open(encoding="utf-8")))
    hold = [r for r in rows if r["experiment_id"] == "statarb_hist_etf_wf_v4_holdout_2025"]
    ok(len(hold) == 1, "C-34 exactly one 2025 ledger row")
    km = json.loads(hold[0]["key_metrics_json"])
    ok(
        km
        == {
            "config_sha256": "5768fd10b63c0436f3ff5aa1863374b348cca5bf72ccfb7d1b01a7ae0d95376d",
            "n_fallback_windows": 0,
            "n_grid_selected_windows": 0,
            "n_no_trade_windows": 3,
            "n_qualifying_windows": 0,
            "n_windows": 3,
        },
        "C-34 ledger key metrics agree with artifact",
    )
    ok(hold[0]["artifact_sha256"] == HASHES["holdout_2025"], "C-34 ledger artifact hash agrees")

    print(f"claim cross-check OK: {CHECKS} independent assertions passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"claim cross-check FAILED: {exc}")
        sys.exit(1)
