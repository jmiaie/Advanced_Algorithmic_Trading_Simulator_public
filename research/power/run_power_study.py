"""CLI: run Study A (A1 screen power + A2 positive control) and write outputs.

Full run (writes results/power/statarb_screen_power.json and .png):

    python -m research.power.run_power_study --full --workers 4

Smoke run (seconds; writes nothing unless --out is given):

    python -m research.power.run_power_study --smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

from .positive_control import grid_fire_probability, run_end_to_end
from .screen_power import (
    MASTER_SEED,
    PLANTED_PAIRS,
    REPO_ROOT,
    SPREAD_INNOVATION_SD,
    START_PRICE,
    Grid,
    aggregate,
    load_frozen_design,
    mde_table,
    run_replication,
)

FULL_GRID = Grid()
SMOKE_GRID = Grid(
    half_lives=(5, 120), ks=(1,), betas_primary=(1.0,), sigmas_sensitivity=(), reps=2
)
A2B_HALF_LIVES_FULL = (1, 2, 3, 5, 10, 20, 40)
A2B_REPS_FULL = 200
A2A_HALF_LIVES_FULL = (2, 20)
A2A_SEED = 0


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, check=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_a1(grid: Grid, workers: int) -> Dict[str, Any]:
    design = load_frozen_design()
    tasks = [(design, grid, s, rep) for s in grid.sigmas for rep in range(grid.reps)]
    records: List[Dict[str, Any]] = []
    if workers > 1:
        with Pool(workers) as pool:
            for batch in pool.imap_unordered(run_replication, tasks, chunksize=4):
                records.extend(batch)
    else:
        for t in tasks:
            records.extend(run_replication(t))
    cells = aggregate(records)
    return {
        "design": {
            "formation_size": design.formation_size,
            "fdr_alpha": design.fdr_alpha,
            "adf_alpha": design.adf_alpha,
            "n_candidate_tests": design.n_candidate_tests,
            "symbols": list(design.symbols),
            "group_membership": dict(design.group_membership),
        },
        "grid": {
            "half_lives": list(grid.half_lives),
            "ks": list(grid.ks),
            "betas_primary": list(grid.betas_primary),
            "sigma_primary": grid.sigma_primary,
            "sigmas_sensitivity": list(grid.sigmas_sensitivity),
            "beta_sensitivity": grid.beta_sensitivity,
            "reps_per_cell": grid.reps,
        },
        "cells": cells,
        "mde": mde_table(cells),
    }


def make_figure(a1: Dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    colors = {0.005: "#7a8fb8", 0.01: "#1f3b73", 0.02: "#c0504d"}
    styles = {1: "-", 3: "--"}
    for cell_sigma in sorted({c["sigma"] for c in a1["cells"]}):
        for k in (1, 3):
            pts = sorted(
                (c["half_life"], c["current"]["power"])
                for c in a1["cells"]
                if c["sigma"] == cell_sigma and c["k"] == k and c["beta"] == 1.0
            )
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                styles[k],
                marker="o",
                ms=3.5,
                lw=1.6,
                color=colors.get(cell_sigma, "black"),
                label=f"k={k}, leg vol {cell_sigma * 100:g}%",
            )
    ax.axhline(0.8, color="grey", lw=0.8, ls=":")
    ax.text(5.2, 0.815, "80% power", color="grey", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(a1["grid"]["half_lives"])
    ax.set_xticklabels([str(h) for h in a1["grid"]["half_lives"]])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("spread half-life (trading days)")
    ax.set_ylabel("P(planted pair is a survivor)")
    ax.set_title(
        "Frozen v4 screen (504-day formation, 55 tests, BH q=0.05 + ADF<0.05), beta=1",
        fontsize=9,
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-a2", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    grid = FULL_GRID if args.full else SMOKE_GRID
    a1 = run_a1(grid, args.workers)
    t_a1 = time.time() - t0

    a2: Dict[str, Any] = {}
    if not args.skip_a2:
        if args.full:
            a2["a2b_grid_fire_probability"] = grid_fire_probability(
                A2B_HALF_LIVES_FULL, A2B_REPS_FULL
            )
            a2["a2a_end_to_end"] = [
                run_end_to_end(half_life=h, seed=A2A_SEED) for h in A2A_HALF_LIVES_FULL
            ]
        else:
            a2["a2b_grid_fire_probability"] = grid_fire_probability((1,), 2)
    t_total = time.time() - t0

    payload = {
        "study": "statarb_screen_power",
        "protocol": "research/power/PROTOCOL.md",
        "mode": "full" if args.full else "smoke",
        "code_commit": _git_head(),
        "seeds": {
            "master_seed": MASTER_SEED,
            "a1_null_legs": "SeedSequence([master, round(sigma*1e4), rep, 0])",
            "a1_spread_innovations": "SeedSequence([master, round(sigma*1e4), rep, 1, slot])",
            "a1_reps": f"0..{grid.reps - 1} for every cell",
            "a2a_panel": f"SeedSequence([master, 2, {A2A_SEED}])",
            "a2b_pair": "SeedSequence([master, 3, rep])",
        },
        "dgp": {
            "legs": "independent geometric random walks, daily log-vol sigma, start price",
            "start_price": START_PRICE,
            "spread": "AR(1) s_t = phi s_{t-1} + eps_t, phi = 0.5**(1/h), stationary start",
            "spread_innovation_sd_price_units": SPREAD_INNOVATION_SD,
            "planted_leg": "y = beta * x + s (y is the production regressand)",
            "planted_pairs_in_order": [list(p) for p in PLANTED_PAIRS],
        },
        "a1": a1,
        "a2": a2,
        "runtime_seconds": {"a1": round(t_a1, 1), "total": round(t_total, 1)},
        "environment": {"python": platform.python_version(), "workers": args.workers},
    }
    out = args.out
    if out is None and args.full:
        out = REPO_ROOT / "results" / "power" / "statarb_screen_power.json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        make_figure(a1, out.with_suffix(".png"))
        print(f"wrote {out} and {out.with_suffix('.png')}")
    for row in a1["mde"]:
        if row["rule"] == "current":
            print(
                f"sigma={row['sigma']} k={row['k']} beta={row['beta']}: "
                f"MDE half-life={row['mde_half_life_days']} power={row['power_by_half_life']}"
            )
    print(f"runtime {t_total:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
