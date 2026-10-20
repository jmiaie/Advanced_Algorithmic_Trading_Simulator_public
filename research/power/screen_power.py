"""Study A1: power / null calibration of the frozen v4 pair screen.

Simulates formation windows that match the frozen v4 design (504 sessions, the
19-symbol / 4-group universe -> 55 within-group Engle-Granger tests, BH at
fdr_alpha, residual-ADF admission at adf_alpha) and plants k truly cointegrated
pairs among otherwise independent geometric random walks.

The screen itself is the PRODUCTION function
``stat_arb_engine.wf_v3_pairs.select_pair_within_groups`` (which calls the
production ``fit_static_ols_hedge_ratio`` and ``benjamini_hochberg``). Nothing
about Engle-Granger, BH or ADF is re-implemented here. The only local statistic
is ``bh_reject_prefix_per_rank``: the PRE-FIX (pre-2026-10-14) per-rank BH rule,
kept here solely as a labelled sensitivity analysis.

Speed: within one replication the 19 null legs are shared by every grid cell
(common random numbers), so most of the 55 pair fits recur with byte-identical
inputs. ``memoized_production_fit`` wraps the production fit with an exact-input
cache (keyed on the raw bytes of both series and their index); every value it
returns was computed by the production function on exactly those inputs.

Run ``python research/power/run_power_study.py --help`` for the CLI.
"""

from __future__ import annotations

import contextlib
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import stat_arb_engine.wf_v3_pairs as wf_v3_pairs
from stat_arb_engine.research import StaticOLSHedgeResult, fit_static_ols_hedge_ratio
from stat_arb_engine.wf_v3_pairs import enumerate_group_candidates, select_pair_within_groups

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG = REPO_ROOT / "configs" / "experiments" / "statarb_historical_etf_wf_v4.yaml"

MASTER_SEED = 20261020
START_PRICE = 100.0
SPREAD_INNOVATION_SD = 1.0  # price units = 1% of START_PRICE; fixed in every cell

# Planted pairs: disjoint symbols (so no extra, transitive cointegrated pairs are
# created). The first symbol of each tuple is the production regressand
# (enumerate_group_candidates sorts symbols; fit_static_ols_hedge_ratio(y=sym_a,
# x=sym_b)), and it is the leg built as y = beta * x + s.
PLANTED_PAIRS: Tuple[Tuple[str, str], ...] = (("XLB", "XLE"), ("QQQ", "SPY"), ("IEF", "TLT"))


@dataclass(frozen=True)
class FrozenDesign:
    """Design parameters read (read-only) from the frozen v4 config."""

    group_membership: Mapping[str, str]
    symbols: Tuple[str, ...]
    formation_size: int
    fdr_alpha: float
    adf_alpha: float
    n_candidate_tests: int


def load_frozen_design(config_path: Path = FROZEN_CONFIG) -> FrozenDesign:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = cfg["universe"]["groups"]
    membership = {sym: grp for grp, syms in groups.items() for sym in syms}
    symbols = tuple(sorted(membership))
    n_tests = len(enumerate_group_candidates(symbols, membership))
    return FrozenDesign(
        group_membership=membership,
        symbols=symbols,
        formation_size=int(cfg["walk_forward"]["formation_size"]),
        fdr_alpha=float(cfg["selection"]["fdr_alpha"]),
        adf_alpha=float(cfg["selection"]["adf_alpha"]),
        n_candidate_tests=n_tests,
    )


# --------------------------------------------------------------------------
# PRE-FIX BH RULE -- sensitivity analysis only. NOT the production rule.
# --------------------------------------------------------------------------
def bh_reject_prefix_per_rank(p_values: Sequence[float], alpha: float) -> np.ndarray:
    """PRE-FIX (superseded 2026-10-14) Benjamini-Hochberg rejection rule, as the
    repository applied it before commit a2a495e: each sorted p-value is compared
    to its OWN rank threshold, ``p_(k) <= alpha * k / m``, instead of rejecting
    every rank up to the largest passing k. Returned in input order. Used only
    to show whether the correction changed the screen's power."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1)
    return np.asarray(p <= alpha * ranks / m)


# --------------------------------------------------------------------------
# Exact-input memoization of the production fit.
# --------------------------------------------------------------------------
_FIT_CACHE: Dict[bytes, StaticOLSHedgeResult] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _series_key(y: Any, x: Any) -> bytes:
    h = hashlib.blake2b(digest_size=20)
    for s in (y, x):
        series = pd.Series(s)
        h.update(series.to_numpy(dtype=float).tobytes())
        h.update(np.asarray(series.index.asi8 if hasattr(series.index, "asi8") else []).tobytes())
    return h.digest()


def _cached_fit(y: Any, x: Any) -> StaticOLSHedgeResult:
    key = _series_key(y, x)
    hit = _FIT_CACHE.get(key)
    if hit is not None:
        _CACHE_STATS["hits"] += 1
        return hit
    _CACHE_STATS["misses"] += 1
    result = fit_static_ols_hedge_ratio(y, x)
    _FIT_CACHE[key] = result
    return result


@contextlib.contextmanager
def memoized_production_fit() -> Iterator[None]:
    """Within the block, select_pair_within_groups calls the production fit
    through an exact-input cache. The cache is emptied on exit."""
    original = wf_v3_pairs.fit_static_ols_hedge_ratio
    wf_v3_pairs.fit_static_ols_hedge_ratio = _cached_fit
    try:
        yield
    finally:
        wf_v3_pairs.fit_static_ols_hedge_ratio = original
        _FIT_CACHE.clear()


# --------------------------------------------------------------------------
# Data generation.
# --------------------------------------------------------------------------
def sigma_code(sigma: float) -> int:
    return int(round(sigma * 10_000))


def null_legs(design: FrozenDesign, sigma: float, rep: int) -> pd.DataFrame:
    """Independent geometric random walks, daily log-vol sigma, start 100."""
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, sigma_code(sigma), rep, 0]))
    n, m = design.formation_size, len(design.symbols)
    log_paths = np.cumsum(rng.normal(0.0, sigma, size=(n, m)), axis=0)
    index = pd.bdate_range("2001-01-01", periods=n)
    return pd.DataFrame(START_PRICE * np.exp(log_paths), index=index, columns=list(design.symbols))


def ar1_spread(innovations: np.ndarray, half_life: float) -> np.ndarray:
    """s_t = phi s_{t-1} + eps_t, phi = 0.5**(1/h), stationary start."""
    phi = 0.5 ** (1.0 / half_life)
    s = np.empty_like(innovations)
    s[0] = innovations[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, innovations.size):
        s[t] = phi * s[t - 1] + innovations[t]
    return s


def spread_innovations(sigma: float, rep: int, pair_slot: int, n: int) -> np.ndarray:
    """Standard normals scaled by SPREAD_INNOVATION_SD. Common random numbers:
    identical across half-life, beta and k for a given (sigma, rep, slot)."""
    rng = np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, sigma_code(sigma), rep, 1, pair_slot])
    )
    return SPREAD_INNOVATION_SD * rng.normal(0.0, 1.0, size=n)


def plant_pairs(
    legs: pd.DataFrame, sigma: float, rep: int, k: int, half_life: float, beta: float
) -> pd.DataFrame:
    panel = legs.copy()
    for slot, (sym_y, sym_x) in enumerate(PLANTED_PAIRS[:k]):
        eps = spread_innovations(sigma, rep, slot, len(panel))
        panel[sym_y] = beta * panel[sym_x].to_numpy() + ar1_spread(eps, half_life)
    return panel


# --------------------------------------------------------------------------
# One screen evaluation -> counts.
# --------------------------------------------------------------------------
def screen_counts(
    panel: pd.DataFrame, design: FrozenDesign, planted: Sequence[Tuple[str, str]]
) -> Dict[str, Any]:
    """Run the production screen and summarise survivors under the current
    (production) rule and the labelled pre-fix rule."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = select_pair_within_groups(
            panel, design.group_membership, fdr_alpha=design.fdr_alpha, adf_alpha=design.adf_alpha
        )
    cand = sel.all_candidates
    keys = list(zip(cand["symbol_a"], cand["symbol_b"], strict=True))
    is_planted = np.array([k in set(planted) for k in keys], dtype=bool)
    eligible_new = cand["eligible"].to_numpy(dtype=bool)
    adf_pass = cand["adf_passes"].to_numpy(dtype=bool)
    fdr_old = bh_reject_prefix_per_rank(cand["engle_granger_pvalue"].to_numpy(), design.fdr_alpha)
    eligible_old = fdr_old & adf_pass
    assert int(eligible_new.sum()) == sel.n_fdr_survivors

    selected_key = None
    if sel.selected is not None:
        selected_key = (str(sel.selected["symbol_a"]), str(sel.selected["symbol_b"]))

    def rule(elig: np.ndarray) -> Dict[str, Any]:
        tp = int((elig & is_planted).sum())
        fp = int((elig & ~is_planted).sum())
        r = tp + fp
        return {
            "true_survivors": tp,
            "false_survivors": fp,
            "fdp": (fp / r) if r else 0.0,
            "any_survivor": r > 0,
        }

    return {
        "n_tests": int(sel.n_candidate_tests),
        "current": rule(eligible_new),
        "prefix": rule(eligible_old),
        "bh_only_true": int((cand["fdr_rejected"].to_numpy(dtype=bool) & is_planted).sum()),
        "adf_only_true": int((adf_pass & is_planted).sum()),
        "selected_is_planted": selected_key is not None and selected_key in set(planted),
    }


# --------------------------------------------------------------------------
# Grid and per-replication task.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    half_lives: Tuple[float, ...] = (5, 10, 20, 40, 60, 120)
    ks: Tuple[int, ...] = (1, 3)
    betas_primary: Tuple[float, ...] = (0.5, 1.0, 2.0)
    sigma_primary: float = 0.01
    sigmas_sensitivity: Tuple[float, ...] = (0.005, 0.02)
    beta_sensitivity: float = 1.0
    reps: int = 500

    def cells_for_sigma(self, sigma: float) -> List[Tuple[int, float, float]]:
        betas = self.betas_primary if sigma == self.sigma_primary else (self.beta_sensitivity,)
        return [(k, h, b) for k in self.ks for h in self.half_lives for b in betas]

    @property
    def sigmas(self) -> Tuple[float, ...]:
        return (self.sigma_primary, *self.sigmas_sensitivity)


def run_replication(args: Tuple[FrozenDesign, Grid, float, int]) -> List[Dict[str, Any]]:
    """All grid cells for one (sigma, rep), plus the k=0 global-null screen."""
    design, grid, sigma, rep = args
    out: List[Dict[str, Any]] = []
    with memoized_production_fit():
        legs = null_legs(design, sigma, rep)
        null = screen_counts(legs, design, planted=())
        out.append({"sigma": sigma, "k": 0, "half_life": None, "beta": None, "rep": rep, **null})
        for k, h, b in grid.cells_for_sigma(sigma):
            panel = plant_pairs(legs, sigma, rep, k, h, b)
            res = screen_counts(panel, design, planted=PLANTED_PAIRS[:k])
            out.append({"sigma": sigma, "k": k, "half_life": h, "beta": b, "rep": rep, **res})
    return out


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def aggregate(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse per-replication records into one row per cell."""
    cells: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for r in records:
        cells.setdefault((r["sigma"], r["k"], r["half_life"], r["beta"]), []).append(r)
    rows: List[Dict[str, Any]] = []
    for (sigma, k, h, b), recs in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3] or 0)
    ):
        n = len(recs)
        row: Dict[str, Any] = {
            "sigma": sigma,
            "k": k,
            "half_life": h,
            "beta": b,
            "n_reps": n,
            "reps": sorted(int(r["rep"]) for r in recs),
            "n_tests": sorted({int(r["n_tests"]) for r in recs}),
        }
        for rule in ("current", "prefix"):
            tp = sum(int(r[rule]["true_survivors"]) for r in recs)
            fp = sum(int(r[rule]["false_survivors"]) for r in recs)
            trials = n * k
            stats: Dict[str, Any] = {
                "planted_survivor_count": tp,
                "planted_pair_trials": trials,
                "false_survivor_total": fp,
                "expected_false_survivors": fp / n,
                "realised_fdr": sum(float(r[rule]["fdp"]) for r in recs) / n,
                "p_any_survivor": sum(bool(r[rule]["any_survivor"]) for r in recs) / n,
            }
            if k > 0:
                stats["power"] = tp / trials
                stats["power_wilson95"] = list(wilson_interval(tp, trials))
                stats["p_all_planted_survive"] = (
                    sum(int(r[rule]["true_survivors"]) == k for r in recs) / n
                )
                stats["p_at_least_one_planted_survives"] = (
                    sum(int(r[rule]["true_survivors"]) > 0 for r in recs) / n
                )
            row[rule] = stats
        if k > 0:
            row["power_bh_only_current"] = sum(int(r["bh_only_true"]) for r in recs) / (n * k)
            row["power_adf_only"] = sum(int(r["adf_only_true"]) for r in recs) / (n * k)
            row["p_selected_is_planted"] = sum(bool(r["selected_is_planted"]) for r in recs) / n
        rows.append(row)
    return rows


def mde_table(rows: Sequence[Mapping[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
    """MDE = largest grid half-life with power >= threshold (None if none)."""
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for r in rows:
        if r["k"] == 0:
            continue
        groups.setdefault((r["sigma"], r["k"], r["beta"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for (sigma, k, beta), rs in sorted(groups.items()):
        for rule in ("current", "prefix"):
            ok = [float(r["half_life"]) for r in rs if r[rule]["power"] >= threshold]
            out.append(
                {
                    "sigma": sigma,
                    "k": k,
                    "beta": beta,
                    "rule": rule,
                    "mde_half_life_days": max(ok) if ok else None,
                    "power_by_half_life": {
                        str(int(r["half_life"])): r[rule]["power"]
                        for r in sorted(rs, key=lambda x: x["half_life"])
                    },
                }
            )
    return out
