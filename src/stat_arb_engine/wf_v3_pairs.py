"""D9-B v3 pair discovery: group-constrained candidate enumeration re-run
independently inside every walk-forward formation window (spec: "a
chronology-only list of windows is insufficient" -- v1/v2's
selection_rescanned_per_window=False is explicitly non-conforming).

Selection order (deterministic, no profitability-based tie-break):
  1. lowest BH-adjusted Engle-Granger p-value
  2. lower residual ADF p-value
  3. lexicographic symbol order
At most one pair per window. No qualifying pair => NO_TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from .research import benjamini_hochberg, fit_static_ols_hedge_ratio


@dataclass(frozen=True)
class PairSelectionResult:
    """Outcome of one formation-window candidate scan."""

    n_candidate_tests: int
    n_fdr_survivors: int
    selected: Mapping[str, object] | None  # None means NO_TRADE
    all_candidates: pd.DataFrame  # full diagnostic table, including rejected


def enumerate_group_candidates(
    symbols: Sequence[str], group_membership: Mapping[str, str]
) -> List[tuple[str, str]]:
    """All within-group symbol pairs present in `symbols`. No cross-group pairs."""
    by_group: Dict[str, List[str]] = {}
    for sym in symbols:
        group = group_membership.get(sym)
        if group is None:
            raise ValueError(f"Symbol {sym} has no group membership; refusing cross-group risk")
        by_group.setdefault(group, []).append(sym)
    pairs: List[tuple[str, str]] = []
    for group_symbols in by_group.values():
        pairs.extend(combinations(sorted(group_symbols), 2))
    return pairs


def select_pair_within_groups(
    formation_prices: pd.DataFrame,
    group_membership: Mapping[str, str],
    *,
    fdr_alpha: float = 0.05,
    adf_alpha: float = 0.05,
) -> PairSelectionResult:
    """Run Engle-Granger on every within-group candidate pair using formation
    data only, apply BH/FDR then residual-ADF, and deterministically select at
    most one pair. `formation_prices` must already be restricted to the
    formation window (no future information) and to symbols with full
    coverage over that window (caller drops symbols not yet listed/inception
    within the window -- see wf_v3_orch.available_symbols_for_window).
    """
    symbols = [s for s in formation_prices.columns if formation_prices[s].notna().all()]
    candidate_pairs = enumerate_group_candidates(symbols, group_membership)

    rows: List[Dict[str, object]] = []
    for sym_a, sym_b in candidate_pairs:
        try:
            fit = fit_static_ols_hedge_ratio(
                formation_prices[sym_a], formation_prices[sym_b]
            )
        except ValueError:
            continue
        rows.append(
            {
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "group": group_membership[sym_a],
                "engle_granger_pvalue": fit.engle_granger_pvalue,
                "adf_pvalue": fit.adf_pvalue,
                "hedge_ratio": fit.slope,
                "intercept": fit.intercept,
                "half_life": fit.half_life,
                "formation_sample_count": fit.formation_sample_count,
            }
        )

    all_candidates = pd.DataFrame(rows)
    n_tests = len(all_candidates)
    if all_candidates.empty:
        return PairSelectionResult(
            n_candidate_tests=0, n_fdr_survivors=0, selected=None, all_candidates=all_candidates
        )

    bh = benjamini_hochberg(all_candidates["engle_granger_pvalue"], alpha=fdr_alpha)
    all_candidates["bh_adjusted_pvalue"] = bh["adjusted_pvalue"].to_numpy()
    all_candidates["fdr_rejected"] = bh["rejected"].to_numpy()
    all_candidates["adf_passes"] = all_candidates["adf_pvalue"] < adf_alpha
    all_candidates["eligible"] = all_candidates["fdr_rejected"] & all_candidates["adf_passes"]

    eligible = all_candidates.loc[all_candidates["eligible"]].copy()
    n_survivors = int(eligible.shape[0])
    if eligible.empty:
        return PairSelectionResult(
            n_candidate_tests=n_tests,
            n_fdr_survivors=0,
            selected=None,
            all_candidates=all_candidates,
        )

    eligible["symbol_pair_key"] = eligible["symbol_a"] + "/" + eligible["symbol_b"]
    eligible = eligible.sort_values(
        ["bh_adjusted_pvalue", "adf_pvalue", "symbol_pair_key"],
        ascending=[True, True, True],
        kind="mergesort",  # stable, deterministic
    )
    top = eligible.iloc[0].to_dict()
    return PairSelectionResult(
        n_candidate_tests=n_tests,
        n_fdr_survivors=n_survivors,
        selected=top,
        all_candidates=all_candidates,
    )
