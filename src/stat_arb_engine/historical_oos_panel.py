"""Panel load + within-sector pair selection for D9 historical OOS."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from .research import benjamini_hochberg, fit_static_ols_hedge_ratio

@dataclass(frozen=True)
class PeriodSpec:
    name: str
    start: str
    end_inclusive: str

def load_close_panel(raw_dir: Path, symbols: Sequence[str]) -> pd.DataFrame:
    frames: Dict[str, pd.Series] = {}
    for symbol in symbols:
        path = raw_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen CSV for {symbol}: {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            raise ValueError(f"{symbol} missing Close column")
        series = df["Close"].astype(float)
        if series.index.has_duplicates:
            raise ValueError(f"{symbol} has duplicate dates")
        frames[symbol] = series
    panel = pd.DataFrame(frames).sort_index()
    if not panel.index.is_monotonic_increasing:
        raise ValueError("Panel index not monotonic increasing")
    return panel

def slice_period(panel: pd.DataFrame, period: PeriodSpec) -> pd.DataFrame:
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end_inclusive)
    out = panel.loc[(panel.index >= start) & (panel.index <= end)].copy()
    return out

def select_pairs_within_sectors(
    price_frame: pd.DataFrame,
    sector_grouping: Mapping[str, str],
    *,
    fdr_alpha: float = 0.05,
    adf_alpha: float = 0.05,
) -> pd.DataFrame:
    """Engle-Granger scan restricted to same-sector pairs; BH/FDR pooled across tests."""
    frame = price_frame.dropna(axis=0, how="any")
    results: List[Dict[str, Any]] = []
    sectors: Dict[str, List[str]] = {}
    for symbol in frame.columns:
        sector = sector_grouping.get(str(symbol))
        if sector is None:
            continue
        sectors.setdefault(sector, []).append(str(symbol))

    for sector, symbols in sectors.items():
        if len(symbols) < 2:
            continue
        for sym_a, sym_b in combinations(sorted(symbols), 2):
            try:
                pair_result = fit_static_ols_hedge_ratio(frame[sym_a], frame[sym_b])
            except ValueError:
                continue
            results.append(
                {
                    "symbol_a": sym_a,
                    "symbol_b": sym_b,
                    "sector": sector,
                    "engle_granger_pvalue": pair_result.engle_granger_pvalue,
                    "adf_pvalue": pair_result.adf_pvalue,
                    "hedge_ratio": pair_result.slope,
                    "intercept": pair_result.intercept,
                    "half_life": pair_result.half_life,
                    "formation_sample_count": pair_result.formation_sample_count,
                    "estimation_mode": pair_result.estimation_mode,
                    "coverage_start": frame.index.min(),
                    "coverage_end": frame.index.max(),
                }
            )

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        return result_frame
    bh = benjamini_hochberg(result_frame["engle_granger_pvalue"], alpha=fdr_alpha)
    result_frame["raw_pvalue"] = bh["raw_pvalue"].values
    result_frame["adjusted_pvalue"] = bh["adjusted_pvalue"].values
    result_frame["qvalue"] = bh["qvalue"].values
    result_frame["rejected"] = bh["rejected"].values & (result_frame["adf_pvalue"] < adf_alpha)
    result_frame["test_count"] = bh["test_count"].values
    result_frame["hypotheses_count"] = bh["hypotheses_count"].values
    result_frame["selection_logic"] = (
        "Within-sector pairs only; BH on Engle-Granger p-values; "
        "residual ADF as secondary diagnostic filter"
    )
    return result_frame.sort_values(
        ["rejected", "qvalue", "engle_granger_pvalue"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

def _pair_ohlc_dict(panel: pd.DataFrame, sym_a: str, sym_b: str) -> Dict[str, pd.DataFrame]:
    aligned = panel[[sym_a, sym_b]].dropna(how="any")
    return {
        sym_a: pd.DataFrame({"Close": aligned[sym_a]}),
        sym_b: pd.DataFrame({"Close": aligned[sym_b]}),
    }
