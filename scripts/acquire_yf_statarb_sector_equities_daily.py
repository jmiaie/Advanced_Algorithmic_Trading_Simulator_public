#!/usr/bin/env python3
"""Acquire and freeze yf_statarb_sector_equities_daily_2015_2025_v1 for the stat-arb study.

Universe: documented PairFinder.SECTOR_PAIRS. Raw under data/raw/ (gitignored).
Local/agent only — never CI. Coverage exclusions recorded at freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_ID = "yf_statarb_sector_equities_daily_2015_2025_v1"
REQUESTED_START = "2015-01-01"
REQUESTED_END_EXCLUSIVE = "2026-01-01"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
ACTION_COLS = ["Dividends", "Stock Splits", "Capital Gains"]

SECTOR_PAIRS: dict[str, list[str]] = {
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
    "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
    "Tech_Hardware": ["AAPL", "MSFT", "DELL", "HPQ", "IBM"],
    "Retail": ["WMT", "TGT", "COST", "DG", "DLTR"],
    "Airlines": ["DAL", "UAL", "LUV", "AAL", "ALK"],
    "Gold_Miners": ["NEM", "GOLD", "AEM", "FNV", "WPM"],
    "Oil_Services": ["SLB", "HAL", "BKR", "FTI"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "REITs": ["PLD", "AMT", "CCI", "EQIX", "SPG", "O"],
    "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
}


def documented_universe() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbols in SECTOR_PAIRS.values():
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
    return ordered


def sector_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sector, symbols in SECTOR_PAIRS.items():
        for symbol in symbols:
            mapping.setdefault(symbol, sector)
    return mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _flatten_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        level0 = out.columns.get_level_values(0)
        level1 = out.columns.get_level_values(1)
        if symbol in set(level1.astype(str)):
            out.columns = [
                str(a) if str(b) == symbol else f"{a}_{b}"
                for a, b in zip(level0, level1, strict=True)
            ]
        else:
            out.columns = [str(c[0]) for c in out.columns]
    out.columns = [str(c).strip() for c in out.columns]
    rename = {}
    for c in out.columns:
        cl = c.lower().replace(" ", "_")
        if cl == "adj_close":
            rename[c] = "Adj Close"
        elif cl == "stock_splits":
            rename[c] = "Stock Splits"
        elif cl == "capital_gains":
            rename[c] = "Capital Gains"
    if rename:
        out = out.rename(columns=rename)
    return out


def download_symbol(
    symbol: str,
    *,
    start: str,
    end: str,
    interval: str,
    auto_adjust: bool,
    actions: bool,
    repair: bool,
    keepna: bool,
) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        repair=repair,
        keepna=keepna,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = _flatten_columns(raw, symbol)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df.loc[(df.index >= start_ts) & (df.index < end_ts)]
    if df.empty:
        raise ValueError(f"Empty frame after date filter for {symbol}")
    missing_ohlcv = [c for c in OHLCV_COLS if c not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"{symbol} missing OHLCV columns: {missing_ohlcv}")
    return df


def validate_frame(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: index not monotonic increasing")
    if df.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate timestamps")
    ohlcv = df[OHLCV_COLS]
    missing = int(ohlcv.isna().sum().sum())
    action_info: dict[str, Any] = {}
    for col in ACTION_COLS:
        if col in df.columns:
            series = df[col].fillna(0)
            nonzero = int((series != 0).sum())
            action_info[col.lower().replace(" ", "_")] = {
                "present": True,
                "nonzero_count": nonzero,
            }
        else:
            action_info[col.lower().replace(" ", "_")] = {
                "present": False,
                "nonzero_count": None,
            }
    return {
        "row_count": len(df),
        "missing_ohlcv_cells": missing,
        "actual_start": df.index.min().strftime("%Y-%m-%d"),
        "actual_end": df.index.max().strftime("%Y-%m-%d"),
        "actions": action_info,
        "columns": list(df.columns),
    }


def coverage_ok(stats: dict[str, Any], *, min_start: str, min_end: str) -> bool:
    return (
        stats["actual_start"] <= min_start
        and stats["actual_end"] >= min_end
        and stats["missing_ohlcv_cells"] == 0
        and stats["row_count"] >= 2000
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_hash(file_hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{k}:{v}" for k, v in sorted(file_hashes.items())) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "Date"
    out.to_csv(path, date_format="%Y-%m-%d")


def build_manifest(
    *,
    dataset_id: str,
    symbols: list[str],
    yf_version: str,
    retrieval_ts: str,
    freeze_ts: str | None,
    status: str,
    per_symbol: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    sha256: dict[str, str] | None,
    notes: str,
    sector_grouping: dict[str, str],
    excluded_symbols: list[dict[str, str]],
    requested_universe: list[str],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source": "yfinance",
        "source_version": yf_version,
        "symbols": symbols,
        "requested_universe": requested_universe,
        "universe_source": "pair_finder.PairFinder.SECTOR_PAIRS",
        "sector_grouping": {s: sector_grouping[s] for s in symbols},
        "excluded_at_freeze": excluded_symbols,
        "interval": parameters["interval"],
        "requested_start": parameters["start"],
        "requested_end_exclusive": parameters["end"],
        "actual_start": {s: per_symbol[s]["actual_start"] for s in symbols},
        "actual_end": {s: per_symbol[s]["actual_end"] for s in symbols},
        "row_counts": {s: per_symbol[s]["row_count"] for s in symbols},
        "missing_counts": {s: per_symbol[s]["missing_ohlcv_cells"] for s in symbols},
        "actions": {s: per_symbol[s]["actions"] for s in symbols},
        "retrieval_timestamp_utc": retrieval_ts,
        "freeze_timestamp_utc": freeze_ts,
        "status": status,
        "sha256": sha256,
        "parameters": parameters,
        "per_symbol_columns": {s: per_symbol[s]["columns"] for s in symbols},
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default=REQUESTED_START)
    parser.add_argument("--end", default=REQUESTED_END_EXCLUSIVE, help="Exclusive end date")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--coverage-min-start", default="2015-01-10")
    parser.add_argument("--coverage-min-end", default="2025-12-15")
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required. pip install yfinance", file=sys.stderr)
        return 2

    root = _repo_root()
    requested = list(args.symbols) if args.symbols else documented_universe()
    sectors = sector_map()
    raw_dir = args.raw_dir or (root / "data" / "raw" / args.dataset_id)
    manifest_path = root / "data" / "manifests" / f"{args.dataset_id}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "auto_adjust": True,
        "actions": True,
        "repair": False,
        "keepna": True,
        "threads": False,
        "group_by": "column",
        "coverage_min_start": args.coverage_min_start,
        "coverage_min_end": args.coverage_min_end,
    }

    retrieval_ts = _utc_now()
    per_symbol: dict[str, dict[str, Any]] = {}
    file_paths: dict[str, Path] = {}
    failed: list[dict[str, str]] = []

    print(f"Acquiring {args.dataset_id} via yfinance {yf.__version__}")
    print(f"Requested [{args.start}, {args.end}) n_symbols={len(requested)}")

    for symbol in requested:
        try:
            df = download_symbol(
                symbol,
                start=args.start,
                end=args.end,
                interval=args.interval,
                auto_adjust=True,
                actions=True,
                repair=False,
                keepna=True,
            )
            stats = validate_frame(symbol, df)
            out_path = raw_dir / f"{symbol}.csv"
            write_csv(out_path, df)
            file_paths[f"{symbol}.csv"] = out_path
            per_symbol[symbol] = stats
            print(
                f"  {symbol}: rows={stats['row_count']} "
                f"actual={stats['actual_start']}..{stats['actual_end']} "
                f"missing_ohlcv_cells={stats['missing_ohlcv_cells']} -> {out_path}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"symbol": symbol, "reason": f"acquire_error:{exc}"})
            print(f"  {symbol}: FAILED {exc}")

    included: list[str] = []
    excluded: list[dict[str, str]] = list(failed)
    for symbol, stats in per_symbol.items():
        if coverage_ok(stats, min_start=args.coverage_min_start, min_end=args.coverage_min_end):
            included.append(symbol)
        else:
            excluded.append(
                {
                    "symbol": symbol,
                    "reason": (
                        f"coverage_gate start={stats['actual_start']} "
                        f"end={stats['actual_end']} missing={stats['missing_ohlcv_cells']} "
                        f"rows={stats['row_count']}"
                    ),
                }
            )
            file_paths.pop(f"{symbol}.csv", None)

    if not included:
        print("ERROR: no symbols passed coverage gates", file=sys.stderr)
        return 1

    included_paths = {f"{s}.csv": raw_dir / f"{s}.csv" for s in included}

    if args.no_freeze:
        status = "VALIDATED"
        freeze_ts = None
        sha256: dict[str, str] | None = None
        notes = "Acquired and validated; NOT frozen. sha256 null until freeze."
    else:
        status = "DATA FROZEN"
        freeze_ts = _utc_now()
        file_hashes = {name: sha256_file(path) for name, path in sorted(included_paths.items())}
        sha256 = {**file_hashes, "dataset_canonical": canonical_dataset_hash(file_hashes)}
        notes = (
            "Frozen after local acquisition+validation. Universe from "
            "pair_finder.PairFinder.SECTOR_PAIRS; coverage exclusions in excluded_at_freeze. "
            "Raw CSVs gitignored under data/raw/. Do not expand universe mid-study."
        )

    manifest = build_manifest(
        dataset_id=args.dataset_id,
        symbols=included,
        yf_version=yf.__version__,
        retrieval_ts=retrieval_ts,
        freeze_ts=freeze_ts,
        status=status,
        per_symbol={s: per_symbol[s] for s in included},
        parameters=parameters,
        sha256=sha256,
        notes=notes,
        sector_grouping=sectors,
        excluded_symbols=excluded,
        requested_universe=requested,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest {manifest_path}")
    print(f"status={status} included={len(included)} excluded={len(excluded)}")
    if sha256 is not None:
        print(f"dataset_canonical sha256={sha256['dataset_canonical']}")
    else:
        print("sha256=null (not frozen)")
    if excluded:
        print("Excluded:")
        for row in excluded:
            print(f"  {row['symbol']}: {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
