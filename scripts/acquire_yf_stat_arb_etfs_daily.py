#!/usr/bin/env python3
"""Acquire and freeze yf_stat_arb_etfs_daily_2015_2025_v1 for Directive #9 D9-B v3.

Authoritative 19-symbol, 4-group ETF universe (no cross-group pairs; see
configs/experiments/statarb_historical_etf_wf_v3.yaml). Raw CSVs under
data/raw/ (gitignored). Local/agent runs only -- never invoke from CI.

Per the authoritative spec: do not substitute symbols and do not shrink the
universe to work around thin coverage. If a symbol's actual historical
coverage is shorter than the requested window (e.g. a sector ETF that
launched after 2015-01-01), record it as a coverage note in the manifest --
the walk-forward engine's formation-window candidate selection is
responsible for excluding a symbol from windows that predate its inception,
not this acquisition step.
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

DATASET_ID = "yf_stat_arb_etfs_daily_2015_2025_v1"
REQUESTED_START = "2015-01-01"
REQUESTED_END_EXCLUSIVE = "2026-01-01"
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
ACTION_COLS = ["Dividends", "Stock Splits", "Capital Gains"]

# Authoritative D9-B v3 universe (spec: exact, no substitutions, no cross-group pairs).
GROUPS: dict[str, list[str]] = {
    "GROUP_A_BROAD_EQUITY": ["SPY", "QQQ", "DIA", "IWM"],
    "GROUP_B_US_EQUITY_SECTORS": [
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    ],
    "GROUP_C_TREASURY_ETFS": ["SHY", "IEF", "TLT"],
    "GROUP_D_PRECIOUS_METALS": ["GLD", "SLV"],
}


def documented_universe() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbols in GROUPS.values():
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
    return ordered


def group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, symbols in GROUPS.items():
        for symbol in symbols:
            mapping[symbol] = group
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


def validate_frame(symbol: str, df: pd.DataFrame, *, requested_start: str) -> dict[str, Any]:
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
    actual_start = df.index.min().strftime("%Y-%m-%d")
    coverage_note = None
    if actual_start > requested_start:
        coverage_note = (
            f"actual_start {actual_start} is after requested_start {requested_start} "
            "(likely post-inception ETF); symbol retained in universe per spec -- "
            "walk-forward candidate selection must exclude it from windows predating "
            "this date, not this acquisition step."
        )
    return {
        "row_count": len(df),
        "missing_ohlcv_cells": missing,
        "actual_start": actual_start,
        "actual_end": df.index.max().strftime("%Y-%m-%d"),
        "actions": action_info,
        "columns": list(df.columns),
        "coverage_note": coverage_note,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_dataset_hash(file_hashes: dict[str, str]) -> str:
    """Deterministic hash over sorted path->sha256 pairs (not raw bytes concat)."""
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
    grouping: dict[str, str],
) -> dict[str, Any]:
    coverage_notes = {
        s: per_symbol[s]["coverage_note"] for s in symbols if per_symbol[s]["coverage_note"]
    }
    return {
        "dataset_id": dataset_id,
        "source": "yfinance",
        "source_version": yf_version,
        "symbols": symbols,
        "universe_source": "Directive_9_D9B_authoritative_ETF_groups_A_D",
        "group_membership": {s: grouping[s] for s in symbols},
        "pair_constraint": "within_group_only",
        "interval": parameters["interval"],
        "requested_start": parameters["start"],
        "requested_end_exclusive": parameters["end"],
        "actual_start": {s: per_symbol[s]["actual_start"] for s in symbols},
        "actual_end": {s: per_symbol[s]["actual_end"] for s in symbols},
        "row_counts": {s: per_symbol[s]["row_count"] for s in symbols},
        "missing_counts": {s: per_symbol[s]["missing_ohlcv_cells"] for s in symbols},
        "actions": {s: per_symbol[s]["actions"] for s in symbols},
        "coverage_notes": coverage_notes,
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
    parser.add_argument(
        "--no-freeze", action="store_true", help="Acquire/validate only; leave sha256 null"
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance required. pip install yfinance", file=sys.stderr)
        return 2

    root = _repo_root()
    requested = list(args.symbols) if args.symbols else documented_universe()
    grouping = group_map()
    missing_group = [s for s in requested if s not in grouping]
    if missing_group:
        print(f"ERROR: symbols not in authoritative groups A-D: {missing_group}", file=sys.stderr)
        return 3

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
    }

    retrieval_ts = _utc_now()
    per_symbol: dict[str, dict[str, Any]] = {}
    file_paths: dict[str, Path] = {}
    failed: list[str] = []

    print(f"Acquiring {args.dataset_id} via yfinance {yf.__version__}")
    print(f"Requested [{args.start}, {args.end}) n_symbols={len(requested)} groups=A-D")

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
            stats = validate_frame(symbol, df, requested_start=args.start)
            out_path = raw_dir / f"{symbol}.csv"
            write_csv(out_path, df)
            file_paths[f"{symbol}.csv"] = out_path
            per_symbol[symbol] = stats
            note = f" NOTE: {stats['coverage_note']}" if stats["coverage_note"] else ""
            print(
                f"  {symbol} [{grouping[symbol]}]: rows={stats['row_count']} "
                f"actual={stats['actual_start']}..{stats['actual_end']} "
                f"missing_ohlcv_cells={stats['missing_ohlcv_cells']} -> {out_path}{note}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(symbol)
            print(f"  {symbol}: FAILED {exc}", file=sys.stderr)

    if failed:
        print(
            f"ERROR: {len(failed)} symbol(s) failed acquisition: {failed}. "
            "Per spec, do not substitute or drop symbols from the universe -- "
            "record this as a blocker and re-run once resolved.",
            file=sys.stderr,
        )
        return 1

    included = requested
    if args.no_freeze:
        status = "VALIDATED"
        freeze_ts = None
        sha256: dict[str, str] | None = None
        notes = (
            "Acquired and validated; NOT frozen. sha256 is null until --freeze "
            "(default path freezes)."
        )
    else:
        status = "DATA FROZEN"
        freeze_ts = _utc_now()
        file_hashes = {name: sha256_file(path) for name, path in sorted(file_paths.items())}
        sha256 = {
            **file_hashes,
            "dataset_canonical": canonical_dataset_hash(file_hashes),
        }
        notes = (
            "Frozen after local acquisition+validation. Raw CSVs remain gitignored "
            "under data/raw/. Re-acquire with the same parameters and compare "
            "dataset_canonical if regenerating. Authoritative D9-B v3 universe "
            "(Groups A-D, no cross-group pairs) -- supersedes the exploratory "
            "PairFinder.SECTOR_PAIRS dataset (yf_statarb_sector_equities_daily_2015_2025_v1, "
            "v1, non-conforming/superseded)."
        )

    manifest = build_manifest(
        dataset_id=args.dataset_id,
        symbols=included,
        yf_version=yf.__version__,
        retrieval_ts=retrieval_ts,
        freeze_ts=freeze_ts,
        status=status,
        per_symbol=per_symbol,
        parameters=parameters,
        sha256=sha256,
        notes=notes,
        grouping=grouping,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote manifest {manifest_path}")
    print(f"status={status}")
    if sha256 is not None:
        print(f"dataset_canonical sha256={sha256['dataset_canonical']}")
    else:
        print("sha256=null (not frozen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
