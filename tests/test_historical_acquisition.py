"""Offline tests for acquisition helpers (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "acquire_yf_statarb_sector_equities_daily.py"
)


def _load_acquire_module():
    spec = importlib.util.spec_from_file_location(
        "acquire_yf_statarb_sector_equities_daily", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def acq():
    return _load_acquire_module()


def _sample_frame(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": prices - 0.1,
            "High": prices + 0.2,
            "Low": prices - 0.2,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Capital Gains": 0.0,
        },
        index=dates,
    )


def test_documented_universe_unique(acq):
    universe = acq.documented_universe()
    assert len(universe) == len(set(universe))
    assert "XOM" in universe and "JPM" in universe


def test_validate_frame_counts(acq):
    df = _sample_frame()
    stats = acq.validate_frame("XOM", df)
    assert stats["row_count"] == 10
    assert stats["missing_ohlcv_cells"] == 0
    assert stats["actual_start"] == "2015-01-02"


def test_validate_frame_rejects_duplicates(acq):
    df = _sample_frame()
    df.index = list(df.index[:-1]) + [df.index[-2]]
    with pytest.raises(ValueError, match="duplicate"):
        acq.validate_frame("XOM", df)


def test_canonical_hash_is_order_independent(acq):
    a = {"JPM.csv": "abc", "XOM.csv": "def"}
    b = {"XOM.csv": "def", "JPM.csv": "abc"}
    assert acq.canonical_dataset_hash(a) == acq.canonical_dataset_hash(b)


def test_coverage_gate(acq):
    ok = {
        "actual_start": "2015-01-02",
        "actual_end": "2025-12-31",
        "missing_ohlcv_cells": 0,
        "row_count": 2700,
    }
    bad = dict(ok, actual_start="2018-12-01", row_count=1500)
    assert acq.coverage_ok(ok, min_start="2015-01-10", min_end="2025-12-15")
    assert not acq.coverage_ok(bad, min_start="2015-01-10", min_end="2025-12-15")
