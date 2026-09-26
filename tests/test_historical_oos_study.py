"""Offline tests for historical OOS helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb_engine.historical_oos import (
    PeriodSpec,
    select_pairs_within_sectors,
    summarize_selection,
)


def _synthetic_panel(n: int = 400) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    # Two cointegrated-ish energy names + one independent bank name.
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    y = 2.0 + 1.2 * x + rng.normal(0, 0.2, n)
    z = 100 + np.cumsum(rng.normal(0, 1.0, n))
    panel = pd.DataFrame({"AAA": y, "BBB": x, "CCC": z}, index=idx)
    sectors = {"AAA": "Energy", "BBB": "Energy", "CCC": "Banks"}
    return panel, sectors


def test_within_sector_selection_excludes_cross_sector():
    panel, sectors = _synthetic_panel()
    selection = select_pairs_within_sectors(panel, sectors, fdr_alpha=0.5, adf_alpha=0.5)
    assert not selection.empty
    pairs = set(zip(selection["symbol_a"], selection["symbol_b"], strict=True))
    assert ("AAA", "CCC") not in pairs and ("CCC", "AAA") not in pairs
    assert ("AAA", "BBB") in pairs or ("BBB", "AAA") in pairs


def test_holdout_blocked_without_freeze():
    from stat_arb_engine.historical_oos import run_period_study

    panel, sectors = _synthetic_panel()
    formation = PeriodSpec("formation_dev", "2015-01-01", "2016-12-31")
    holdout = PeriodSpec("holdout", "2017-01-01", "2017-06-30")
    config = {
        "status": "not-yet-frozen-for-holdout",
        "selection": {"fdr_alpha": 0.5, "adf_alpha": 0.5, "top_n_pairs": 1},
        "signals": {"entry_z": 2.0, "exit_z": 0.5, "z_window": 20},
        "kalman": {"process_variance": 1e-4, "observation_variance": 1e-2},
        "costs": {
            "commission_per_share": 0.005,
            "min_commission": 1.0,
            "spread_bps": 0.0,
            "slippage_bps": 0.0,
            "impact_bps": 0.0,
            "scenario_label": "test",
        },
        "execution": {
            "starting_cash": 100000.0,
            "sizing_mode": "fixed_shares",
            "fixed_shares": 10,
            "gross_notional": 10000.0,
        },
        "walk_forward": {"formation_size": 100, "validation_size": 20, "test_size": 20},
    }
    with pytest.raises(RuntimeError, match="Holdout evaluation blocked"):
        run_period_study(
            full_panel=panel,
            formation=formation,
            eval_period=holdout,
            sector_grouping=sectors,
            config=config,
            allow_holdout=True,
        )


def test_summarize_selection_null_safe():
    summary = summarize_selection(pd.DataFrame(), top_n=5)
    assert summary["n_tests"] == 0
    assert summary["top_pairs"] == []
