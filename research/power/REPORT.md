# Stat-arb screen power / null-calibration study — report

Frozen design: `research/power/PROTOCOL.md` (commit `4aa0b36`). Results: `results/power/statarb_screen_power.json` and `.png`.

## Summary

We re-ran the frozen v4 pair screen through its production code: 504-day formation window, 55 within-group Engle-Granger tests, Benjamini-Hochberg (BH) at q = 0.05, then ADF at p < 0.05. The input was synthetic windows with 1 or 3 truly cointegrated pairs planted among independent random walks, 500 replications per cell.

How often the screen found a planted pair, by spread half-life:

| Half-life | Detection rate |
|---|---|
| 5 days | 98% |
| 10 days | 46–61% |
| 20 days | 4–8% |
| 40, 60 or 120 days | 2% or less |

**Minimum detectable effect (MDE, 80% power): half-life of 5 trading days, the same in every σ / β / k cell.** The 2025 result (0 survivors in each of 3 windows) could have detected cointegration that reverts within about a week. It cannot tell "no cointegration" from "screen too strict" for half-lives of 20 days or more, the usual range for ETF pairs.

The 2026-10-14 BH correction moved power by at most 2.5 points, which is immaterial. The validation grid that fills `validation_ols_sharpe` does fire on a strong synthetic pair: it was populated in 27/27 development windows. The 2025 nulls in that field come from a code path that sets it to null by construction in holdout mode.

## Design freeze

- Commit 1 (protocol, simulation code, tests): `4aa0b36fd17c604f42ec6948170b86104f5965dc`, pushed **2026-10-20T06:52:54Z**, before the full run.
- Commit 2 (results): `7ccf05e`. There are no amendments to `research/power/PROTOCOL.md`.

## MDE headline

| Setting | MDE | Power at 10 days | Power at 20 days |
|---|---|---|---|
| k=1, β=1, σ=1% | **5 days** | 0.512 | 0.042 |
| k=3, β=1, σ=1% | **5 days** | 0.596 | 0.043 |

The MDE is 5 days in all 20 (σ, β, k) cells under both BH rules. The true 80% crossing lies between 5 and 10 days; the grid does not resolve it further.

- BH is the gate that binds, not ADF. Power with BH alone equals full-screen power. The planted pair passes ADF 85% of the time at 20 days but cannot clear the 55-test BH threshold (about p ≤ 0.0009 when there is one true pair).
- Under the global null (no planted pair), the chance of any survivor in a window is 0.064 at σ=1%, which is at nominal. It is 0.088 at σ=0.5% and 0.094 at σ=2%, which is mildly anti-conservative.
- Realised FDR in planted cells is 0.06–0.11. It reaches 0.33 at σ=0.5%, h=5. The likely cause is unit-root over-rejection on a leg that is a small random walk plus large stationary noise; this was not separately verified.
- The published survivor rate (0.23% of tests) is in line with the simulated global null (0.28–0.40%). It is equally in line with pairs at half-lives of 20 days or more being present, since the screen would almost never admit them.

## A2 finding

1. **The DEV grid-selection path can fire.** I ran the unmodified `scripts/run_statarb_wf_v4_study.py` with the frozen config on a synthetic 2015–2025 panel with a planted XLB/XLE pair at h=2. The planted pair was selected in every window, and `validation_ols_sharpe` was populated in 27/27 dev, 1/1 boundary and 3/3 val_2024 windows. Outputs went to a temporary directory; no frozen artifact or ledger was touched.
2. **The production grid search on its own:** P(grid selects) is 0.99 at h=1, 0.86 at h=5 and 0.57–0.67 at h=10–40 (200 reps each).
   - The 10-trade minimum (`wf_v4_orch.py:47`, applied at `:162`) is reachable but nearly binding: for h ≥ 10, the median best grid point produces exactly 10 trades.
   - So the published DEV nulls (all 4 traded windows used the fallback) are a property of the data, not an unreachable branch.
3. **In holdout mode, `validation_ols_sharpe` and `validation_kalman_sharpe` are null by construction.**
   - `scripts/run_statarb_wf_v4_study.py:266` passes `frozen_signal_params` under `--allow-holdout`.
   - `src/stat_arb_engine/wf_v4_orch.py:345-353` then sets both fields to `None` without ever calling the grid search.
   - In the synthetic run the planted pair was selected and traded in all 3 holdout windows, yet the field was null in 3/3.
   - So claim C-33 could not have come out any other way, and the Static-OLS vs Kalman comparison can never be evidenced from holdout windows. This is reported as a finding only; `src/` is not changed.

## Interpretation

The 2025 null rules out a within-group ETF pair with week-scale mean reversion in a given window. It does not rule out cointegration at month-to-half-year horizons; at those horizons "0 survivors" is the expected outcome either way. This holds per window: the three 2025 formation windows share 441 of 504 days, so they are not independent trials.

## Files

All files are new; nothing frozen is edited:
- `research/power/PROTOCOL.md`
- `research/power/screen_power.py`
- `research/power/positive_control.py`
- `research/power/run_power_study.py`
- `tests/test_power_study.py`
- `results/power/statarb_screen_power.json`
- `results/power/statarb_screen_power.png`

`research/power/REPORT.md` was **not** added. My tooling would not let me write that report file, so the full report text (findings, results tables, limitations) is in this description, and the file still has to be committed by a follow-up push.

The pre-fix BH rule is implemented locally, clearly labelled, in `screen_power.bh_reject_prefix_per_rank`.

## Reproduction

```
pip install -r requirements-lock.txt && pip install -e . --no-deps
python -m research.power.run_power_study --full --workers 4   # 4,707 s on 4 workers
```

## Verification (local, lock environment, Python 3.11)

- `python -m pytest -q`: 101 passed, 1 skipped. This includes the new tests: a smoke simulation under 30 s, a check that cached and uncached results are identical, the pre-fix rule, and a short-panel runner positive control.
- `ruff check .`: passed
- `mypy`: no issues in 34 files
- `publication_pack.py check`: OK
- `publication_pack.py hashcheck`: OK
- `claim_crosscheck.py`: 101 assertions passed

