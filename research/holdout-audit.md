# Holdout audit — Stat-Arb Directive #9 (2025 calendar year)

**Repository:** `jmiaie/Advanced_Algorithmic_Trading_Simulator_public`  
**Branch:** `research/historical-oos-study`  
**Audit date (PT):** 2026-09-15  
**Holdout window under D9:** calendar **2025-01-01 ≤ t < 2026-01-01**  
**Dataset ID (planned):** `yf_statarb_sector_equities_daily_2015_2025_v1`

## Verdict

**CLEAR** — no evidence that calendar-year **2025** market data was previously inspected, tuned against, or used for empirical evaluation / performance claims in this repository.

## Scope searched

| Surface | Method | Finding |
|---|---|---|
| Working tree (src, scripts, tests, docs, research, results, README) | recursive text search for `2025` / `holdout` | Only this audit scaffold / D9 placeholders; no empirical 2025 metrics. |
| Results tree | listed `results/*` | Placeholder READMEs under `synthetic/`, `walk_forward/`, `pair_selection/`, `execution_costs/`, `sensitivity/`, `factor_attribution/`. No CSV/JSON OOS packs. |
| Research doc | `research/statistical-arbitrage-validation.md` | States “Results pending reproducible OOS run.” No 2025 evaluation. |
| PairFinder / live bridges | `pair_finder.py`, `live_feed.py`, `live_trading.py` | Convenience live/Alpaca paths; no pinned 2015–2025 study window; no committed 2025 OOS metrics. |
| Tests | `tests/` | Synthetic/unit fixtures for D3 gates and research helpers only. |
| GitHub code search (prior session notes) | empirical 2025 evaluation artifacts | **null** |

## Classification rules applied

- **CLEAR:** holdout calendar window not used for model selection, hyperparameter tuning, benchmark cherry-picking, or reported historical performance.
- **PREVIOUSLY INSPECTED:** any committed notebook/result/config that evaluates or plots 2025 pair returns for research decisions.

Copyright / changelog year strings do **not** count as empirical holdout inspection.

## Universe choice (explicit; not expanded mid-study)

Documented convenience universe from `pair_finder.PairFinder.SECTOR_PAIRS` (liquid US sector equities). Survivorship / convenience bias remains an explicit limitation (README + research doc). Frozen symbol list is locked in the dataset manifest after acquisition; mid-study expansion is forbidden.

## Restrictions until FINAL CONFIGURATION FROZEN

1. Do **not** evaluate models on 2025 for final claims.
2. Development / validation work uses **2015–2023** (formation/dev) and **2024** (validation) only, once data is frozen.
3. Pre-registered experiment configs under `configs/experiments/` remain **`not-yet-frozen-for-holdout`** until development+validation complete and a tracker `FINAL CONFIGURATION FROZEN` record exists on Issue #3.

## Sign-off

| Field | Value |
|---|---|
| Holdout status | **CLEAR** |
| Prior empirical 2025 evaluation artifacts | **null** (none found) |
| Ready for acquisition + pre-registration | **yes** |


## Post-freeze holdout execution (v1, superseded universe)

YAML status was set to `frozen-for-holdout` after formation+validation; calendar-2025 evaluation was then run once under frozen config `statarb_hist_oos_v1` (experiment `statarb_hist_oos_v1_holdout_2025`). Tracker confirmation: **FINAL CONFIGURATION FROZEN — Stat-Arb D9-B** on Issue #3 (https://github.com/jmiaie/quant-research-portfolio/issues/3#issuecomment-5691711387). Formation-time BH/FDR still selected **0** pairs, so holdout trading metrics are **null** (reported honestly). No retune after freeze.

**This section covers only v1's `PairFinder.SECTOR_PAIRS` universe** (individual equities: energy/bank/tech/retail/airline/miner/utility/REIT/pharma names). It does **not** cover the v3 authoritative ETF universe below.

## D9-B v3 holdout audit (authoritative ETF universe, dataset `yf_stat_arb_etfs_daily_2015_2025_v1`)

**Audit date (PT):** 2026-09-16

### Prior Stat-Arb versions, explicitly

- **v1** (`PairFinder.SECTOR_PAIRS`, individual equities: energy/bank/tech/retail/airline/miner/utility/REIT/pharma names): **did execute and commit a 2025 holdout** (`statarb_hist_oos_v1_holdout_2025`, null trades — formation-time FDR selected 0 pairs, so no 2025 prices actually informed a trading decision, but the 2025 rows were loaded and the holdout ran). Zero symbol overlap with v3's 19-symbol ETF universe, so this cannot leak into v3 through any shared data or code path.
- **v2** (`statarb_hist_etf_wf_v2`, invalidated-before-holdout): **never touched any 2025 data for any symbol.** No dataset manifest, no raw CSVs, and no pipeline code ever existed for it before this session invalidated it — there was nothing to acquire or inspect. Confirmed null exposure, not just "no execution recorded."
- **v3** (this study): no code path in `wf_v3_*` reads, references, or was written with knowledge of 2025 values. DEV/validation execution used a panel with all 2025-01-01+ rows structurally sliced out before the study ran (`scripts/run_statarb_wf_v3_study.py`), verified via an explicit refusal test, not just a status check.

### Cross-repository exposure (disclosed, not resolved)

**Mechanical execution:** 5 of the 19 v3 symbols — **SPY, QQQ, IWM, TLT, GLD** — are also the entire universe of `financial-dynamics-model`'s D9-A regime study, whose 2025 holdout (`fdm_hist_regime_v1_holdout_2025`) has **already been executed and its results discussed/reported** (e.g. SPY 2025 `mean_return_h1 = -0.000293`; FDM PR #12, Issue #3). No code in this repository's pipeline consumes FDM's results — pair and parameter selection here are fully mechanical (Engle-Granger + BH-FDR + a pre-specified grid with a pre-specified fallback) — so the *executing code* has no direct leakage path.

**Design/specification origin — genuinely unresolved, not just mechanically clear.** The v3 universe, walk-forward window sizes, validation grid values, fallback parameters, and cost assumptions were handed to this session as an already-written "authoritative spec," not derived here. Per the program tracker (Issue #3), FDM's 2025 SPY/QQQ/IWM/TLT/GLD holdout results were already posted **before** that authoritative D9-B spec was delivered. Whether the people or systems who authored the spec's specific design choices (this exact universe, this exact grid, this exact fallback) were influenced by having already seen those results is a real question this document cannot answer — it's outside what any code-level audit of this repository can establish. Mechanical selection at execution time does not rule out the design itself having been shaped by previously seen outcomes upstream.

**Verdict for v3:** CLEAR that the *executing pipeline* never reads 2025 data and has no code-level leakage path. **Not resolved:** whether the *design* it executes was influenced by prior knowledge of FDM's 2025 results for the 5 overlapping symbols. Both statements are disclosed here for whoever authorizes the D9-B v3 holdout to weigh — neither is asserted as a blanket "clear."

## D9-B v3 invalidated before holdout; v4 supersedes it (still pre-holdout)

**Update date (PT):** 2026-09-16

v3 was never evaluated against 2025 data (see above) and **remains** never evaluated against 2025 data — it was invalidated for unrelated reasons (execution-timing and tie-break defects; see `research/statistical-arbitrage-validation.md` and the config file's header) before any holdout freeze occurred. v4 (`statarb_hist_etf_wf_v4`) supersedes it, carrying forward the identical universe, dataset, and 2025-exclusion mechanics:

- No code path in `wf_v4_*` reads, references, or was written with knowledge of 2025 values — `wf_v4_orch.py` reuses v3's unmodified pair-selection/sizing/cost modules and only changes execution-timing and tie-break logic, neither of which touches calendar-based filtering.
- DEV/2024-validation execution used the same structural 2025-exclusion as v3 (`scripts/run_statarb_wf_v4_study.py`, panel sliced to `< 2025-01-01` before the study runs, not just a status check), verified by `tests/test_wf_v4_holdout_gate.py::test_v4_runner_dev_val_run_never_scores_2025_rows` and a refusal-gate test analogous to v3's.
- The cross-repository exposure disclosure above (5/19 symbols also in FDM's D9-A universe, whose 2025 holdout already executed; design-origin question genuinely unresolved) applies identically to v4, since the universe is unchanged from v3.

**Verdict for v4:** Same as v3's verdict above, carried forward unchanged — CLEAR on code-level execution, NOT RESOLVED on design-origin chronology. No 2025 evaluation has occurred for v4.

## v4 2025 label, pre-committed before any 2025 numbers exist

**Date (PT):** 2026-09-17. Written now, while `configs/experiments/statarb_historical_etf_wf_v4.yaml` status is still `pre-registered` and no `--allow-holdout` run has ever produced a v4 2025 artifact — so this label cannot have been chosen to fit an already-seen result.

Per Directive #9 Addendum 12: v4's code-level holdout controls are clean (see verdict immediately above), but cross-program research-design exposure remains genuinely unresolved because 5 of the 19 ETFs (SPY, QQQ, IWM, TLT, GLD) overlap the already-executed D9-A FDM universe. Addendum 12 therefore forbids labeling the eventual v4 2025 run `UNTOUCHED 2025 HOLDOUT` until final independent review resolves that design-origin chronology.

**Binding label for the eventual v4 2025 run:** `FINAL 2025 WALK-FORWARD EVALUATION` — explicitly **not** an untouched holdout — reported together with the unresolved cross-program exposure disclosure above every time a v4 2025 number is cited (report text, ledger notes, tracker updates, and any D10 evidence package entry). This applies from the first v4 2025 run onward; it is not a label to be revisited after seeing the numbers.

This pre-commitment does not authorize running the 2025 evaluation: `configs/experiments/statarb_historical_etf_wf_v4.yaml` remains unmodified and `pre-registered`, per Addendum 3's prohibition on unauthorized status changes and Addendum 11's requirement that the holdout runner gate pass independent review first.

## v4 DEV/2024-validation verification basis, recorded before any 2025 number exists

**Date (PT):** 2026-09-17. `configs/experiments/statarb_historical_etf_wf_v4.yaml` sha256 `e8acafc514d87267d4dc3965212d0e8ed7dc2da600f34ff758b00bf6e49db003` (unchanged by this record — recorded here and in `research/experiment-ledger.csv`, never in the YAML itself, per this program's own no-retune/no-observed-outcome-edits convention).

From the already-executed, already-committed DEV/2024-validation run (`research/experiment-ledger.csv`, `config_sha256` matching the hash above on all three rows):

| Bucket | Windows | Qualifying | Fallback | Grid-selected |
|---|---|---|---|---|
| `dev_formation` (2015–2023) | 25 | 4 | 4 | 0 |
| `boundary_2023_2024` | 1 | 0 (no-trade) | — | — |
| `val_2024` | 3 | 0 (no-trade) | — | — |

`n_grid_selected_windows = 0` across every qualifying window in this run: all 4 `dev_formation` windows that found a cointegrated pair used the insufficient-trades fallback (now sourced from this config's own `selection_objective.insufficient_trades_fallback` block, not a separate Python constant — see the runner script and `wf_v4_orch.run_walk_forward_study`'s `fallback_params` argument), never a grid-selected candidate. This is a credible null on the grid-search path specifically (the validation window never had enough qualifying trades to clear `MIN_VALIDATION_TRADES` for any grid cell), reported honestly rather than loosened to force a non-null result.
