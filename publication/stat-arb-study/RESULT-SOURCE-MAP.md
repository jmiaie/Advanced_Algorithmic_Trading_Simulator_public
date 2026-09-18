# RESULT-SOURCE-MAP.md — D10-B `stat-arb-study`

Every quantitative statement in `TECHNICAL-PAPER.md`, `CASE-STUDY.md` and `CLAIM-REGISTER.md` maps to an
exact artifact field below. JSON pointers are relative to the artifact root.

Artifact hash notation: `` `path` (sha256 `64-hex`) ``. All hashes are re-verified by
`scripts/publication_pack.py check` (fail-closed).

Canonical artifact references used throughout:

- **HOLD** = `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json` (sha256 `b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66`) — PRIMARY 2025 artifact
- **DEV** = `results/historical_oos/statarb_hist_etf_wf_v4_dev_formation.json` (sha256 `52a45f3b8bbd59ba0795442b3cc4ed58d1c2b2465b0b89f99d012f715e7e3146`)
- **BND** = `results/historical_oos/statarb_hist_etf_wf_v4_boundary_2023_2024.json` (sha256 `e311a0e27b3be19ff66c496849a87717852f027c03fa4210aa641b67709b25f2`)
- **VAL** = `results/historical_oos/statarb_hist_etf_wf_v4_val_2024.json` (sha256 `d59c9747044262a2b736f455a5924fde13dedadb5ff8567d3964fa9b919dfbb8`)
- **CFG** = `configs/experiments/statarb_historical_etf_wf_v4.yaml` (sha256 `5768fd10b63c0436f3ff5aa1863374b348cca5bf72ccfb7d1b01a7ae0d95376d`)
- **MAN** = `data/manifests/yf_stat_arb_etfs_daily_2015_2025_v1.json` (sha256 `b2557499091cd87e412365db2a4a32eac7134995bf641e35e5676114f6cc2e30`)
- **LED** = `research/experiment-ledger.csv` (sha256 `85fd6bf54713d640676bb4174ddfdf9fe2833762d0d919b9ed8a9e7c2af57dd8`)
- **AUD** = `research/holdout-audit.md` (sha256 `d1f15db859957ff88a186474892e2cde2e63be6dc0b35bb5c636321dbcf4bf24`)
- **VALDOC** = `research/statistical-arbitrage-validation.md` (sha256 `dde09a88c6c69b6027b18ed316156686dcb2555fd8b28804dec08e1633ce1370`)

---

## A. Primary 2025 result (the subject of the paper)

| ID | Statement | Artifact | JSON pointer |
|----|-----------|----------|--------------|
| C-01 | Bucket `holdout_2025` contains 3 evaluation windows | HOLD | `period_name`, `n_windows` |
| C-02 | 0 qualifying windows / 0 windows with a selected pair in 2025 | HOLD | `n_qualifying_windows`, `selected_pairs` |
| C-03 | All 3 windows are `no_trade = true` | HOLD | `windows[0..2].no_trade`, `n_no_trade_windows` |
| C-04 | 55 candidate pair tests per window (165 total) | HOLD | `windows[i].n_candidate_tests` (i=0,1,2) |
| C-05 | 0 BH-FDR survivors in each window | HOLD | `windows[i].n_fdr_survivors` |
| C-06 | No fallback parameters were used in 2025 | HOLD | `n_fallback_windows`, `windows[i].used_fallback` |
| C-07 | No selected pair, no frozen signal parameters in 2025 | HOLD | `windows[i].selected_pair`, `windows[i].frozen_params` (both `null`) |
| C-08 | The 2025 parameter-distribution arrays are empty | HOLD | `param_distribution` (`entry_z`/`exit_abs_z`/`kalman_process_variance`/`trailing_z_window` all `[]`) |
| C-09 | 2025 test windows are 2025-01-08→2025-04-09, 2025-04-10→2025-07-11, 2025-07-14→2025-10-09 (window_index 30/31/32) | HOLD | `windows[i].window_index`, `.test_start`, `.test_end` |
| C-10 | Trailing validation sub-windows end strictly before each test start (no look-ahead) | HOLD | `windows[i].validation_end` < `windows[i].test_start` (2025-01-07<2025-01-08; 2025-04-09<2025-04-10; 2025-07-11<2025-07-14) |
| C-11 | No Sharpe / drawdown / turnover / return figure exists for 2025 | HOLD | absence of any return metric in `windows[i]`; `param_distribution` empty (structural absence) |

## B. Pre-2025 buckets (context for the 2025 null)

| ID | Statement | Artifact | JSON pointer |
|----|-----------|----------|--------------|
| C-12 | DEV/formation bucket: 25 windows, 4 qualifying, 21 no-trade | DEV | `n_windows`, `n_qualifying_windows`, `n_no_trade_windows` |
| C-13 | All 4 DEV qualifying windows used the pre-specified insufficient-trades fallback (0 grid-selected windows) | DEV, CFG | `n_fallback_windows`; `windows[i].used_fallback`; CFG `freeze_record.selection_basis.n_grid_selected_windows_total` (= 0) |
| C-14 | DEV fallback parameters = entry_z 2.0, exit_abs_z 0.5, trailing_z_window 40, kalman_process_variance 1.0e-4 | DEV, CFG | DEV `windows[1].frozen_params`; CFG `selection_objective.insufficient_trades_fallback` |
| C-15 | DEV selected pairs: XLP/XLU (×3), XLU/XLV (×1) | DEV | `selected_pairs`, `windows[i].selected_pair` |
| C-16 | DEV candidate tests total 1,339; 4 FDR survivors (1 per qualifying window) | DEV | `sum(windows[*].n_candidate_tests)`, `sum(windows[*].n_fdr_survivors)` |
| C-17 | Boundary bucket: 1 window (window_index 25, 2023-10-06→2024-01-05), 55 tests, 0 survivors, no-trade | BND | `n_windows`, `windows[0]` |
| C-18 | 2024 validation bucket: 3 windows (window_index 26/27/28), 55 tests each, 0 survivors, all no-trade | VAL | `n_windows`, `windows[i]`, `n_no_trade_windows` |
| C-19 | All-bucket totals: 32 windows, 1,724 candidate tests, 4 FDR survivors, 4 windows with a selected pair | DEV+BND+VAL+HOLD | `sum(n_candidate_tests)`, `sum(n_fdr_survivors)`, `count(no_trade == false)` — reproduced in `tables/bucket-summary.csv` |
| C-20 | The v4 boundary and 2024-validation artifacts are **byte-identical** to the v3 artifacts (all-no-trade buckets unaffected by the v4 fixes); DEV is not | v3/v4 artifacts, LED | file hashes `e311a0e2…` (boundary) and `d59c9747…` (val) identical across v3/v4; DEV `c7ca93a6…` (v3) ≠ `52a45f3b…` (v4) |

## C. Method / design (frozen configuration)

| ID | Statement | Artifact | Pointer |
|----|-----------|----------|---------|
| C-21 | Universe: 19 ETFs in 4 within-group buckets (A broad equity 4, B sectors 10, C Treasuries 3, D metals 2); pairs only within group | CFG | `universe.groups`, `universe.pair_constraint` |
| C-22 | Walk-forward geometry: formation 504, validation 126, test 63, step 63; pair rediscovery per window | CFG | `walk_forward.*` |
| C-23 | Signal grid: entry_z {1.5, 2.0, 2.5} × exit_abs_z {0.25, 0.5, 0.75} × trailing_z_window {20, 40, 60}; Kalman process variance {1e-5, 1e-4, 1e-3}; observation variance 1e-2 | CFG | `signal` (grid), `kalman.*` |
| C-24 | Selection objective: net Sharpe (BASE cost, Static OLS on the validation block), min 10 validation trades, tie-break lower max drawdown then lower turnover; fallback when trades are insufficient | CFG | `selection_objective.*` |
| C-25 | Cost scenarios GROSS (`analytical_zero`), BASE, STRESS: commission 0.005/share + 1.0 min; spread/slippage/impact per side BASE 1.0/1.0/0.5 bps, STRESS 3.0/3.0/2.0 bps; borrow BASE 50 bps, STRESS 200 bps annualized | CFG | `costs.BASE`, `costs.STRESS`, `costs.analytical_zero` |
| C-26 | Cost scenarios are **assumptions, not measured historical execution costs** | CFG | `costs.notes` |
| C-27 | Sizing: fixed gross notional, 1.0× allocated NAV, max gross leverage 1.0, fixed-shares banned for the primary run | CFG | `sizing.*` |
| C-28 | Data interval daily; raw data not committed (manifests only) | MAN | `interval`, `sha256`, `row_counts` |
| C-29 | Dataset coverage 2015-01-02 → 2025-12-31, 52,361 rows, 0 missing rows per symbol; XLRE begins 2015-10-08 | MAN | `actual_start`, `actual_end`, `row_counts`, `missing_counts`, `coverage_notes` |
| C-30 | Dataset canonical `a4082398…` equals the freeze record's `dataset_canonical_sha256` | MAN, CFG | MAN `sha256.dataset_canonical`; CFG `freeze_record.dataset_canonical_sha256` |

## D. Static OLS vs Kalman — evidence coverage (partial, disclosed)

| ID | Statement | Artifact | Pointer |
|----|-----------|----------|---------|
| C-31 | `validation_ols_sharpe` is `null` in **all 32** windows | DEV+BND+VAL+HOLD | `windows[*].validation_ols_sharpe` |
| C-32 | `validation_kalman_sharpe` is populated in **3** windows only, all in DEV: −0.14987589534869653 (window_index 2), 0.8398060100944159 (3), −2.117478227370739 (4) | DEV | `windows[1..3].validation_kalman_sharpe` |
| C-33 | No 2025 window carries either comparison metric | HOLD | `windows[*].validation_ols_sharpe` / `.validation_kalman_sharpe` = `null` |

## E. Provenance / governance

| ID | Statement | Artifact | Pointer / note |
|----|-----------|----------|----------------|
| C-34 | 2025 was executed once, immediately after the freeze, with no change in between | git | commit bodies of `28be77fb` (freeze) and `d1ef6fd0` (evidence); LED rows for `holdout_2025` |
| C-35 | One boundary-straddling window was excluded from the 2025 bucket by design and surfaced via the runner warning (not silently dropped) | git, HOLD | `d1ef6fd0` commit body; HOLD `n_windows` = 3 |
| C-36 | Pre-committed 2025 label is FINAL 2025 WALK-FORWARD EVALUATION, not an untouched holdout | AUD | holdout-audit label sections; commit bodies |
| C-37 | 5 of 19 symbols overlap the FDM D9-A executed universe | AUD | cross-program exposure disclosure section |
| C-38 | A superseded exploratory v1 study on a different (equity-pair) universe also executed its own 2025 window with 0 pairs selected and zero symbol overlap with this universe | `results/historical_oos/statarb_hist_oos_v1_holdout_2025.json` (sha256 `4fd88a8204ecb4b92cc5017764fcaf0e159b20b202d971b9edc85d4128d6f271`), AUD | v1 artifact; AUD version history |
| C-39 | `pair_persistence` / `hedge_variation` diagnostics are not present in any v4 artifact | DEV+BND+VAL+HOLD | absence of such keys in the artifact schema (`windows[*]` key set) |
| C-40 | Superseded lineage inventory: the v1 result artifacts (3 files) and the v3 result artifacts (3 JSON + window-boundaries CSV) remain in `results/historical_oos/` under their original labels; the v2 lineage is config-only (`configs/experiments/statarb_historical_etf_wf_v2.yaml` — v2 produced no result artifacts before invalidation); the ledger holds 23 rows including one `statarb_hist_etf_wf_v2_INVALIDATED`, four v1 rows (`dev_formation`, `val_2024`, `holdout_2025`, `prereg`) and three explicit `…SUPERSEDED_*` rows | `research/experiment-ledger.csv` (sha256 `85fd6bf54713d640676bb4174ddfdf9fe2833762d0d919b9ed8a9e7c2af57dd8`) — see `scripts/claim_crosscheck.py` for the enforced values; `configs/experiments/statarb_historical_etf_wf_v2.yaml`; `results/historical_oos/statarb_hist_etf_wf_v3_*` | enforced by `claim_crosscheck.py` (no citation depends on a superseded artifact's content) |

## F. Pointers deliberately NOT cited
- Any Sharpe / drawdown / turnover / return / alpha number for 2025 — **does not exist** (C-11).
- Any factor attribution, capacity or market-impact estimate — **not produced** by this lane.
- Any 2025 C-selection or threshold-tuning result — **not produced**; no tuning path exists for this experiment.
