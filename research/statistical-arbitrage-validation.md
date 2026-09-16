# Statistical Arbitrage Validation Report

## Objective
Harden the repository into a recruiter-facing **Statistical Arbitrage & Execution Research Engine** with temporal discipline, accounting correctness, execution-cost awareness, and truthful public claims.

## Defects verified before remediation
- README claimed a dynamic hedge ratio while the implementation used a fixed full-history `np.polyfit` estimate.
- Pair signals could be generated from current-leg / stale-leg observations because events were processed one symbol at a time.
- Signal normalization allowed the current spread to participate in its own rolling mean / standard deviation estimate.
- Cumulative fill cash flow was mislabeled as PnL/equity.
- The execution simulator was stylized but not clearly labeled as such.
- Pair scanning used raw p-values without multiple-testing correction.
- Automated tests were materially too weak / absent for quantitative research hardening.

## Remediation summary
- Added synchronized same-timestamp event buffering with monotonic and duplicate timestamp validation.
- Tightened the synchronized buffer to reject duplicate timestamps even after a batch has already been emitted.
- Added a portfolio ledger with cash, signed quantities, average cost, realized/unrealized PnL, commissions, execution costs, exposure, and NAV.
- Added explicit gross/net PnL and cost-basis-style snapshot fields so equity is not inferred from cumulative cash flow.
- Formalized the static baseline as **Static OLS Hedge Ratio** and added a sequential Kalman hedge estimator.
- Added chronological split and walk-forward helpers with formation-only pair selection, explicit pre-OOS cutoffs, and BH/FDR control.
- Changed signal normalization to use trailing observations only, excluding the current spread from its own mean/std estimate.
- Added stylized cost-aware execution simulation with partial fills and VWAP.
- Repositioned the README to make limitations, research chronology, and claim accuracy explicit.

## Static vs dynamic hedge models
- **Static OLS Hedge Ratio**: baseline regression fit on the relevant history window.
- **Sequential Kalman model**: decision-time filtering of `alpha_t` and `beta_t` with configurable process / observation variance.
- Any smoother should be treated as diagnostic only; the current public engine exposes the filtered path used for sequential estimation.

## Multiple testing control
- Pair scans now apply **Benjamini-Hochberg / FDR** correction to Engle-Granger scan p-values.
- Both raw and adjusted p-values, rejection flags, and test counts are exposed in the research outputs.
- Residual ADF is used as a secondary diagnostic filter rather than a second independent hypothesis test, avoiding double-counting the same evidence.

## Execution and accounting assumptions
- Execution remains a **stylized scenario engine** with configurable commissions, spread, slippage, and impact costs.
- Borrow-cost configuration exists, but no empirical financing engine is claimed in this public repo.
- Partial fills, VWAP, and unfilled remainder behavior are modeled.
- This is **not** empirical market microstructure reconstruction and should not be represented as such.

## Validation status
- Offline unit tests cover synchronization, no-stale-leg behavior, duplicate emitted timestamps, static OLS slope recovery, sequential Kalman behavior, BH/FDR correction, chronological splits, walk-forward formation-only chronology, trailing-only normalization, sizing bounds, execution partial fills, and portfolio accounting invariants.
- Reproducible OOS experiment artifacts: **Results pending reproducible OOS run**.

## Limitations and unfinished items
- Full factor attribution, capacity / ADV constraints, block-bootstrap uncertainty estimates, and deeper sensitivity studies remain future work.
- The public universe examples are still static and therefore susceptible to survivorship / convenience bias.
- Walk-forward helpers are present, but the repository does not yet ship precomputed public OOS result packs.
- Read-only audit note: `jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst` currently makes broader public claims than this validated engine supports; recommended path is consolidation or claim narrowing rather than parallel divergence.

## Directive #3 release-gate verification

Regression coverage for the D3 gate lives in `tests/test_directive3_gates.py`:

1. **Same-timestamp pair signals** — strategy waits for both legs; incomplete `MARKET_BATCH` payloads are rejected.
2. **Marked-to-market NAV** — ledger `nav` equals cash + net market value and moves with marks without new fills.
3. **Decision-time pair research** — `as_of_frame` / walk-forward cutoffs keep selection and hedge fit on formation-only data.
4. **Static-vs-sequential-Kalman distinction** — explicit `estimation_mode` labels (`static_batch_ols` vs `sequential_filtered_kalman`) plus causal Kalman prefix checks.
5. **Correct analytics** — conventional Sharpe and trailing max-drawdown formulas; NAV-based net return with qualified cost add-back for gross return.

Qualified claim: this verifies research-engine invariants on synthetic/unit fixtures. It does **not** assert empirical out-of-sample trading performance.

## Directive #9 D9-B status (corrective note)

- **v1** (`statarb_hist_oos_v1_*`, `configs/experiments/statarb_historical_oos_study_v1.yaml`):
  exploratory sector-equity study, universe = `PairFinder.SECTOR_PAIRS`.
  **SUPERSEDED / EXPLORATORY / NON-CONFORMING** to the authoritative D9-B spec
  (wrong universe). Preserved unmodified in the ledger and results tree.
- **v2** (`statarb_hist_etf_wf_v2`, `configs/experiments/statarb_historical_etf_wf_v2.yaml`):
  **INVALIDATED BEFORE HOLDOUT**. A tracker comment previously recorded this
  as "FINAL CONFIGURATION FROZEN," but independent audit found no dataset
  manifest, no pipeline code, and no DEV/2024-validation artifacts anywhere
  in the repository for this config -- only the YAML specification document
  existed. No 2025 v2 result was ever executed or exists. See the config
  file's header for the full corrective record (original freeze timestamp,
  invalidation timestamp, reason).
- **v3** (`statarb_hist_etf_wf_v3`, `configs/experiments/statarb_historical_etf_wf_v3.yaml`):
  the active conforming experiment, **status: pre-registered**. A real
  walk-forward pipeline now backs it (`src/stat_arb_engine/wf_v3_*.py`):
  per-window pair rediscovery (not chronology-only), a fair static-OLS-vs-
  Kalman comparison (Kalman is traded, not diagnostics-only), fixed-gross-
  notional sizing, and GROSS/BASE/STRESS costs with real daily-accrued
  borrow (unlike `execution.CostModel`, whose `borrow_cost` is a stub that
  always returns 0.0). 19 offline tests pass (`tests/test_wf_v3_pipeline.py`,
  synthetic fixtures only). **Not yet run against real data**: dataset
  acquisition is blocked by this session's egress policy (Yahoo Finance
  denied with policy 403; see `scripts/acquire_yf_stat_arb_etfs_daily.py`).
  No DEV/2024-validation results exist yet, and no 2025 evaluation of any
  kind has occurred for v3.
