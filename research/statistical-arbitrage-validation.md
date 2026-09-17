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
  **INVALIDATED BEFORE HOLDOUT** (2026-09-16T20:30:00Z). DEV+2024-validation
  ran successfully against the real acquired dataset (no 2025 evaluation
  ever occurred for v3 -- see holdout-audit.md), but independent code
  review found two defects in `src/stat_arb_engine/wf_v3_*.py`, and a third
  was found independently while porting the fix into v4:
  1. **Execution timing**: a position decided using information through
     bar t (the causal z-score/hedge-ratio computed from bar t's own
     close) was filled AT bar t's own price -- an unrealistic same-close
     assumption, most consequential for the sequential Kalman hedge ratio.
  2. **Turnover tie-break omitted**: `_validation_tiebreak_metric`'s own
     docstring said turnover was "omitted... not expected to bind,"
     contradicting the pre-registered spec's
     `tie_break: [lower_max_drawdown, lower_turnover]`.
  3. **Drawdown tie-break sign inversion** (found while implementing v4's
     fix for #2, not part of the original review): v3's tie-break
     comparison, worked through with concrete numbers, actually preferred
     the *deeper* of two tied-Sharpe candidates' drawdowns -- the opposite
     of "lower max drawdown." Caught by a v4 unit test, never manifested
     in an executed v3 run.
  4. **Drawdown calculation baseline** (found via independent external
     review 2026-09-17, verified directly against the code and the actual
     v4 artifact before acting on it; also present in v4, fixed there --
     see below): `simulate_pair_backtest`'s max_drawdown calculation had
     two distinct sub-bugs. It excluded the pre-trade starting NAV from
     its running peak, which understates -- in the worst case zeroing
     out -- drawdown for a window whose losses start from a fresh
     baseline (e.g. a single-bar loss from entry costs alone). It also
     divided by the constant `allocated_nav` rather than the running peak
     at each point, which -- a second independent review caught this
     precisely, correcting an earlier draft of this note that had the
     direction backwards -- can only OVERSTATE drawdown magnitude in
     isolation (e.g. allocated_nav=100, NAV path 100 -> 110 -> 95: -15.00%
     against the constant vs. the correct -13.64% against the peak of
     110), since the running peak is always >= allocated_nav once the
     first sub-bug is fixed. The two sub-bugs' combined effect on any
     given window's reported number therefore depends on that window's
     own NAV path, not a single universal direction. v3's own
     already-committed DEV/2024 max_drawdown figures carry both defects
     and are not corrected (v3 is not re-run).
  See the config file's header for the full corrective record. Superseded
  by v4.
- **v4** (`statarb_hist_etf_wf_v4`, `configs/experiments/statarb_historical_etf_wf_v4.yaml`):
  the active conforming experiment, **status: frozen-for-holdout** (frozen
  2026-09-17, owner-authorized, after independent review found zero open
  code/provenance items at HEAD `3527634`; see the config file's
  `freeze_record` block and `research/holdout-audit.md` for the full
  basis). Identical
  dataset/universe/grid/costs/hypotheses to v3; fixes defects 1-3 above via
  `src/stat_arb_engine/wf_v4_backtest.py` (execution-lag enforcement:
  `lag_for_execution` + a lag-aware `simulate_pair_backtest`) and
  `wf_v4_orch.py` (corrected three-level tie-break: Sharpe -> drawdown ->
  turnover, with the correct sign on each). Defect 4 (drawdown calculation
  baseline) was caught by a separate independent review after v4's initial
  run and fixed in a follow-up commit to the same `wf_v4_backtest.py`
  module -- see below. `tests/test_wf_v4_pipeline.py` and
  `tests/test_wf_v4_holdout_gate.py` together now hold 11 tests (all
  synthetic fixtures only), including dedicated causality tests proving: a
  later price cannot change an earlier decision, a decision at t cannot
  alter holdings at t (only from t+1 onward), and the hedge ratio used to
  execute a t-decided trade is the estimate as of t, never a later Kalman
  update, plus a dedicated regression test for defect 4 reproducing a
  single-executed-bar loss that the old formula always reported as exactly
  0% drawdown. The full repo suite (78 tests, including the unaffected
  pre-existing v3/Directive-3-gate tests) passes, along with a clean
  ruff/mypy run.

  **Run against the real acquired dataset** (DEV 2015-2023 + 2024
  validation only; 2025 structurally excluded, not just status-gated --
  see `tests/test_wf_v4_holdout_gate.py`): 25 dev_formation windows (4
  qualifying, 21 no-trade), 1 boundary window (no-trade), 3 val_2024
  windows (all no-trade -- a credible null result, not loosened to force a
  trade). The no-trade windows' summaries are byte-identical to v3's for
  the same windows (pair selection is unaffected by any of the four
  defects, so this is an expected and reassuring consistency check, not
  evidence the fixes did nothing); the dev_formation summary, where trades
  did occur, differs from v3's. Two dev_formation hashes now exist in the
  ledger for v4 itself: `a1b033f9...` (post defects 1-3, pre defect-4 fix)
  and the current `52a45f3b...` (post defect-4 fix) -- both preserved in
  `research/experiment-ledger.csv` per this program's no-overwrite audit
  convention; only `52a45f3b...` is current. Re-running after the defect-4
  fix changed every qualifying window's reported max_drawdown magnitude but
  did not flip pair selection or any tie-break winner in this already-
  executed run.

  **Disclosed plainly, not buried in a hash (flagged by independent
  review):** all 4 of dev_formation's qualifying windows have
  `used_fallback: true` -- the validation grid search
  (`_validation_grid_search_signal_params`, `MIN_VALIDATION_TRADES=10`)
  never found an `(entry_z, exit_abs_z, trailing_z_window)` combination
  clearing the minimum trade count in any of them, so every one ran on the
  pre-registered fallback (`entry_z=2.0, exit_abs_z=0.5,
  trailing_z_window=40`), not a validated selection. The ledger now records
  this as a first-class number (`n_grid_selected_windows: 0` alongside
  `n_qualifying_windows: 4`, plus `config_sha256` for the exact config that
  produced it), not just inferable from per-window JSON. Because the
  pre-registered primary objective (net Sharpe on the validation block) is
  gated by this same `MIN_VALIDATION_TRADES` filter, the parameter-selection
  step is effectively inert for this entire run -- the reported dev_formation
  numbers are fixed-parameter results, not validated-parameter results. This
  is accepted as a genuine null (the grid/FDR/universe/`min_trade_count`
  are **not** being relaxed in response to it -- that would be exactly the
  retune-on-observed-outcome this program's protocol forbids), but it should
  weigh on how much confidence review places in the dev_formation result
  before authorizing freeze. `tests/test_wf_v4_pipeline.py` now has two
  dedicated tests for `_validation_grid_search_signal_params` that were
  previously untested in either direction: one confirming it returns `None`
  when no candidate clears the threshold (a flat, zero-variance spread), one
  confirming it returns a real selection when trades do clear it (an
  engineered spread with reliable entry/exit crossings).

  **FINAL 2025 WALK-FORWARD EVALUATION** (executed once, 2026-09-17,
  immediately after the freeze commit above -- not "UNTOUCHED 2025
  HOLDOUT," per Directive #9 Addendum 12: 5 of the 19 symbols overlap
  FDM's already-executed D9-A universe; see `research/holdout-audit.md`
  for the full cross-program exposure disclosure). Config sha256 at
  execution time: `5768fd10b63c0436f3ff5aa1863374b348cca5bf72ccfb7d1b01a7ae0d95376d`
  (post-freeze; the pre-freeze reviewed config was
  `e8acafc514d87267d4dc3965212d0e8ed7dc2da600f34ff758b00bf6e49db003` --
  only the `status` field and the `freeze_record` block differ between
  the two, no numerical parameter). Result artifact:
  `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json`,
  sha256 `b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66`.

  **Result: a credible, complete null.** All 3 pure-2025 test windows
  (`test_start`/`test_end` both in 2025) found `n_fdr_survivors: 0` --
  the Engle-Granger + Benjamini-Hochberg FDR pair scan (55 within-group
  candidate tests per window) selected no cointegrated pair in any of
  them, so every window is `no_trade: true` with `selected_pair: None`
  and `frozen_params: None`. No trade was ever placed against 2025 data;
  there is no Sharpe, drawdown, or turnover number to report for this
  period, and none is fabricated. This is not loosened or re-run under
  different settings in response to the null -- doing so would be
  exactly the retune-on-observed-outcome this program's protocol
  forbids. One additional window (`window_index=29`, test period
  straddling 2024-2025) was excluded from this bucket by design (it
  tests neither a pure-2024 nor a pure-2025 period) and is reported via
  a runner warning, not silently dropped -- see
  `tests/test_wf_v4_holdout_gate.py::test_v4_runner_holdout_empty_bucket_guard_actually_fails`
  for the regression test proving an empty 2025 bucket cannot pass
  silently.

  **Reading this result in context:** this null is consistent with,
  not contradicted by, the DEV/2024-validation pattern above -- pair
  selection was already weak throughout this dataset (`boundary_2023_2024`
  and `val_2024` were both all-no-trade; only 4 of 25 `dev_formation`
  windows found a qualifying pair at all, and every one of those used
  the fallback rather than a validated selection). The 2025 result
  extends that same pattern rather than diverging from it. No claim of
  trading-alpha or risk-adjusted outperformance is or can be made from
  this study; the honest finding is that formation-period cointegration
  in this authoritative ETF universe did not persist into a tradeable
  2025 signal under this pre-registered design.

  Per Directive #9 Addendum 11/20: this result and the freeze record
  above are now committed. **STOPPING FOR INDEPENDENT POST-HOLDOUT
  REVIEW** -- no further code, parameter, or config change on this
  experiment; no merge; D10 not started.
