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
