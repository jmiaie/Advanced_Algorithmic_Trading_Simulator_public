# Pair Selection Under Multiple-Testing Control in an ETF Statistical-Arbitrage Walk-Forward Study

**A pre-registered historical walk-forward evaluation with a complete null in 2025**

D10-B publication pack · experiment `statarb_hist_etf_wf_v4` · directive #10 (publication layer over accepted
D9 evidence) · 2026-09-18

> **Evidence basis.** Every number below is read from the accepted v4 artifacts on the accepted HEAD
> `d1ef6fd0` and its frozen config. Pointer IDs (`C-nn`) resolve field-by-field in
> [`RESULT-SOURCE-MAP.md`](RESULT-SOURCE-MAP.md). The source gates are recorded in
> [`SOURCE-GATE.md`](SOURCE-GATE.md). Nothing here was re-run, re-tuned, or re-derived.
>
> **What this study is not.** It is not a strategy-performance report. In the 2025 evaluation period the
> pre-specified screen selected no pair, the portfolio held no position, and therefore **no return, Sharpe,
> drawdown or turnover quantity exists for 2025** (C-11). None is estimated, modelled or implied here.

---

## Abstract

We report a pre-registered historical walk-forward evaluation of a within-group ETF pairs strategy in which
pair selection is gated by Engle-Granger cointegration and a Benjamini-Hochberg false-discovery-rate (FDR)
screen, with signal thresholds selected on a validation block under an explicit insufficient-trades fallback.
Across four walk-forward buckets (32 windows, 1,724 candidate pair tests) the screen produced exactly four
FDR survivors — **all in the 2015-2023 development/formation bucket, all executed with fallback parameters, and
none selected from the signal grid** (C-12, C-13, C-19). The 2024-validation bucket (3 windows) and the
boundary bucket (1 window) are entirely no-trade (C-17, C-18). **In the 2025 evaluation bucket, all three windows
found zero FDR survivors, hence no pair, hence no position: a complete null trading outcome** (C-02, C-03,
C-04). Because no pair passed the frozen screen, the portfolio held no position; the 2025 period therefore
contributes an opportunity-screening outcome rather than a trading-performance estimate. We state the null plainly, disclose the cross-program exposure that
prevents the 2025 period from being an untouched holdout (C-36, C-37), and deliberately make no claim of edge,
alpha, economic significance or capacity (CL-21, CL-22).

## 1. Research Question

Does a *pre-specified, multiple-testing-controlled* pair-selection procedure in a static ETF universe produce
qualifying, tradeable statistical-arbitrage opportunities out-of-sample — and what happens to the study's
evidentiary claims when it does not produce any?

The question is deliberately ordered this way. The design fixes the screening rule in advance
(§5, §6) and then asks what the pipeline actually emits; it does not search for a specification that trades.
In this experiment the honest answer for the evaluation period is "it emitted nothing", and the secondary
question — how such a null must be reported — is answered in §11 and §12.

## 2. Hypothesis

**H0 (pre-specified form):** within-group ETF pairs selected by formation-window cointegration and screened at
a 5% BH-FDR level do not reliably survive into out-of-sample test windows; consequently the number of windows
in which a signal is actually traded, and the number selected by grid search, may both be small.

**Direction of prediction:** H0 predicts *qualifying-window scarcity*, not a particular profit or loss. This
matters for interpretation: the design is falsifiable only in the sense that windows *can* qualify — and in the
development bucket four did (C-12, C-13), which establishes the mechanism fires, while zero did in 2025 (C-02).

**No alternative hypothesis was tested**, no parameter was chosen to improve outcomes, and no hypothesis was
changed after execution (D10 scope; `SOURCE-GATE.md` Field 9).

## 3. Data

- **Dataset:** `yf_stat_arb_etfs_daily_2015_2025_v1`; 19 US-listed ETFs; daily bars; 2015-01-02 → 2025-12-31;
  52,361 rows; 0 missing rows per symbol (C-28, C-29).
- **Coverage exception:** XLRE starts 2015-10-08; the walk-forward engine excludes a symbol only from windows
  that predate its own inception — no substitution was made (C-29, CFG `universe` note).
- **Frozen identity:** canonical SHA-256 `a4082398991400c171062ee4ff2b12dba87e84c4f0133462ed9ee2fbb13f9fa8`
  (C-30), identical in the manifest and in the frozen config's freeze record.
- **Reproducibility caveat (disclosed):** raw per-symbol CSVs are not committed to the repository
  (`data/` holds manifests only), so the canonical hash is an accepted, manifest-declared identity — it cannot
  be recomputed from repository bytes (`SOURCE-GATE.md` Field 7 limit).
- **Price basis:** vendor daily bars (yfinance family). Not exchange-official data; no tick or quote data.

## 4. Universe

Nineteen ETFs in four within-group buckets; **pairs are formed only inside a group** (C-21):

| Group | Members | n |
|---|---|---|
| A — broad equity | SPY, QQQ, DIA, IWM | 4 |
| B — US sectors | XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY | 10 |
| C — Treasuries | SHY, IEF, TLT | 3 |
| D — precious metals | GLD, SLV | 2 |

Within-group pairing bounds the candidate set and prevents economically spurious cross-asset pairings, at the
cost of a smaller search space (relevant to §6). **Universe selection is static and reflects present-day
knowledge of liquid, long-lived ETFs** — a survivorship/convenience bias disclosed in §13 and in
`SOURCE-GATE.md` Field 13 item 2.

## 5. Pair-Selection Procedure

Per window, on the formation block only:

1. Form all within-group pairs of symbols available in that window.
2. Estimate the pair's cointegrating relation (Engle-Granger, static OLS on the formation block).
3. Test the residual for stationarity (ADF) as a secondary diagnostic.
4. Screen for multiple testing with BH-FDR across the window's candidate set (§6).
5. **Only surviving pairs** proceed to validation, where signal thresholds are chosen from the grid.
6. If validation produces fewer trades than the pre-specified minimum, apply the **insufficient-trades
   fallback** rather than forcing a grid selection (§7, C-24).
7. Trade the selected pair in the test window under a fixed-gross-notional sizing rule (C-27).

Two properties are load-bearing: **selection is never conditioned on profitability** (no profitable pair
search, no post-hoc pair substitution), and **pair discovery is repeated per window** rather than fixed once
(C-22). Both were fixed before the freeze and are unchanged in the accepted result.

## 6. Multiple-Testing Controls

Candidate counts per window are large relative to the number of economically connected pairs, so an unbiased
screen is essential. Across the four buckets the pipeline ran **1,724 candidate tests** and produced
**4 survivors** (C-19; `tables/bucket-summary.csv`), i.e. a survivor rate of 0.23%.

| Bucket | Windows | candidate tests | FDR survivors |
|---|---:|---:|---:|
| dev_formation (2015-2023) | 25 | 1,339 | 4 |
| boundary 2023-10 → 2024-01 | 1 | 55 | 0 |
| val_2024 | 3 | 165 | 0 |
| holdout_2025 | 3 | 165 | 0 |
| **total** | **32** | **1,724** | **4** |

Controls: BH-FDR applied to the per-window candidate set; residual ADF retained as a **secondary** diagnostic
only (it does not by itself admit a pair); no profitability-driven selection. **Disclosed ceiling:** block-bootstrap
uncertainty over the selection stage was **not** computed — the gate lists it as unfinished
(`SOURCE-GATE.md` Field 13 item 4), and no uncertainty estimate for the survivor count is claimed here.

## 7. Walk-Forward Design

Frozen geometry (C-22): formation 504 sessions, validation 126, test 63, step 63, pair rediscovery per window,
under-length windows rejected.

For each test window the ordering is strictly causal: **formation → validation (selection) → test**, with
`validation_end < test_start` verified in every window including all three 2025 windows (C-10). Threshold grid
(C-23): entry_z ∈ {1.5, 2.0, 2.5} × exit_abs_z ∈ {0.25, 0.5, 0.75} × trailing_z_window ∈ {20, 40, 60};
Kalman variant: process variance ∈ {1e-5, 1e-4, 1e-3}, observation variance 1e-2.

Selection objective (C-24): **net Sharpe under BASE costs, computed on the validation block with static OLS**,
minimum 10 validation trades, ties broken by lower max drawdown then lower turnover. If the validation trade
count is insufficient, the pre-specified fallback parameters apply
(entry_z 2.0, exit_abs_z 0.5, trailing_z_window 40, kalman_process_variance 1.0e-4) — this is a *fallback*, not
a tuned selection, and it is recorded per window (`used_fallback`).

## 8. Static OLS vs Kalman

The design pre-specifies a comparison between the static-OLS hedge and a sequential Kalman-filter hedge using
the same pair, the same signal rules, the same costs and the same sizing. **The accepted evidence for this
comparison is partial, and we report that rather than a comparison we do not have:**

- `validation_ols_sharpe` is `null` in **all 32** windows (C-31).
- `validation_kalman_sharpe` is populated in **3** windows only, all in the development bucket:
  −0.1499 (window_index 2), +0.8398 (3), −2.1175 (4) (C-32).
- No 2025 window carries either metric (C-33).

Consequently: **no claim is made about the relative merit of static OLS and Kalman hedging**, and the three
populated development-window values are reported only as what they are (three validation-block Sharpe
observations from a bucket that also produced all four traded windows). Treating those three numbers as a
hedge-method conclusion would be exactly the kind of overreach this pack is built to avoid.

## 9. Execution-Cost Model

Three scenarios are declared in the frozen config (C-25) and are **assumptions, not measured historical
execution costs** (C-26):

| Scenario | commission/share | min commission | spread (bps/side) | slippage (bps/side) | impact (bps/side) | borrow (bps p.a.) |
|---|---:|---:|---:|---:|---:|---:|
| GROSS (`analytical_zero`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BASE | 0.005 | 1.0 | 1.0 | 1.0 | 0.5 | 50 |
| STRESS | 0.005 | 1.0 | 3.0 | 3.0 | 2.0 | 200 |

Borrow accrues daily as annualized basis points / 252 on the short market value actually held that day.
Sizing is fixed-gross-notional at 1.0× allocated NAV with max gross leverage 1.0, and fixed-share sizing is
banned for the primary run (C-27). **Disclosed ceiling:** no market-impact calibration and no capacity
analysis were performed (`SOURCE-GATE.md` Field 13 item 4), so the cost model bounds *stylized* outcomes
only — and in the evaluation period there were no positions for it to price.

## 10. Results

### 10.1 Development / formation bucket (2015-2023), 25 windows
4 qualifying windows (all traded), 21 no-trade, 4 fallback-applied, **0 windows selected by grid search**
(C-12, C-13). The four traded windows used the fallback parameters on pairs XLP/XLU (three windows) and XLU/XLV
(one) (C-13, C-15). This bucket is also the only source of populated hedge-comparison metrics (§8).

### 10.2 Boundary bucket (2023-10-06 → 2024-01-05), 1 window
55 candidate tests, 0 survivors, **no-trade** (C-17).

### 10.3 2024 validation bucket, 3 windows
55 candidate tests per window (165 total), 0 survivors in each, **all no-trade** (C-18).

### 10.4 2025 evaluation bucket, 3 windows — the primary result of this paper

| window_index | test start | test end | candidate tests | FDR survivors | no_trade | selected pair |
|---:|---|---|---:|---:|---|---|
| 30 | 2025-01-08 | 2025-04-09 | 55 | 0 | true | — |
| 31 | 2025-04-10 | 2025-07-11 | 55 | 0 | true | — |
| 32 | 2025-07-14 | 2025-10-09 | 55 | 0 | true | — |

`n_qualifying_windows = 0`, `n_no_trade_windows = 3`, `n_fallback_windows = 0`, `selected_pairs = []`, and the
parameter-distribution arrays are empty (C-01…C-08). See `tables/window-level.csv` and
`figures/selection-funnel.svg` (generated from the artifacts; provenance in `RESULT-SOURCE-MAP.md`).

### 10.5 Cross-bucket summary
Of 32 windows and 1,724 candidate tests, 4 produced a selected pair — **all of them before 2024**, all via the
fallback rule. Per-bucket figures: `tables/bucket-summary.csv`.

### 10.6 Superseded-lineage check
The v4 boundary and 2024-validation artifacts are **byte-identical** to their v3 counterparts (hashes
`e311a0e2…` and `d59c9747…` match across versions); the DEV artifact differs (`c7ca93a6…` → `52a45f3b…`),
consistent with the DEV bucket being the only one where a fix could change a traded outcome (C-20).
**Superseded lineage inventory (C-40, CL-20):** the v1 result artifacts (3 files) and the v3 result artifacts
(3 JSON + window-boundaries CSV) remain in `results/historical_oos/` under their original labels; the v2 lineage
is config-only (no v2 result artifacts exist — v2 was invalidated before holdout) alongside one
`statarb_hist_etf_wf_v2_INVALIDATED` ledger row; the ledger holds 26 rows including four v1 rows, three v1 corrective rows and six
explicit `…SUPERSEDED_*` rows. **Disclosure (corrected 2026-09-18):** the `artifact_sha256` values recorded on the three v1 ledger rows resolve to no artifact present in this repository or its history (every blob scanned, dangling included); the v1 artifacts on disk are the slimmed forms (2,496 B `29b6effa…`, 2,163 B `e3396ba0…`, 2,156 B `4fd88a82…`) and the recorded values are unverifiable — disclosed by three corrective `…SUPERSEDED_UNVERIFIABLE_ARTIFACT_HASH` ledger rows (CL-20). Nothing here is edited, re-labelled or removed, and no citation in this pack
depends on a superseded artifact's *content*.

## 11. No-Trade Result

**The 2025 evaluation period produced no position, and that is the reported outcome** (C-02, C-03, C-06).

Three things this section does *not* do, each deliberately:

1. It does not attach a performance number to a period with no position. There is no 2025 Sharpe, return,
   drawdown or turnover figure in the accepted artifacts, and this pack does not manufacture one (C-11, CL-21).
2. It does not characterise the no-trade outcome as an implementation failure. The same screen admitted four
   windows in the earlier development bucket (C-12, C-13), the run completed and wrote its ledger row
   (C-34: ledger created 2026-09-17T03:44:34Z, evidence commit 03:48:08Z), and the single boundary-straddling
   window was excluded by design with a surfaced warning rather than silently dropped (C-35).
   The narrow interpretation CL-13 asserts exactly this and nothing more; it carries an explicit reversal
   condition.
3. It does not claim the strategy was "flat by choice" or "defensively correct in hindsight". The screen found
   no qualifying opportunity; the portfolio therefore held nothing. Motive is not observable in the artifacts.

What the no-trade result *does* tell a reader: for this universe, sample period and frozen rule set, the
formation-time cointegration + FDR gate did not admit an ETF pair in the evaluation year. Whether that
reflects a real absence of cointegration, or a screen calibrated too strictly for this asset class, **is not
determined by this design** (§13).

## 12. Negative / Null Findings

Stated as findings:

1. **Null (primary):** zero qualifying 2025 buy/sell opportunities under the pre-specified screen — no
   opportunity to measure, hence no measurement (C-02, C-03, C-11).
2. **Null (design-level):** grid selection **never** fired in any bucket: `n_grid_selected_windows_total = 0`
   (C-13). Every traded window used the fallback. The tuned-threshold path is therefore *untested in
   effect*, and no claim about grid-selected thresholds is possible.
3. **Weak pre-2025 signal:** 2024-validation and boundary buckets were entirely no-trade (C-17, C-18).
4. **Partial evidence:** the OLS/Kalman comparison produced almost no populated metrics (§8).
5. **Absent diagnostics:** pair-persistence and hedge-variation diagnostics are not present in the artifacts
   (C-39), so the gate's "where present" qualifier resolves to *not present* — no persistence claim is made.

## 13. Limitations

1. **Cross-program exposure.** 5 of 19 symbols (SPY, QQQ, IWM, TLT, GLD) overlap the already-executed FDM
   D9-A 2025 evaluation universe, and the design-origin chronology relative to FDM results is **unresolved**
   (C-36, C-37). The binding label is therefore **FINAL 2025 WALK-FORWARD EVALUATION**, *not* untouched holdout.
2. **Static, present-day universe** → survivorship and convenience bias (§4).
3. **No 2025 risk measurement.** Zero positions means zero volatility/drawdown/turnover observation. Absence of
   measurement is not evidence of stability, and this pack does not read it as such.
4. **Unfinished analyses (gate-confirmed):** factor attribution, capacity/liquidity, and block-bootstrap
   uncertainty (CL-21 — no such quantity is claimed).
5. **Stylized costs** (§9) — assumptions, not measured execution.
6. **Partial hedge-comparison evidence** (§8).
7. **Missing persistence/hedge diagnostics** (§12 item 5).
8. **Raw data not in-repo** (§3) — provenance verified to the frozen manifest, not to raw bytes.
9. **Rolling re-selection inside the 2025 bucket:** trailing validation sub-windows include earlier 2025
   sessions. Causally safe (`validation_end < test_start`, C-10), but the bucket is a rolling re-selection path,
   not a single frozen projection.
10. **Single evaluation invocation.** One run, one label; no stability check across repeated runs of the 2025
    evaluation was performed (and rerunning is prohibited in this lane).
11. **Evidence class.** Program-audited accepted evidence, not external peer review
    (`SOURCE-GATE.md` Field 9).
12. **Calendar coverage of the evaluation bucket.** The three 2025 windows span **2025-01-08 → 2025-10-09**; the
    remainder of the calendar year is not covered by the frozen 63-session geometry, and one boundary-straddling
    window was excluded by design with a surfaced warning (§10.2, C-35). No sentence in this pack characterises
    the result as a full-year statement.

## 14. Model Risk

- **Frozen-parameter model risk:** all thresholds are fixed ex ante (§7), so any bias in the grid or the
  fallback transfers directly into the (zero) 2025 outcome. The fallback used in every traded window
  (C-13, C-14) is a single parameter set, never re-estimated out-of-sample.
- **Filter-risk asymmetry:** a conservative screen that finds nothing is indistinguishable, within this design,
  from a screen that *would* find opportunities under a slightly different threshold. This cannot be resolved
  without changing the design — which this lane prohibits.
- **Multiple-testing residual risk:** BH-FDR controls the false-discovery proportion under its assumptions
  (independence/positive dependence). The 0.23% survivor rate (§6) is consistent with a strict gate; whether the
  gate is *too* strict is unmeasured.
- **Execution-model risk:** costs are stylized (§9); borrow is modelled, not observed.
- **Provenance risk:** this pack's guard is byte-level hash agreement with the accepted artifacts and frozen
  manifest (`SOURCE-GATE.md` Field 14). It does not re-derive the result, so any error inside the accepted
  artifacts would be inherited, not caught — disclosed, not hidden.
- **No live-capital claim:** nothing in this study supports allocating capital, and none is implied (CL-21,
  CL-23).

## 15. Reproducibility

- **Offline, deterministic check:**
  `python3 publication/stat-arb-study/scripts/publication_pack.py check` — stdlib only, zero network, no writes.
  It (1) re-hashes every accepted artifact named in `reproducibility.json`, (2) regenerates `tables/` and
  `figures/` and requires byte-identical output, and (3) resolves every `` `path` (sha256 `…`) `` citation in
  `RESULT-SOURCE-MAP.md`. Any failure exits non-zero (fail-closed).
- **CI:** `.github/workflows/publication-pack.yml` runs the same command on push and pull request.
- **Regeneration only, never re-derivation:** `build` re-emits tables/figures *from* the accepted artifacts;
  it never recomputes cointegration, thresholds, or portfolio results.
- **Provenance chain:** accepted HEAD `d1ef6fd0` → freeze commit `28be77fb` → post-freeze config
  `5768fd10…` → primary artifact `b1197d0f…` (1,833 B) → ledger row (created 2026-09-17T03:44:34Z) →
  this pack. Each link is hash-checked or git-verifiable.
- **Environment independence:** Python 3.11+ stdlib only; no pandas/numpy, no network, no credentials.

## 16. Conclusion

A pre-registered, multiple-testing-controlled ETF pairs screen was evaluated across 32 walk-forward windows.
It admitted four tradable pairs — all pre-2024, all via the fallback rule, none via grid selection — and in the
2025 evaluation period it admitted none, so the strategy held no position and the period yields no performance
measurement. **The null is the finding**, and the design behaved as pre-specified in producing it.

The defensible claims are narrow: the screening pipeline runs, fires when opportunities exist in-sample
(4/1,724), and is silent in 2025. The indefensible claims — an edge, a Sharpe, an alpha, economic
significance, capacity, or a validated strategy — are not made anywhere in this pack, and
`CLAIM-REGISTER.md` records that omission as an explicit, checkable decision (CL-21…CL-23) rather than a gap.

The next decision this study rewards is not "tune until it trades" (prohibited, and it would destroy the
pre-registration) but whether the screen's strictness is appropriate for this asset class — a question that
requires a **new, separately pre-registered** design, not a revision of this one.

---

### Artifacts referenced by this paper

| Artifact | Role |
|---|---|
| `tables/window-level.csv` | all 32 windows with dates, candidate counts, survivor counts, no-trade flags, params |
| `tables/bucket-summary.csv` | per-bucket counts and totals (§6, §10.5) |
| `tables/artifact-hashes.csv` | resolved hash + byte size of every source artifact at build time |
| `figures/selection-funnel.svg` | candidate tests vs selected-pair vs no-trade windows, per bucket |
| `RESULT-SOURCE-MAP.md` | claim → artifact pointer → hash |
| `CLAIM-REGISTER.md` | what may and may not be asserted, with reversal conditions |
| `SOURCE-GATE.md` | 14-field evidence contract and gate provenance |
| `reproducibility.json` | machine-readable manifest of source artifacts and generation steps |
