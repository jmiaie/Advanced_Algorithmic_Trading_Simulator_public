# SOURCE-GATE.md — D10-B `stat-arb-study`

Publication evidence contract for the D10-B publication pack.
Lane: **D10-B (Hai)**. Directive: **#10** (publication layer over accepted D9 evidence only).

## Field 0 — Gate provenance
- **Value:** authoritative Phase 0 materials supplied by the owner:
  `SOURCE-GATE-MATRIX.md` (FINAL — generated 2026-09-17 evening PT by box-local re-verify; method: live
  GitHub re-check of accepted HEADs + reuse of previously raw-fetched artifact hashes where HEAD unchanged;
  read-only; **repo mutations this pass: NONE**) and `SOURCE-GATE-PACKAGE.md` (user-facing consolidated
  Phase 0 gate).
- **Stream B gate result: PASS.** Field 11: PASS / YES. HEAD drift check: **no drift** (PR #7 head =
  accepted SHA `d1ef6fd`).
- **Status:** ACCEPTED (owner-supplied) + MEASURED (each gate value applicable to B independently re-checked
  against repository bytes during this pack build — see `RESULT-SOURCE-MAP.md`).
- Every field below is either *measured in this repository* (`MEASURED`) or an *accepted value handed down in
  the gate* (`ACCEPTED`). **No field is inferred.**

---

## Field 1 — Pack identity
- **Value:** D10-B publication pack; slug `stat-arb-study`; directive 10; lane B; created 2026-09-18.
- **Status:** MEASURED (`reproducibility.json`).
- **Scope:** publication / communication over accepted D9-B evidence only. No retrain, retune, reacquire,
  rerun, or artifact rewrite is permitted in this lane.

## Field 2 — Repository
- **Value:** `jmiaie/Advanced_Algorithmic_Trading_Simulator_public` (public).
- **Status:** ACCEPTED (gate Field 1) + MEASURED (clone `origin`).

## Field 3 — Accepted source PR / authoritative branch
- **Value:** source PR **#7** (`research/historical-oos-study` → `main`), **open, unmerged, not draft**;
  title *"D9-B: historical Stat-Arb walk-forward study (v1 exploratory superseded; v3 invalidated; v4
  conforming, in review)"*.
- **Status:** ACCEPTED (gate Field 2) + MEASURED (live `pulls/7`).
- **Rule for this lane:** PR #7 is the accepted evidence path and is **not** touched, edited, retargeted or
  merged by D10. This pack is published on a **separate** branch (Field 3b).

## Field 3b — Publication branch and base
- **Value:** branch `publication/stat-arb-study`, created **from the exact accepted SHA** — not from `main`.
  Base SHA = `d1ef6fd04ac3f6e48f20f9424a029e19064af619`. Draft PR only. **No merge.**
- **Status:** MEASURED (`git rev-parse` on the branch, `git merge-base` vs base).

## Field 4 — Accepted HEAD
- **Value:** `d1ef6fd04ac3f6e48f20f9424a029e19064af619` — *"D9-B: FINAL 2025 WALK-FORWARD EVALUATION —
  executed once, credible null"* (2026-09-17T03:48:08Z).
- **Status:** ACCEPTED (gate Field 3; live PR #7 head, zero drift) + MEASURED (branch tip equality).

## Field 5 — Freeze commit
- **Value:** `28be77fb018d14098a9079cd7e880f456d308599` — *"FINAL CONFIGURATION FROZEN BEFORE HOLDOUT
  EVALUATION"*, status `pre-registered -> frozen-for-holdout`, and the parent of Field 4. Reviewer code HEAD
  at freeze: `3527634430a8db2dacbdb83e5f2fe9405973451e`.
- **Status:** ACCEPTED (gate Field 7) + MEASURED (parent relationship; freeze commit body).

## Field 6 — Experiment and frozen configuration
- **Value:** experiment `statarb_hist_etf_wf_v4` (authoritative; superseded/invalidated:
  `statarb_hist_oos_v1`, v2 INVALIDATED, v3 invalidated-before-holdout);
  config `configs/experiments/statarb_historical_etf_wf_v4.yaml`
  (sha256 `5768fd10b63c0436f3ff5aa1863374b348cca5bf72ccfb7d1b01a7ae0d95376d`) — **post-freeze config SHA-256**;
  pre-freeze SHA-256 `e8acafc514d87267d4dc3965212d0e8ed7dc2da600f34ff758b00bf6e49db003`.
  The two differ **only** in the `status` field and the added `freeze_record` block (verified from the freeze
  commit body and `freeze_record.pre_freeze_config_sha256`; re-verified by hashing the file bytes here).
- **Status:** ACCEPTED (gate Fields 4, 8) + MEASURED.

## Field 7 — Dataset
- **Value:** `yf_stat_arb_etfs_daily_2015_2025_v1`; 19 symbols; interval `1d`; coverage 2015-01-02 →
  2025-12-31; 52,361 rows; 0 missing rows per symbol (XLRE begins 2015-10-08);
  canonical SHA-256 `a4082398991400c171062ee4ff2b12dba87e84c4f0133462ed9ee2fbb13f9fa8`;
  manifest `data/manifests/yf_stat_arb_etfs_daily_2015_2025_v1.json`
  (sha256 `b2557499091cd87e412365db2a4a32eac7134995bf641e35e5676114f6cc2e30`), `DATA FROZEN`,
  freeze timestamp `2026-09-16T17:28:59Z`.
- **Status:** ACCEPTED (gate Field 6) + MEASURED (manifest bytes hashed; manifest's declared
  `sha256.dataset_canonical` equals the accepted canonical and the config's
  `freeze_record.dataset_canonical_sha256`).
- **Limit (disclosed):** the raw per-symbol CSVs are **not committed** to this repository (`data/` holds
  manifests only), so the canonical hash cannot be re-derived from repository bytes. It is an accepted,
  disclosure-limited input — not re-computed, not fabricated. **Consequence for readers:** this pack verifies
  *provenance to the frozen manifest*, not raw-file integrity.

## Field 8 — Primary 2025 artifact
- **Value:** `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json`
  (sha256 `b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66`), 2,359 bytes, bucket
  `holdout_2025`. Ledger row `statarb_hist_etf_wf_v4_holdout_2025` (created 2026-09-17T03:44:34Z) records the
  same hash and the same key metrics.
- **Status:** ACCEPTED (gate Field 10) + MEASURED (bytes re-hashed here; ledger row agrees).

## Field 9 — Evidence class
- **Value:** accepted **D9 empirical result** — frozen-design walk-forward evaluation, program-audited at the
  four-stream level. **Not** externally peer-reviewed material, and **not** re-derived by this pack.
- **Status:** ACCEPTED.
- **Consequence:** this pack may report, explain and cite the accepted result; it may not extend, re-tune,
  re-run, or "improve" it. No claim of external validation or of generalizability is made anywhere in the pack.

## Field 10 — Label discipline (2025)
- **Value:** binding label **FINAL 2025 WALK-FORWARD EVALUATION** (gate Field 7). **Must not** be called
  untouched / pristine / fresh / independent final holdout.
- **Status:** ACCEPTED (gate) + MEASURED (pre-committed in `research/holdout-audit.md`
  (sha256 `d1f15db859957ff88a186474892e2cde2e63be6dc0b35bb5c636321dbcf4bf24`) and in the freeze + evidence
  commit bodies).
- **Reason:** 5 of 19 symbols (SPY, QQQ, IWM, TLT, GLD) overlap the already-executed FDM D9-A research
  universe; a superseded exploratory v1 study on a *different* (equity-pair) universe also ran its own 2025
  window. Both exposures are disclosed (Field 13).

## Field 11 — Program audit citation (authoritative, as handed down)
> "Independent review status: Directive #9 Final Four-Stream Independent Program Audit (Grokbot,
> 2026-09-17 evening PT) — PROGRAM SIGN-OFF: YES; P0=0; P1=0; HISTORICAL EMPIRICAL VALIDATION
> COMPLETE / ACCEPTED. Streams A@5d01cf8, B@d1ef6fd, C@db9cf44, D@9184eff each ACCEPT. Hub Issue #3
> remains stale by design (no hub update yet). Field 11 = PASS / YES."

- **Status:** ACCEPTED — this is the gate's own Field 11 text. The hub `quant-research-portfolio` Issue #3 is
  stale **by design**; it is neither awaited nor modified by this lane.

## Field 12 — Result of record (2025)
- **Value:** 3 pure-2025 evaluation windows (window_index 30/31/32; tests 2025-01-08→2025-04-09,
  2025-04-10→2025-07-11, 2025-07-14→2025-10-09). **All three `no_trade = true`.** 0 qualifying pairs,
  0 trades, 55 candidate tests per window (165 total), **0 BH-FDR survivors**. `selected_pairs` and
  `param_distribution` empty; no fallback parameters used. **No Sharpe, drawdown, turnover or return figure
  exists for 2025 and none is fabricated.**
- **Status:** MEASURED — `RESULT-SOURCE-MAP.md` claims C-01…C-11 (artifact pointers) and C-34…C-35 (provenance).

## Field 13 — Limitations and disclosures carried into the paper
1. **Cross-program exposure:** 5/19 symbols shared with the already-executed FDM D9-A 2025 evaluation; the
   design-origin chronology relative to FDM results is **unresolved** (gate Field 14). This is why the label is
   FINAL 2025 WALK-FORWARD EVALUATION and not an untouched holdout.
2. **Survivorship / convenience bias:** the ETF universe is static and selected with present-day knowledge.
3. **No 2025 risk measurement:** the 2025 bucket contains no return, volatility, Sharpe, drawdown or turnover
   observation — the design produced no position. Absence of measurement is not evidence of stability.
4. **Unfinished per the gate:** factor attribution, capacity/liquidity analysis, and block-bootstrap
   uncertainty are **unfinished** and are therefore **not claimed** here.
5. **Partial OLS-vs-Kalman evidence:** `validation_ols_sharpe` is unpopulated in all 32 windows and
   `validation_kalman_sharpe` in 29 of 32 (3 DEV windows populated). The pre-specified comparison is therefore
   only partially evidenced (see `RESULT-SOURCE-MAP.md` C-31…C-33).
6. **Missing diagnostics:** `pair_persistence` / `hedge_variation` fields do not exist in these artifacts
   (C-39) — the gate's earlier "where present" wording resolves to *not present*.
7. **Stylized costs:** the GROSS/BASE/STRESS scenarios are assumptions, not measured historical execution
   costs (config `costs.notes`); no market-impact calibration was performed.
8. **Raw data not in-repo:** See Field 7 limit.
9. **Rolling re-selection inside the 2025 bucket:** trailing formation/validation sub-windows include earlier
   2025 sessions. No look-ahead — `validation_end < test_start` in every window (C-10) — but the 2025 bucket is
   a rolling re-selection path, not a single frozen projection.
10. **Superseded lineage preserved, not edited:** the v1 result artifacts (3 files) and v3 result artifacts
    (3 JSON + boundaries CSV) remain under their original labels in `results/historical_oos/`; the v2 lineage is
    **config-only** (no v2 result artifacts were produced before invalidation) plus one
    `statarb_hist_etf_wf_v2_INVALIDATED` ledger row; the ledger's 23 rows retain four v1 rows and three explicit
    `…SUPERSEDED_*` rows. All are cited as superseded. **The 2025 label on the superseded v1 study was not
    changed.**
11. **Sources of record are prose + artifact, not a paper:** the D9-B prose record is
    `research/statistical-arbitrage-validation.md`
    (sha256 `dde09a88c6c69b6027b18ed316156686dcb2555fd8b28804dec08e1633ce1370`) and
    `research/holdout-audit.md` (sha256 `d1f15db859957ff88a186474892e2cde2e63be6dc0b35bb5c636321dbcf4bf24`).
    This pack cites them; it does not overwrite or reinterpret them.
- **Status:** MEASURED (each item checkable against the artifact/config/commit named in `RESULT-SOURCE-MAP.md`).

## Field 14 — Verification contract (this pack)
- **Command:** `python3 publication/stat-arb-study/scripts/publication_pack.py check` — offline, deterministic,
  stdlib only, no network, no writes.
- **Checks:** (1) every source-artifact hash in `reproducibility.json` matches disk; (2) `tables/` and
  `figures/` are byte-identical to a deterministic rebuild from the accepted artifacts; (3) every
  `` `path` (sha256 `…`) `` citation in `RESULT-SOURCE-MAP.md` resolves to a real file with a matching hash.
- **Fail-closed:** any missing file, hash mismatch, drift, or unfound citation exits non-zero.
- **Publication CI:** `.github/workflows/publication-pack.yml` runs the same check on push and PR.
- **What this does NOT verify:** the empirical result itself (accepted, not re-derived), raw input bytes
  (Field 7 limit), or any claim the gate marked unfinished (Field 13 item 4).
