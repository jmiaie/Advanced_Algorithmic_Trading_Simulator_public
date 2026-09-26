# CLAIMS-AND-LIMITS.md — `stat-arb-study`

Every claim this bundle is permitted to make, classified, with its evidence and its *reversal condition*
(what observation would falsify or force re-wording).

Classes:
- **SUPPORTED** — directly read from an accepted artifact/config field.
- **SUPPORTED-LIMITED** — supported but only partially evidenced, or with a disclosed ceiling.
- **NULL-RESULT** — a reported absence of effect/opportunity; stated as a finding, not a gap.
- **INTERPRETATION** — a reading of the evidence; carries an explicit reversal condition.
- **NOT-CLAIMED** — deliberately not asserted in this bundle (and why).

| ID | Claim as stated in the bundle | Class | Evidence | Reversal condition |
|----|------------------------------|-------|----------|--------------------|
| CL-01 | The 2025 bucket contains 3 evaluation windows (window_index 30/31/32). | SUPPORTED | `RESULT-SOURCE-MAP.md` C-01, C-09 | Artifact re-hash differing, or a window set ≠ {30,31,32} |
| CL-02 | No qualifying pair was found in any 2025 window; `selected_pairs` is empty. | **NULL-RESULT** | C-02, C-07 | Any 2025 window with `n_fdr_survivors > 0` or a non-null `selected_pair` |
| CL-03 | All 3 2025 windows are `no_trade = true` — the strategy took no 2025 position. | SUPPORTED | C-03 | Any window with `no_trade = false` |
| CL-04 | Each 2025 window tested 55 candidate pairs (165 total); 0 survived BH-FDR. | SUPPORTED | C-04, C-05 | Different `n_candidate_tests`, or any survivor |
| CL-05 | No fallback parameters were applied in 2025 (`n_fallback_windows = 0`). | SUPPORTED | C-06 | Non-zero `n_fallback_windows`, or any `used_fallback = true` |
| CL-06 | No Sharpe / drawdown / turnover / return figure exists for 2025, and none is fabricated. | SUPPORTED (structural absence) | C-11 | The appearance of any such metric in the artifact |
| CL-07 | Pre-2025 DEV bucket: 25 windows, 4 qualifying, 21 no-trade, 4 fallback-applied; **0 grid-selected**. | SUPPORTED | C-12, C-13 | Different counts; or a non-zero `n_grid_selected_windows_total` |
| CL-08 | The 4 DEV traded windows used the pre-specified fallback parameters (entry_z 2.0, exit_abs_z 0.5, trailing_z_window 40, kalman_process_variance 1.0e-4) on pairs XLP/XLU (×3) and XLU/XLV. | SUPPORTED | C-14, C-15 | Different params/pairs, or any non-fallback selection |
| CL-09 | The 2024-validation bucket (3 windows) and the boundary bucket (1 window) are entirely no-trade. | SUPPORTED | C-17, C-18 | Any traded window in either bucket |
| CL-10 | Across all 4 buckets: 32 windows, 1,724 candidate tests, 4 FDR survivors, 4 windows with a selected pair — all of them pre-2025. | SUPPORTED | C-19 (generator-derived, `tables/bucket-summary.csv`) | Any arithmetic disagreement with the artifacts |
| CL-11 | The v4 boundary and 2024-validation artifacts are byte-identical to the v3 artifacts; the DEV artifact is not. | SUPPORTED | C-20 (file hashes: `e311a0e2…`, `d59c9747…` shared; `c7ca93a6…` ≠ `52a45f3b…`) | Hash equality/inequality differing |
| CL-12 | Validation trades (if any) would not have been measurable in 2025: the bucket records no risk or return observation. | SUPPORTED | C-11 | Any 2025 return/risk field appearing |
| CL-13 | The "no qualifying pair" outcome is the result of the pre-specified formation-time cointegration + BH-FDR screen, not a crash or an error path: the same mechanism produced 4 qualifying windows in the earlier DEV bucket. | INTERPRETATION | C-07, C-12, C-13, C-15 (mechanism demonstrably fires); C-34 (single run, exit 0, ledger row written) | Evidence that the 2025 run aborted, silently skipped the screen, or used a different universe/grid/cost set |
| CL-14 | The GROSS/BASE/STRESS cost scenarios are stylized assumptions, not measured historical execution costs. | SUPPORTED-LIMITED | C-25, C-26 (config `costs.notes`) | Discovery of a measured-cost source backing them |
| CL-15 | The pre-specified Static-OLS vs sequential-Kalman comparison is only partially evidenced: `validation_ols_sharpe` is null in all 32 windows, `validation_kalman_sharpe` populated in 3 DEV windows only. | SUPPORTED-LIMITED | C-31, C-32, C-33 | Additional populated fields appearing |
| CL-16 | The 2025 evaluation ran once, immediately after the freeze, with no code or parameter change in between. | SUPPORTED | C-34 (freeze commit body; evidence commit body; ledger created 03:44:34Z vs commit 03:48:08Z) | Evidence of a second run, or of an interposed change |
| CL-17 | One boundary-straddling window was excluded from the 2025 bucket by design and surfaced via the runner warning rather than silently dropped. | SUPPORTED | C-35 (evidence commit body); C-01 (3 windows) | Absence of the warning mechanism on a re-run, or a different window count |
| CL-18 | The binding 2025 label is FINAL 2025 WALK-FORWARD EVALUATION; 5/19 symbols overlap the executed FDM (factor-model study) universe; design-origin chronology is unresolved. | SUPPORTED-LIMITED | C-36, C-37; holdout audit | Resolution of the chronology question (would strengthen, not weaken, the disclosure) |
| CL-19 | `pair_persistence` / `hedge_variation` diagnostics are absent from these artifacts. | SUPPORTED (structural absence) | C-39 | Those keys appearing |
| CL-20 | Superseded lineage is preserved as files, labels and ledger rows: the v1 result artifacts (3 files) and the v3 result artifacts (3 JSON + window-boundaries CSV) remain in `results/historical_oos/` under their original labels; the v2 lineage is **config-only** (v2 produced no result artifacts before invalidation) plus one `statarb_hist_etf_wf_v2_INVALIDATED` ledger row; six explicit `…SUPERSEDED_*` ledger rows remain. No superseded artifact, label or ledger row was altered. **DISCLOSURE (corrected 2026-09-18): the claim as first written — that the superseded lineage is *unedited*, recorded hashes included — is falsified.** The `artifact_sha256` values recorded on the three v1 ledger rows resolve to no artifact present in this repository or its history (every blob scanned, dangling included); the v1 artifacts on disk are the slimmed forms — dev_formation 2,496 B `29b6effab77d5440eeb57e33d59e45793105c60547f3b137f9aa0e4584f3d5a7`, val_2024 2,163 B `e3396ba00a11f4fddbaaaccbc0221e2830e4093914b28018edfabb23db3c8a14`, holdout_2025 2,156 B `4fd88a8204ecb4b92cc5017764fcaf0e159b20b202d971b9edc85d4128d6f271` — and the recorded values are unverifiable. Three corrective `…SUPERSEDED_UNVERIFIABLE_ARTIFACT_HASH` ledger rows record this. No ledger cell was rewritten (the ledger is append-only) and no artifact byte was altered. | SUPPORTED-LIMITED (divergence disclosed) | C-20, C-38, C-40 | Any superseded file, label or ledger row differing from the accepted record |
| CL-21 | The strategy is *not* claimed to have a validated edge, an alpha, a Sharpe, an economic benefit, or capacity — no such quantity is estimated anywhere in this bundle. | **NOT-CLAIMED** | TECHNICAL-PAPER.md §13 (limitations) | A future authorized study producing those measurements |
| CL-22 | The 2025 result is *not* claimed to imply the method is unprofitable in general, nor that the design generalizes to other universes, periods or asset classes. | **NOT-CLAIMED** | TECHNICAL-PAPER.md §13 items 1–4 | A pre-registered, non-overlapping multi-universe study |
| CL-23 | Nothing in this bundle is claimed to constitute investment advice, a solicitation, or a performance representation. | **NOT-CLAIMED** | Write-up scope (reporting only) | n/a — scope boundary |

## Notes
1. **No claim in this list depends on a number that is absent from `RESULT-SOURCE-MAP.md`.**
2. **No claim restates a provenance identifier as an original finding.** Commit and hash identifiers are
   used only as provenance.
3. **CL-13 is the only INTERPRETATION**, and it is deliberately narrow: it asserts *how the null arose*
   (pre-specified screen, working mechanism), not that the null was inevitable, profitable-adjacent, or
   favourable. If a reader prefers the strictly descriptive wording, deleting CL-13 changes no other claim.
4. **The bundle's headline is the null itself** (CL-02 + CL-03 + CL-06). This is the accepted pre-registered result and is
   reported as such — not softened, not padded, and not reframed as a data problem.
