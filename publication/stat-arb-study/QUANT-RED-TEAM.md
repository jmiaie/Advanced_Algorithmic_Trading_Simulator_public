# QUANT-RED-TEAM.md — D10-B `stat-arb-study`

**Role:** adversarial quantitative review of this publication pack, performed by the pack author and then
re-checked against the artifacts. Scope is the D10 question: *does the pack report the accepted evidence
faithfully, with no invented or inflated quantity, and are its inferential claims proportional to what the
evidence actually supports?*

**Constraint honoured:** no re-running, re-tuning, or re-deriving of D9-B results. Every check below reads
accepted artifacts, the frozen config, or the git record.

## Summary

| Severity | Open | Closed-by-disclosure | Notes |
|---|---:|---:|---|
| **P0** (would block review) | 0 | — | — |
| **P1** (must be disclosed before review) | 0 | 2 | Q-1, Q-2 |
| **P2** (reviewer-facing notes) | 2 | 3 | Q-3 … Q-7 |

**Verdict: PASS for the publication layer.** The pack adds no quantity that is not in the accepted artifacts,
and every inferential statement carries a reversal condition. The limitations below are properties of the
*accepted evidence*, disclosed rather than papered over.

---

## Q-1 — The pre-specified OLS-vs-Kalman comparison is almost entirely unpopulated (P1 → closed by disclosure)

**Finding.** `validation_ols_sharpe` is `null` in **all 32** windows. `validation_kalman_sharpe` is populated in
**3** windows only (all development: −0.1499, +0.8398, −2.1175). No 2025 window carries either.

**Why it matters.** A reader who skims the design would expect a hedge-method comparison. The evidence for it
essentially does not exist, and three observations from a bucket that also generated all four traded windows are
far too thin to support any hedge-method conclusion.

**Attacked how.** Attempted to construct a hedge-method claim from the three populated values (e.g. "Kalman
dominates"); rejected — 1 of 3 is positive, the OLS side is empty, and the sample is 3.

**Disposition.** Disclosed in the paper (§8) and register (CL-15), with an explicit statement that no hedge-method
claim is made. **No claim in the pack depends on these three numbers.** Residual risk: a reader may still
over-read them; mitigated by §8's "report what we have" wording and §12 item 4.

## Q-2 — The primary selection objective was never exercised (P1 → closed by disclosure)

**Finding.** `n_grid_selected_windows_total = 0` across the entire study, while the four traded development
windows all used the insufficient-trades fallback. Combined with the selection objective's
`min_trade_count: 10`, this means the validation block never produced enough trades to permit a grid selection.

**Why it matters.** The headline methodology (validation-selected thresholds, ties broken by drawdown then
turnover) is described in the design but **never actually selected a threshold in any window of the study**. A
reader could reasonably assume the tuned path was used. It was not.

**Attacked how.** Tried to find a window where grid selection occurred — none exists in any bucket
(`n_grid_selected_windows_total = 0` in both the artifacts and the freeze record).

**Disposition.** Stated as a finding in the paper (§12 item 2, titled "Null (design-level)") and in the register
(CL-07, CL-13). This is arguably the most interesting structural fact in the pack and is reported prominently
rather than buried.

## Q-3 — The 2025 evaluation does not cover the full calendar year (P2, open)

**Finding.** The three 2025 windows cover **2025-01-08 → 2025-10-09** (C-09). November–December 2025 are not
covered, and one boundary-straddling window was excluded from the bucket by design (C-35), consistent with the
frozen 63-session geometry.

**Why it matters.** "2025 evaluation" could be misread as a full-year statement. It is a three-window,
~nine-month walk-forward segment under the frozen geometry.

**Disposition.** Disclosed here and in `CLAIM-REGISTER.md`; the paper states the exact window dates in a table
(§10.4) rather than characterising the period as a whole year. **No language anywhere in the pack claims full-year
coverage.** Left open (not "fixed") because changing the window set is prohibited in this lane.

## Q-4 — Rolling re-selection inside the evaluation bucket (P2, open)

**Finding.** Window 31's validation block spans 2024-10-08 → 2025-04-09 and window 32's spans
2025-01-08 → 2025-07-11 — i.e. earlier 2025 sessions participate in selection for later 2025 test windows.

**Verified safe on the one thing that would be fatal:** `validation_end < test_start` in all three windows
(C-10), so no test-window information enters selection. But the bucket is a *rolling re-selection path*, not one
frozen projection evaluated over nine months.

**Disposition.** Disclosed in the paper (§13 item 9) and `SOURCE-GATE.md` Field 13 item 9.

## Q-5 — Missing diagnostics (P2 → closed by disclosure)

`pair_persistence` and `hedge_variation` keys exist in **no** artifact (C-39). The gate's "where present" phrasing
therefore resolves to *not present*, and the pack makes no persistence or hedge-stability claim, including for the
four traded development windows. Disclosed in §12 item 5 and `SOURCE-GATE.md` Field 13 item 6.

## Q-6 — Unfinished uncertainty quantification (P2 → closed by disclosure)

No block-bootstrap uncertainty over the selection stage, no factor attribution, no capacity/liquidity analysis
(gate Field 14; `SOURCE-GATE.md` Field 13 item 4). **No uncertainty estimate, confidence interval, p-value, or
capacity figure appears anywhere in this pack** — verified by the CLAIM red-team's vocabulary sweep.

## Q-7 — Statistical scale of the traded sample (P2, open)

The four traded windows are all development-bucket, all fallback-parameterised, and — because 2025 produced none
— the study contains **no out-of-sample performance observation at all**. Any reader tempted to treat the four
traded windows as "the result" should note they are in-sample-adjacent development evidence under a declared
fallback, which is exactly why the pack reports no performance claim for them beyond counts and parameters.

---

## Checks performed (and their outcomes)

| Check | Method | Result |
|---|---|---|
| No invented metric | Vocab sweep for return/Sharpe/alpha/economic/capacity terms across all pack docs; every hit classified as negation, structural-absence, or a pre-2025 development diagnostic | PASS (see `CLAIM-RED-TEAM.md` sweep table) |
| No 2025 risk figure | Enumerated every key in all 32 windows; required absence of return/risk keys outside `validation_*` | PASS (`claim_crosscheck.py`, `C-11` assertion) |
| Null reported, not softened | Headline is the null in §11/§12; no "no signal found, but…" hedge, no implied favourable reading | PASS |
| Counts internally consistent | 32 windows / 1,724 tests / 4 survivors / 4 traded re-derived independently of the pack generator | PASS |
| 2024/boundary nulls not omitted | Both buckets reported explicitly, not collapsed into the 2025 discussion | PASS |
| Fallback vs grid selection distinguished | `used_fallback` and `n_grid_selected_windows_total` reported per bucket and flagged as a design-level null | PASS |
| Superseded evidence preserved | v1 (3 files) + v3 (3 JSON + CSV) result artifacts present under original labels; v2 config-only (no result artifacts exist); 23 ledger rows incl. 1 `…_v2_INVALIDATED` and 3 explicit `…SUPERSEDED_*` rows | PASS (CL-20, C-40) |
| Frozen-input integrity | v4 boundary/validation artifacts byte-identical to v3; DEV differs (consistent with the defect-fix lineage) | PASS (C-20) |

## What would change this verdict

- An artifact field (e.g. a populated 2025 metric or a non-null `selected_pair`) appearing after this pack's
  hash set — the cross-check would fail closed and the pack would be wrong.
- Evidence that the 2025 run aborted, retried, or ran against a different universe/grid/cost set — CL-13 and
  CL-16 rest on the single-run commit/ledger chronology.
- Any prose in the pack that a reader could read as a performance or edge claim. None was found; the sweep is
  reproducible.
