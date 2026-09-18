# CLAIM-RED-TEAM.md — D10-B `stat-arb-study`

**Role:** adversarial review of *what the pack claims*. Not "is the study good?" but "does any sentence assert
more than the accepted evidence supports, and is any prohibited claim smuggled in anyway?"

**Method:** every claim in `CLAIM-REGISTER.md` attacked individually; then a whole-pack vocabulary sweep for
performance/edge vocabulary; then an absence audit of what the pack does *not* say.

---

## 1. Claim-by-claim attack

| Claim | Attack attempted | Outcome |
|---|---|---|
| CL-01 (3 windows) | Recount windows directly from JSON, ignoring the artifact's own `n_windows` | HOLDS — 3 windows, indices 30/31/32 |
| CL-02 (no qualifying pair) | Search for any non-null `selected_pair` or survivor > 0 in 2025 | HOLDS — none; empty `selected_pairs` |
| CL-03 (all no_trade) | Search for any `no_trade: false` in 2025 | HOLDS — all three true |
| CL-04 (55 tests ×3, 0 survivors) | Recount per window | HOLDS — 55/55/55, 0/0/0 |
| CL-05 (no fallback in 2025) | Check `used_fallback` per window and the bucket counter | HOLDS — false; `n_fallback_windows = 0` |
| CL-06 (no 2025 performance metric) | Enumerate **every** key of every 2025 window and bucket, looking for a return/risk field | HOLDS — only `validation_*` metric keys exist, and both are null |
| CL-07 (DEV 25/4/21/4, 0 grid-selected) | Recount DEV; cross-check `n_grid_selected_windows_total` in config freeze record | HOLDS — counts agree, grid total is 0 in both |
| CL-08 (fallback params, pairs XLP/XLU ×3 + XLU/XLV) | Read `frozen_params` of each traded window and the pair list | HOLDS — params identical to the config fallback block; pairs as stated |
| CL-09 (2024 + boundary all no-trade) | Recount both buckets | HOLDS |
| CL-10 (32 / 1,724 / 4 / 4 totals) | Re-derive with independent code (`claim_crosscheck.py`) | HOLDS |
| CL-11 (v3≡v4 for boundary/val, ≠ for DEV) | Hash all four files | HOLDS — `e311a0e2…` and `d59c9747…` match; DEV differs |
| CL-12 (no measurable validation performance) | Same as CL-06 | HOLDS |
| CL-13 (null is the screen's result, not a crash) | Tried to falsify: look for an aborted run, a skipped screen, or a different universe/grid | HOLDS on available evidence — 4 DEV windows traded, ledger row written 3m24s before the evidence commit, single invocation |
| CL-14 (costs are assumptions) | Read config `costs.notes` | HOLDS — verbatim |
| CL-15 (partial OLS/Kalman evidence) | Count populated metrics | HOLDS — OLS 0/32, Kalman 3/32 |
| CL-16 (one run, no interposed change) | Compare freeze commit timestamp, freeze commit body, ledger `created_utc`, evidence commit timestamp | HOLDS — 03:42:45Z freeze → 03:44:34Z ledger → 03:48:08Z commit; commit body states no parameter change |
| CL-17 (boundary window excluded by design, warned) | Read evidence commit body; check window count | HOLDS — stated in commit body; bucket holds 1 window |
| CL-18 (label + 5/19 overlap + unresolved chronology) | Read holdout-audit + gate Field 14 | HOLDS — not upgraded to "untouched holdout" anywhere |
| CL-19 (no persistence/hedge diagnostics) | Inspect artifact schema | HOLDS — keys absent |
| CL-20 (superseded evidence preserved) | List v1/v3 result artifacts, the v2 config, and the ledger's superseded rows; hash them; check the v1 label was not upgraded | **FOUND A DEFECT (fixed)** — the first draft of CL-20 said "v1/v2/v3 artifacts remain in place", but no v2 **result artifacts** exist (v2 was invalidated before holdout; only a config and an `…_v2_INVALIDATED` ledger row). CL-20, paper §10.6 and `SOURCE-GATE.md` F13 were re-worded to the measured inventory, and the C-40 assertion now enforces it |
| CL-21/22/23 (NOT-CLAIMED set) | Sweep every doc for the prohibited vocabulary | HOLDS — see §2 |

**Result: all 23 register claims (CL-01…CL-23) hold as published. 22 held as first drafted; 1 (CL-20) was found
over-general, re-worded to the measured inventory, and then held.** No claim required downgrading, and no claim
was dropped. The defect found is recorded
in §1 and in `CITATION-RED-TEAM.md` §4 rather than silently corrected — the register's own reversal conditions are
what surfaced it.

## 2. Whole-pack vocabulary sweep (the "did something slip in?" test)

Terms swept across all nine pack documents: `sharpe, alpha, profitab*, economic, capacity, validat*, outperform,
edge, return*, signal, advice`.

| Term | Hits | Classification of **every** hit | Verdict |
|---|---:|---|---|
| sharpe | 24 | negation ("no Sharpe exists"), structural-absence statement, or reference to the *pre-2025* `validation_*` diagnostics; plus the design's objective name | PASS |
| alpha | 6 | all negations ("not claimed to have… an alpha") | PASS |
| profitab* | 4 | all negations or design statements ("selection is never conditioned on profitability") | PASS |
| economic | 8 | "economically related groups", "economically spurious", or negations of economic significance | PASS |
| capacity | 11 | all negations or "unfinished/not produced" disclosures | PASS |
| validat* | 54 | methodology (validation block/window/bucket), file/claim identifiers, or "no validated edge" negations | PASS |
| outperform | 0 | — | PASS |
| edge | 37 | substring noise (`knowl**edge**`, `acknowl**edge**`) + "no validated edge" negations | PASS |
| return* | 11 | all negations or "return observation does not exist" disclosures | PASS |
| signal | 9 | methodology (signal grid, signal thresholds) or "weak pre-2025 signal" | PASS |
| advice | 1 | explicit disclaimer (CL-23) | PASS |

**No occurrence of any performance vocab term anywhere in the pack constitutes an assertion of results.** The
sweep is reproducible by the reviewer in one command (documented in `D10-STATUS.md`).

## 3. Absence audit — things the pack deliberately does NOT say

| Not said | Why | Where the refusal is recorded |
|---|---|---|
| Any 2025 return/Sharpe/drawdown/turnover/alpha | No position existed; nothing to measure | CL-06, CL-21 |
| "The strategy correctly avoided risk in 2025" | Motive and foresight are not in the artifacts | paper §11 item 3 |
| "The method doesn't work / sentiment-free strategies are dead" | Out of scope; a null in one screen ≠ a general negative | CL-22 |
| "Market-only is proven superior" (no such claim here, but guarded by analogy) | Single study, single universe | CL-22 |
| Any capacity, factor, or market-impact figure | Gate marks these unfinished | CL-21, `SOURCE-GATE.md` Field 13 item 4 |
| "Untouched 2025 holdout" | 5/19 symbol overlap; label pre-committed | CL-18 |
| Hedge-method preference (OLS vs Kalman) | Comparison metrics ~absent | CL-15, paper §8 |
| Live-trading readiness or advice | Scope prohibition | CL-23 |

## 4. Softening test (the null must not be dressed up)

Checked for the standard evasions and found none:
- no "no signal was found, **but** the framework is promising";
- no reframing of no-trade as a *feature* of the strategy ("disciplined", "defensive", "by design") — the paper
  states the screen admitted nothing and the portfolio held nothing;
- no substitution of a friendlier bucket's result for the evaluation period;
- no shift of the answer to a different metric that happens to look better (there is none to shift to);
- no hedging of the headline: the null is the headline in §11 and §12 and is repeated in the abstract,
  conclusion, case study, and this register.

## 5. Register integrity

- Every SUPPORTED / NULL-RESULT / SUPPORTED-LIMITED claim carries a pointer ID resolvable in
  `RESULT-SOURCE-MAP.md`.
- Every claim carries a **reversal condition** capable of returning a different verdict — the register has no
  unfalsifiable entry except the three NOT-CLAIMED scope statements (which are *refusals*, not assertions).
- The single INTERPRETATION (CL-13) is narrowly scoped and explicitly removable without affecting any other
  claim.

## 6. Findings

**0 P0, 0 P1.** CLAIM-RED-TEAM findings: none requiring remediation. Three P2 review notes are recorded in
`QUANT-RED-TEAM.md` (Q-3 window coverage, Q-4 rolling re-selection, Q-7 traded-sample scale); each is disclosed
in-pack and none is a defect in the pack itself.
