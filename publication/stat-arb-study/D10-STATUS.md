# D10-STATUS.md — D10-B `stat-arb-study`

**Lane:** D10-B (Hai) · **Directive:** #10 (publication/communication over accepted D9 evidence only)
**Status: READY FOR INDEPENDENT D10 REVIEW.** No merge, no external publication, no D11.

*(Authoring-phase status — 2026-09-18. Superseded as a statement of current state: the pack is now integrated on `main`. See "Post-review integration status" at the top of this file.)*

*Date:* 2026-09-18 · *Pack:* `publication/stat-arb-study/` · *Every number below is captured from a command run,
not typed by hand.*

---

## Final Directive #10 program sign-off — 2026-09-23

**This section is current. It supersedes every "sign-off pending" statement in this
file**, including the reconciliation-phase lifecycle conclusion and program-state
lines recorded below, which are retained unedited as the historical record.

An independent clean-room review of the live four-stream D10 heads concluded
`DIRECTIVE #10 PUBLICATION PACK SIGN-OFF: YES`, with final counts **P0 = 0, P1 = 0,
P2 = 3, P3 = 14** and this lane's verdict **ACCEPT-WITH-RESIDUALS**. No accepted
dataset, configuration, manifest, result artifact, experiment identity, ledger row,
table, figure, estimate, interval or finding was found to have changed.

D10 publication-pack program sign-off for D10-B is therefore **COMPLETE** as of
**2026-09-23**.

| Final closure record | Value |
| --- | --- |
| Independently signed reconciliation head | `980f660f0a9f20386c7b2591f8580577bec854f6` |
| Current `main` merge head | `c73b61dbeec89955aed31d8cb346256f3532f4a9` |
| Merge tree vs signed head | **Byte-identical — zero changed files.** `git diff --name-only 980f660f0a9f20386c7b2591f8580577bec854f6 c73b61dbeec89955aed31d8cb346256f3532f4a9` returns empty (re-measured 2026-09-23). The integration carried the reviewed tree forward unchanged. |
| Accepted evidence changed by integration | **None.** No rerun, retrain, refit, retune, reacquire, or result replacement was performed. |
| Administrative status | Closure **packaged, not merged**. This section is a status record; it is **not merge authorization**, and it does not decide whether Directive #10 is formally closed. |

### What this sign-off does not assert

It records that the publication pack on `main` is the independently reviewed pack.
**Explicitly refused claims.** This section does **not** assert any of the following
phrases, or their substance: "predictive edge"; "alpha"; "economic significance";
deployment, production or live-trading approval; or prospective/live validation. **No 2025 Sharpe ratio, drawdown or turnover figure exists or is implied**; the study contains no out-of-sample performance observation, so nothing here is a strategy-performance claim.

### Preserved findings and classifications

- **The accepted 2025 null result**: three pure-2025 windows, 55 tests per window,
  **zero FDR survivors, zero qualifying pairs, zero trades**, and no 2025 Sharpe,
  drawdown or turnover.
- The disclosed main-history/provenance gap: PR #8's merge `887ad649` is **not an
  ancestor of `main`**; the publication commits appear in `main`'s linear history.
  Independently reviewed and disclosed — it is not an open P1 blocking sign-off.
- The binding 2025 period label and the development-bucket fallback-parameterisation
  limitation (the pack records no out-of-sample performance observation at all).

### Residual register — preserved, not closed

Sign-off does **not** imply zero remaining maintenance work, and this closure change
does not silently close any residual. The independent review recorded:

**P2 (disclosed, open)**

1. The evaluation bucket covers 2025-01-08 → 2025-10-09, not the full calendar year;
2. rolling re-selection inside the 2025 bucket (causally safe, `validation_end < test_start`);
3. the four traded windows are development-bucket fallback-parameterised — the study
   contains no out-of-sample performance observation at all.
   See §4 of this file for the full register; none of the three is repaired here.

**P3 (shared register, disclosed, open)**

- FDM formation/development explanatory prose and early publication-commit ordering;
- Stat-Arb canonical source-gate presentation and stale superseded-ledger-row count;
- Options stale source-map manifest-hash instruction;
- Sentiment combined-model margin wording and majority-baseline model description;
- the shared reconciliation-baseline table omission (**corrected in this round**) and the
  A/B/C workflow-trigger prose (**corrected in this round**) — the only two shared
  administrative items authorized for correction in this round.

### Supersession wording

The reconciliation-phase conclusions **"INTEGRATED ON MAIN / FINAL D10 PROGRAM
SIGN-OFF PENDING"** and **"Final D10 program sign-off remains PENDING"** are
superseded by this statement:

> **D10 publication-pack program sign-off is complete as of 2026-09-23 at merged
> head `c73b61dbeec89955aed31d8cb346256f3532f4a9`, on the independently signed head `980f660f0a9f20386c7b2591f8580577bec854f6`, with the merge tree
> byte-identical to that signed head and the P2/P3 residuals above preserved and
> undisputed.**

The superseded wording is retained verbatim below as the reconciliation-phase
record rather than rewritten.

---

## Post-review integration status

This publication pack was originally authored and reviewed under a
no-merge / stop-at-independent-review instruction. That language is preserved
below as a historical record of the authoring phase.

The pack has subsequently been integrated into `main`. This integration does
not, by itself, constitute Directive #10 program sign-off.

Current lifecycle status: **SUPERSEDED 2026-09-23 — see "Final Directive #10
program sign-off" at the top of this file.** As recorded at the 2026-09-21
reconciliation, this cell read **"INTEGRATED ON MAIN / FINAL D10 PROGRAM
SIGN-OFF PENDING."** That wording is retained here as the reconciliation-phase
record rather than rewritten.

| Integration record | Value |
| --- | --- |
| Accepted D9 head | `d1ef6fd04ac3f6e48f20f9424a029e19064af619` |
| Cleared publication head / accepted publication ancestor | `77c8fbd83ea87804bdde90139c4415f88f649a9d` (PR #8 head; later cleared line at `main` is `372d5571`) |
| `main` head at the reconciliation baseline (frozen 2026-09-21; a reference point, not a permanently-current value — verify with `git ls-remote <repo> refs/heads/main`)  `372d5571ca72913ed0c69c53337473ffe63f8c94` — restored 2026-09-23; this cell was left blank at reconciliation. |
| Integration path | Pack authored on `publication/stat-arb-study`; PR **#8** merged it into `research/historical-oos-study` (merge `887ad649`, 2026-09-18T13:22:48Z). `main` carries the pack as linear descendants of the accepted D9-B head: `d1ef6fd` → `b35862e` → `77c8fbd` → `63d0434` → `1758de4` → `372d557`. |
| Relevant pull requests | #8 (pack → study branch, merged); #7 remains open, unmerged, not draft (PR hygiene inventory) |
| Exact-head CI evidence | At exact `main` head `372d5571`: `ci` run `35401862776` (success) — https://github.com/jmiaie/Advanced_Algorithmic_Trading_Simulator_public/actions/runs/35401862776 ; `ci` run `35391301575` (success); `publication-pack` run `35391301475` (success) — https://github.com/jmiaie/Advanced_Algorithmic_Trading_Simulator_public/actions/runs/35391301475 . |
| Exact-head CI evidence — remediation branch | `reconcile/d10-b-lifecycle`. **Corrected 2026-09-23 — "Both workflows run on every push to this branch" was true only of one of the two workflows.** Measured against the workflow definitions at `main` (`c73b61db`) and against the runs actually recorded on this branch: the repository `ci` workflow triggers on `push` for **all** branches (`branches: ["**"]`) and on `pull_request`; the `publication-pack` workflow triggers on `push` only for `publication/stat-arb-study`, and on `pull_request` for changes under `publication/stat-arb-study/**`. The branch carries **6 runs**: `ci` ×2 `push` and ×2 `pull_request`, `publication-pack` ×2 `pull_request` and **zero `push`**. Repository CI therefore does verify a push to a reconciliation branch, but the pack workflow does not — pack verification of a reconciliation head occurs through the **PR event**. The two runs listed immediately below are `pull_request` runs. Most recent completed runs, at commit `c93e1aebe3` — the commit immediately preceding this edit: `ci` run `35660173452` (success) — https://github.com/jmiaie/Advanced_Algorithmic_Trading_Simulator_public/actions/runs/35660173452 ; `publication-pack` run `35660173465` (success) — https://github.com/jmiaie/Advanced_Algorithmic_Trading_Simulator_public/actions/runs/35660173465 . |
| Diff from accepted D9 is publication-only | **No — publication pack plus one append-only ledger file.** `git diff --name-status d1ef6fd 372d557` yields the pack, `.github/workflows/publication-pack.yml`, and `research/experiment-ledger.csv`. The ledger change is **3 inserted rows / 0 deletions / 0 modifications** (`3 +++`), all corrective provenance rows named `*_SUPERSEDED_UNVERIFIABLE_ARTIFACT_HASH`, recording that three previously recorded `artifact_sha256` values resolve to no blob in the object store. The superseded rows are left unedited; no accepted result artifact was changed. |
| Disclosed integration nuance | **No pull request records a merge of this pack into `main`.** PR #8's merge commit `887ad649` is not an ancestor of `main`; the pack commits appear in `main`'s linear history directly. Recorded as a provenance gap for the independent re-audit; no content difference is implied. |

*(Superseded 2026-09-23 for D10 only: D10 publication-pack program sign-off is now
complete — see "Final Directive #10 program sign-off" at the top of this
file. The D9 and D11–D13 wording below is unchanged and remains current.)*

**Current program state.** D9: COMPLETE / ACCEPTED. D10: TECHNICALLY
INTEGRATED / FORMAL SIGN-OFF PENDING. D11: PARTIALLY STARTED THROUGH THE
PUBLIC HUB / NOT FORMALLY ACTIVATED. D12: DRAFTED / BLOCKED BY D11 HIRING
EVIDENCE. D13: DRAFTED / NOT YET JUSTIFIED.

**Superseded 2026-09-23 — an authoritative D10 publication-pack program
sign-off has since been issued; see "Final Directive #10 program sign-off" at
the top of this file. The paragraph below is the reconciliation-phase record,
retained unedited.** No authoritative
`DIRECTIVE #10 PUBLICATION PACK SIGN-OFF: YES` has been issued for this pack.
A D9 program sign-off is not a D10 program sign-off. This section records
integration state only: it is not a sign-off, and it does not strengthen,
weaken, or restate any finding, number, or claim in the pack.

### How to read the rest of this directory

Every "no merge", "no pull request merged", "draft PR only", "not on `main`",
"not from `main`", "no external publication", and "READY FOR INDEPENDENT
(D10) REVIEW" statement preserved below, or elsewhere in this directory, is
**authoring-phase language** kept deliberately as the contemporaneous record
(append-only history; the historical record is not rewritten). Where such a
statement could be read as describing the *current* lifecycle state, this
section supersedes it; the statement itself is left unedited. The
machine-readable `reproducibility.json` field `merge` is likewise left
byte-unchanged on purpose, so the pack's own hash and regeneration gates stay
valid at the recorded tip.

*Repository visibility note:* the host repository is public, so this pack is
world-readable on `main`. No PyPI/npm release, website deployment, or other
external-service publication was performed.

---

*Post-review integration section added 2026-09-21 as documentation-only
reconciliation. No empirical artifact, configuration, dataset manifest,
experiment identity, ledger row, number, or finding was changed; no
rerun, retune, or reacquisition was performed.*

## 1. Publication branch

| Item | Value |
|---|---|
| Repository | `jmiaie/Advanced_Algorithmic_Trading_Simulator_public` |
| Publication branch | `publication/stat-arb-study` |
| Base SHA (branch created from — **not** from `main`) | `d1ef6fd04ac3f6e48f20f9424a029e19064af619` |
| Branch HEAD | exact HEAD SHA, CI run ID and `head_sha` are reported in the return package and PR checks |
| Draft PR | opened as **draft**; number/URL in the return package |
| Merge performed | **NO** |
| External publication | **NONE** |

Source PR **#7** (`research/historical-oos-study`) remains open, unmerged and **untouched**; the pack is
published on a separate branch from the exact accepted SHA, not from `main`.

*(Authoring-phase statement. “Not from `main`” no longer describes the current state: `main` now carries the pack — see "Post-review integration status".)*

## 2. Deliverables

| # | Deliverable | Path | Lines |
|---:|---|---|---:|
| 1 | Source gate (14 fields; field 11 = D9 sign-off authority) | `SOURCE-GATE.md` | 163 |
| 2 | Technical paper (17 sections + method-fidelity item 12) | `TECHNICAL-PAPER.md` | 336 |
| 3 | Result → source map (40 claim pointers C-01…C-40) | `RESULT-SOURCE-MAP.md` | 91 |
| 4 | Machine-readable reproducibility manifest | `reproducibility.json` | 84 |
| 5 | Case study | `CASE-STUDY.md` | 122 |
| 6 | Claim register (23 claims, CL-01…CL-23, + reversal conditions) | `CLAIM-REGISTER.md` | 47 |
| 7 | Quant red-team | `QUANT-RED-TEAM.md` | 123 |
| 8 | Claim red-team | `CLAIM-RED-TEAM.md` | 102 |
| 9 | Citation red-team | `CITATION-RED-TEAM.md` | 92 |
| 10 | This status file | `D10-STATUS.md` | 134 |
| 11 | Deterministic generator/verifier + hash-integrity sweep | `scripts/publication_pack.py` | 384 |
| 12 | Independent claim cross-check (shares no code with the above) | `scripts/claim_crosscheck.py` | 363 |
| 13 | Offline CI | `.github/workflows/publication-pack.yml` | 22 |

**Tables** (`tables/`): `window-level.csv` (all 32 windows), `bucket-summary.csv` (per-bucket counts),
`artifact-hashes.csv` (resolved source hashes + sizes).
**Figures** (`figures/`): `selection-funnel.svg` (candidate tests vs selected-pair vs no-trade windows per bucket).
Every table and figure is regenerated deterministically from the accepted artifacts; provenance per item is in
`RESULT-SOURCE-MAP.md`. No hand-entered numbers exist in any table or figure.

## 3. Checks (offline, fail-closed, zero network)

```
$ python3 publication/stat-arb-study/scripts/publication_pack.py check
publication-pack check OK: 13 artifact hashes, 4 generated files, 11 map citations verified

$ python3 publication/stat-arb-study/scripts/publication_pack.py hashcheck
hash-integrity check OK: 14 published sha256 values verified (62 occurrences), 3 full git SHAs printed, 15 abbreviations resolved

$ python3 publication/stat-arb-study/scripts/claim_crosscheck.py
claim cross-check OK: 92 independent assertions passed

$ ruff check .                     # the same gate the repository CI runs
All checks passed!

$ mypy .                           # publication/ is outside the configured mypy scope
Success: no issues found in 39 source files
```

CI: `.github/workflows/publication-pack.yml` runs the first three gates on every push to
`publication/stat-arb-study` and on any PR touching the pack (`fetch-depth: 0` so `hashcheck` can re-measure the
config blob at the freeze commit's parent). The workflow is deliberately not self-referential: the **run ID,
`head_sha` and per-job conclusions are reported against the exact branch HEAD in the D10 return package and the
PR checks page**, so they can never disagree with the commit they describe.

## 4. Findings

| Severity | Open | Summary |
|---|---:|---|
| **P0** | **0** | none |
| **P1** | **0** | none open; two items (Q-1 partial OLS/Kalman evidence, Q-2 primary selection objective never exercised) are **closed by disclosure** in the paper, register and red-team |
| **P2** | **3** | Q-3 evaluation bucket covers 2025-01-08 → 2025-10-09, not the full calendar year; Q-4 rolling re-selection inside the 2025 bucket (causally safe, `validation_end < test_start`); Q-7 the four traded windows are development-bucket fallback-parameterised — the study contains no out-of-sample performance observation at all |

Also disclosed: two non-hash-verifiable citation classes (owner-supplied gate documents; uncommitted raw price
CSVs) — `CITATION-RED-TEAM.md` §4.

## 5. Confirmations required by Directive #10

| # | Requirement | Status | Evidence |
|---:|---|---|---|
| 23 | No empirical rerun | **CONFIRMED** | no execution command was run against the study pipeline; only read-only hashing, table regeneration and assertions |
| 24 | Accepted D9 artifacts unchanged | **CONFIRMED** | all artifact hashes match the accepted values; `git status` shows no modification under `results/` |
| 25 | No merge | **CONFIRMED** | draft PR only; no merge command issued; PR #7 untouched |
| 26 | No external publication | **CONFIRMED** | nothing pushed outside the repository branch; no external service used |
| 27 | No central hub / Issue #3 edits | **CONFIRMED** | no write of any kind to `quant-research-portfolio` or Issue #3 |
| 28 | (stream D) stale PR #4 untouched | n/a for B | — |
| 29 | Claude's lanes A/C untouched | **CONFIRMED** | no access to, or write into, `financial-dynamics-model` or `options-volatility-risk-lab` |
| — | No retrain / retune / reacquire / rerun 2025 / label change / hypothesis change / universe change / target change | **CONFIRMED** | pack is read-only over accepted evidence; superseded labels retained |
| — | No invented metrics / significance / Sharpe / alpha / economic significance | **CONFIRMED** | `CLAIM-RED-TEAM.md` vocabulary sweep; `claim_crosscheck.py` assertions |

## 6. Integrity proofs

**A. Accepted D9 artifact integrity.** `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json`
re-hashed on disk = `b1197d0fb5280b54849270cfadd50bd450458522bbf6e482a1870cec723b6d66` — exact match to the accepted value. Enforced on every run by
`publication_pack.py check` (13 artifact hashes re-measured).

**B. Exact-head CI.** Run ID, `head_sha` and per-job conclusions are returned against the exact branch HEAD in
the D10 return package, with the PR checks page as the independent record. No CI claim is made in this file that
cannot be checked against the commit it names.

**C. No D9 empirical mutation.** The branch adds two paths only — `publication/` and
`.github/workflows/publication-pack.yml` — and modifies no tracked file: `git status --short` shows only those
untracked additions, and the diff against the accepted SHA is empty for `results/`, `configs/`, `data/`, `src/`
and `research/`. No empirical rerun, no new 2025 empirical artifact, no retune, no data acquisition, no result
replacement. The superseded generations (v1 result artifacts, the v2 invalidated config, v3 result artifacts and
the ledger's superseded rows) remain exactly as accepted, labelled as superseded.

**D. Hash integrity after CD-1.** CD-1 was an **intermediate draft** defect: the tail of one ledger hash on map
row C-40 was fabricated from an 8-character prefix. This build therefore ends with a fail-closed sweep of every
hash printed anywhere in the pack (`publication_pack.py hashcheck`): each published sha256 must be the measured
sha256 of a working-tree file, a value recorded in the accepted frozen manifest, or the config blob at the freeze
commit's parent; every abbreviated hex must resolve to a full form printed in the pack. No padded, reconstructed,
guessed, prefix-expanded or placeholder digests survive the sweep — `hash-integrity check OK: 14 published sha256 values verified (62 occurrences), 3 full git SHAs printed, 15 abbreviations resolved`.

## 7. How a reviewer reproduces this in three commands

```bash
git checkout publication/stat-arb-study
python3 publication/stat-arb-study/scripts/publication_pack.py check
python3 publication/stat-arb-study/scripts/publication_pack.py hashcheck
python3 publication/stat-arb-study/scripts/claim_crosscheck.py
```

All three are stdlib-only, offline, read-only and fail closed. The cross-check shares no code, helpers or bucket
table with the generator, so a defect in one cannot validate a false claim in the other.
