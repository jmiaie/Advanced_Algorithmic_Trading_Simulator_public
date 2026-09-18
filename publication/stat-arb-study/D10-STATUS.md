# D10-STATUS.md — D10-B `stat-arb-study`

**Lane:** D10-B (Hai) · **Directive:** #10 (publication/communication over accepted D9 evidence only)
**Status: READY FOR INDEPENDENT D10 REVIEW.** No merge, no external publication, no D11.

*Date:* 2026-09-18 · *Pack:* `publication/stat-arb-study/` · *Every number below is captured from a command run,
not typed by hand.*

---

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
