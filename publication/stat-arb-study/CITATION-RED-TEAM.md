# CITATION-RED-TEAM.md — D10-B `stat-arb-study`

**Role:** adversarial review of *every citation in this pack*. For each cited path, number, hash or commit: does
it resolve, does the cited value match the bytes on disk, and is anything cited as current that is actually
superseded?

**Standing rule enforced here:** a hash proves a value is *intact*, never that it is *right for the name it
carries*; a filename is a *claim*; absence of a record is not absence of an event. Citations below are therefore
checked by **reading and hashing the file**, not by trusting a label.

---

## 1. Hash citations

| Cited artifact | Cited in | sha256 (cited) | Re-hashed on disk | Verdict |
|---|---|---|---|---|
| `results/historical_oos/statarb_hist_etf_wf_v4_holdout_2025.json` | gate, map C-01, SOURCE-GATE F8 | `b1197d0f…b6d66` | match | VERIFIED |
| `…_v4_dev_formation.json` | map C-12 | `52a45f3b…e3146` | match | VERIFIED |
| `…_v4_boundary_2023_2024.json` | map C-17 | `e311a0e2…9b25f2` | match | VERIFIED |
| `…_v4_val_2024.json` | map C-18 | `d59c9747…9dfbb8` | match | VERIFIED |
| `…_v3_boundary_2023_2024.json` | map C-20 | `e311a0e2…9b25f2` | match (≡ v4) | VERIFIED |
| `…_v3_val_2024.json` | map C-20 | `d59c9747…9dfbb8` | match (≡ v4) | VERIFIED |
| `…_v3_dev_formation.json` | map C-20 | `c7ca93a6…` | match (≠ v4) | VERIFIED |
| `results/historical_oos/statarb_hist_oos_v1_holdout_2025.json` | map C-38 (superseded) | `4fd88a82…` | match | VERIFIED |
| `configs/experiments/statarb_historical_etf_wf_v4.yaml` | SOURCE-GATE F6 | `5768fd10…5376d` | match | VERIFIED |
| `data/manifests/yf_stat_arb_etfs_daily_2015_2025_v1.json` | SOURCE-GATE F7 | `b2557499…c2e30` | match | VERIFIED |
| `research/experiment-ledger.csv` | map C-34, C-40 | `cb4b7b0f…3211` | match (re-hashed after D10 remediation) | VERIFIED |
| `research/holdout-audit.md` | SOURCE-GATE F13, map C-36 | `d1f15db8…f4bf24` | match | VERIFIED |
| `research/statistical-arbitrage-validation.md` | SOURCE-GATE F13 | `dde09a88…ce1370` | match | VERIFIED |

The hash set above is machine-enforced: `publication_pack.py check` re-hashes each path recorded in
`reproducibility.json` and fails closed on any mismatch.

## 2. Commit / provenance citations

| Cited | Claim | Verified how | Verdict |
|---|---|---|---|
| `d1ef6fd0…` accepted HEAD | branch base, PR #7 head | `git rev-parse`, live `pulls/7` | VERIFIED (no drift) |
| `28be77fb…` freeze commit | parent of accepted HEAD | `git log`/parent check | VERIFIED |
| freeze body text | pre-freeze→post-freeze config sha; no parameter change; "executed once" | commit message read verbatim | VERIFIED |
| evidence commit body | boundary-straddling window excluded, warning surfaced; 3 windows | commit message read verbatim | VERIFIED |
| `created_utc` 2026-09-17T03:44:34Z | ledger row written after freeze, before evidence commit | CSV field vs commit timestamp | VERIFIED |
| reviewer code HEAD `3527634430…` at freeze | cited in SOURCE-GATE F5 | freeze commit body | VERIFIED (commit body) |
| PR #7 open/unmerged/not-draft + title | SOURCE-GATE F3 | live API | VERIFIED |
| hub Issue #3 "stale by design" | FIELD 11 wording | gate text; **hub not modified, not awaited** | VERIFIED as cited-from-gate |

## 3. Label discipline checks

| Label | Required form | Present as | Verdict |
|---|---|---|---|
| 2025 period | FINAL 2025 WALK-FORWARD EVALUATION (never "untouched holdout") | used verbatim in paper §13/§15, SOURCE-GATE F10, CL-18 | PASS |
| superseded v1 | retained, labelled superseded — **not** edited | cited as superseded only | PASS (no v1 label change anywhere in pack) |
| v2 / v3 | superseded / invalidated — not cited as current evidence | cited only in the lineage/defect discussion (C-20, CL-20) | PASS |

## 4. Citations that could NOT be hash-verified (disclosed, not hidden)

| Citation | Status | Consequence |
|---|---|---|
| `SOURCE-GATE-MATRIX.md`, `SOURCE-GATE-PACKAGE.md` (owner-supplied Phase 0 gate) | **Not in this repository** — cited by name and date only | Their Stream-B row values were **independently re-checked against repository bytes** (canonical hash, both config SHAs, freeze commit, artifact hash, accepted HEAD, PR #7 head) — all matched. The documents themselves are accepted inputs, not verifiable repo objects |
| Raw per-symbol price CSVs (`yf_stat_arb_etfs_daily_2015_2025_v1`) | **Not committed** (`data/` holds manifests only) | The dataset canonical SHA is verified only to the *frozen manifest* (which declares it, and which the config freeze record repeats). Raw-file integrity cannot be re-derived from repository bytes — stated as a limit in SOURCE-GATE F7 and paper §3 |
| v2 result artifacts | **do not exist** — the v2 lineage is a config file (`configs/experiments/statarb_historical_etf_wf_v2.yaml`, sha256 `0c6b8405011205bdce0fd49d50230b584e4faaa345ed4d3fbb1432cea142db11`) plus one `…_v2_INVALIDATED` ledger row | Corrected during this build: an earlier draft stated "v1/v2/v3 artifacts remain in place"; measurement found no v2 result artifacts, and the wording in CL-20 / §10.6 / `SOURCE-GATE.md` was fixed to match the bytes rather than the lineage story |

## 5. Numerical citations

Every number in `TECHNICAL-PAPER.md` and `CASE-STUDY.md` carries a `C-nn` pointer or a section-level reference to
`RESULT-SOURCE-MAP.md`. Re-derivation of the pointer values is performed by
`scripts/claim_crosscheck.py` (independent code path, assertion-based) plus the generator's citation-existence
check. Numbers that appear in **both** the artifact and the ledger (2025 bucket counts, config sha, artifact
hash) were required to agree in both places — they do.

**No number in this pack originates from memory, from the gate's summary table alone, or from any source other
than the accepted artifacts, the frozen config, the frozen manifest, the ledger, or the git record.**

## 6. Findings

**0 P0, 0 P1.** Two disclosed non-verifiable citation classes (§4), both inherent to the accepted evidence
(owner-supplied gate documents; uncommitted raw data) and both stated in `SOURCE-GATE.md`. Three P2 review notes
live in `QUANT-RED-TEAM.md`. No citation in the pack resolves to a missing file, a mismatched hash, or a
superseded artifact presented as current.

### 6.1 Defects found in this pack during review (recorded, not hidden)

Both were found by this lane's own verification apparatus, and both were fixed before the pack's first commit.
They are logged here because a pack that claims hash discipline should show what its own checks caught.

| ID | Defect | How it surfaced | Fix |
|----|--------|-----------------|-----|
| CD-1 | An intermediate draft of map row **C-40** carried a **fabricated 64-hex tail** for the ledger hash — the first 8 hex characters (`85fd6bf5`) were real, the remaining 56 were not (the ledger hash was re-measured to `85fd6bf54713d640676bb4174ddfdf9fe2833762d0d919b9ed8a9e7c2af57dd8`). | Self-caught while re-reading the inserted row; the hash-mismatch check in `publication_pack.py check` would also have failed the pack. | Row re-written with the measured hash. No other hash in the pack is a prefix-completion: every published hash was produced by `sha256sum` on the file in this working tree. **Later change (2026-09-18, D10 CL-20 remediation):** three corrective `…SUPERSEDED_UNVERIFIABLE_ARTIFACT_HASH` rows were appended, so the ledger hash recorded here is retired; the current value is `cb4b7b0f…3211` (hash table above, and `RETIRED_HASHES` in `scripts/publication_pack.py`). |
| CD-2 | Draft wording in **CL-20 / paper §10.6 / `SOURCE-GATE.md` F13 / `CASE-STUDY.md` / `QUANT-RED-TEAM.md`** claimed "superseded **v1/v2/v3 artifacts** remain in place". Measurement found **no v2 result artifacts exist at all** (v2 was invalidated before holdout; its lineage is a config plus one `…_v2_INVALIDATED` ledger row), and a first count of v1 ledger rows said three when the ledger holds **four** (`prereg` was missed). | The `C-40` assertion in `claim_crosscheck.py` failed closed on both counts. | All five documents re-worded to the measured inventory; the assertion now enforces `1 v2_INVALIDATED + 4 v1 rows + 3 SUPERSEDED_* rows` out of 23 (26 as of the 2026-09-18 CL-20 remediation, which added three v1 corrective rows; the v1 and v2 row counts it checks are unchanged, the SUPERSEDED total is now six). |

Neither defect reached a claim about the *study*: CD-1 concerned a hash rendering and CD-2 concerned lineage
bookkeeping. No finding, null result or label was affected in any direction.
