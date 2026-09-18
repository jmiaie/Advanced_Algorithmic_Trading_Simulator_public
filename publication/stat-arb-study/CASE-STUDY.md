# CASE-STUDY.md — D10-B `stat-arb-study`

**Reader-facing companion to the technical paper.** Same evidence, no new numbers.

*What this document is for:* a reader who wants to know what was built, what happened, and how to tell whether
the numbers are trustworthy — without reading the paper's design sections first.

---

## 1. In one paragraph

A statistical-arbitrage study looked for cointegrated ETF pairs inside four economically related groups, screened
candidate pairs with a false-discovery-rate control, and traded whatever survived — under a rule set frozen
before the D9-B 2025 evaluation run was executed. Over 32 walk-forward windows it admitted four tradable pairs, all of
them before 2024 and all under a pre-declared fallback rule. In the 2025 evaluation period it admitted none, so
the strategy held nothing. **That is the result: a complete null, reported as a null.**

## 2. The question worth asking

Anyone can build a pairs-trading backtest that trades. The interesting question is what happens when you forbid
yourself the usual escapes: no picking pairs because they worked, no widening a screen until something passes,
no re-running the evaluation period until it looks better. This study spent that discipline and got a null. The
null is more informative than a hand-picked positive would have been, because it is what the pre-declared
procedure actually emits.

## 3. What was built

| Layer | What it does | Where |
|---|---|---|
| Data | 19 US ETFs, daily bars, 2015-01-02 → 2025-12-31, frozen and hashed | §3 of `TECHNICAL-PAPER.md` (C-28…C-30) |
| Universe | 4 within-group buckets; pairs only inside a group | §4 (C-21) |
| Selection | Engle-Granger cointegration → BH-FDR → validation-block thresholds | §5, §6 (C-23, C-24) |
| Walk-forward | 504 / 126 / 63 / 63 sessions; rediscovery each window; strictly causal | §7 (C-22, C-10) |
| Costs | GROSS / BASE / STRESS scenarios; fixed-gross-notional sizing | §9 (C-25…C-27) |
| Publication | This pack: hashes, tables, figures, red-teams, offline verification | `SOURCE-GATE.md`, `RESULT-SOURCE-MAP.md` |

## 4. What actually happened

| Bucket | Windows | Traded | No trade | Grid-selected | Fallback used |
|---|---:|---:|---:|---:|---:|
| Development (2015-2023) | 25 | **4** | 21 | **0** | 4 |
| Boundary (2023-10 → 2024-01) | 1 | 0 | 1 | 0 | 0 |
| Validation (2024) | 3 | 0 | 3 | 0 | 0 |
| **Evaluation (2025)** | **3** | **0** | **3** | **0** | **0** |

Of 1,724 candidate pair tests across all buckets, four survived the false-discovery-rate screen — a survivor
rate of 0.23%. In the 2025 bucket: 165 tests, zero survivors, zero pairs, zero positions.

**The tuned-threshold path never fired.** Every traded window used the pre-declared fallback parameters rather
than a grid-selected optimum, and the frozen config records zero grid-selected windows across the entire study.
That is a finding about the design's selectivity, and it is one of the more interesting facts in the pack
(CL-13's reversal condition is the check on how far it can be pushed).

## 5. Why "no trades" is not "we broke it"

Three observable facts separate a legitimate null from a broken pipeline — all checkable in the accepted
artifacts and the git record:

1. **The mechanism demonstrably fires.** The same screen admitted four windows earlier in the study, with real
   selected pairs and recorded parameters (C-07, C-08, C-15).
2. **The run completed normally.** The 2025 evaluation was executed once, wrote its ledger row at
   2026-09-17T03:44:34Z, and its evidence commit landed at 03:48:08Z — no failure path, no partial artifact (C-34).
3. **Nothing was silently dropped.** One window straddling the 2024/2025 boundary was excluded from the 2025
   bucket *by design*, and the exclusion was surfaced through the runner's warning rather than disappearing from
   the counts (C-35, C-17).

What remains genuinely unresolved is *why* no 2025 pair passed: a real absence of cointegration, or a screen
whose strictness is mismatched to this asset class. This design cannot distinguish the two, and the pack says so
rather than guessing (paper §13, §16).

## 6. How to read the numbers safely

**Do read:**
- Window counts, candidate counts, survivor counts, no-trade flags, and the frozen parameters — these are direct
  artifact fields, hash-pinned, and reproduced in `tables/window-level.csv`.
- The claim → pointer map in `RESULT-SOURCE-MAP.md`; every number in the paper resolves to a field.

**Do not read:**
- Any performance implication for 2025. There is no 2025 return, Sharpe, drawdown or turnover figure, because
  there was no position to measure (C-11). If a reader finds such a number attributed to this study, it did not
  come from here.
- The three populated Kalman validation-Sharpe values as a verdict on hedge methods: OLS metrics are
  unpopulated throughout, so the pre-specified comparison is only partially evidenced (C-31…C-33).
- Anything about alpha, economic significance, capacity, or live viability. None were estimated, and
  `CLAIM-REGISTER.md` records that omission as an explicit decision (CL-21…CL-23), not an oversight.

## 7. The governance story (why the label matters)

The 2025 period carries the label **FINAL 2025 WALK-FORWARD EVALUATION**, deliberately *not* "untouched
holdout". Reason: 5 of 19 symbols overlap a separate, already-executed research program's 2025 universe, and the
chronology between the two designs is unresolved (C-18). Calling it untouched would be a stronger claim than the
evidence supports — precisely the kind of upgrade this pack exists to refuse.

Also refused: framing the null as "the strategy correctly stayed out of the market". The artifacts show no
qualifying opportunity and no position. They do not show motive, foresight, or defensive skill, and the pack does
not attribute any (paper §11).

## 8. Verification you can run yourself

```bash
# 1. hashes + deterministic regeneration + every citation resolves (fail-closed)
python3 publication/stat-arb-study/scripts/publication_pack.py check

# 2. independent re-assertion of the claim register values (separate code path)
python3 publication/stat-arb-study/scripts/claim_crosscheck.py
```

Both are stdlib-only, offline, and read-only. The second exists specifically so that a bug in the first cannot
validate a false claim: they share no code and parse the artifacts independently.

## 9. What comes next (and what does not)

- **Not** re-tuning this design until it trades. That would destroy the pre-registration that gives the null its
  value, and it is prohibited in this lane.
- **Not** quietly extending the label. Superseded lineage stays in place and labelled: the v1 result artifacts (3 files) and v3 result artifacts (3 JSON + a window-boundaries CSV) remain under their original labels; the v2 lineage is config-only (v2 produced no result artifacts before being invalidated, and this pack says so rather than implying otherwise); three explicit `…SUPERSEDED_*` ledger rows remain. Neither the artifacts nor the labels were touched.
- **Yes, if pursued deliberately:** a *new*, separately pre-registered study testing whether the screen's
  strictness matches this asset class — the one question this study raises and cannot answer (paper §16).

## 10. Provenance in one line

Accepted HEAD `d1ef6fd0` → freeze `28be77fb` → frozen config `5768fd10…` → primary artifact `b1197d0f…`
(2,359 B) → ledger row (2026-09-17T03:44:34Z) → this pack, offline-verified and CI-checked.
