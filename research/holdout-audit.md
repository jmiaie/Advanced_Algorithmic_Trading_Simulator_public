# Holdout audit — Stat-Arb Directive #9 (2025 calendar year)

**Repository:** `jmiaie/Advanced_Algorithmic_Trading_Simulator_public`  
**Branch:** `research/historical-oos-study`  
**Audit date (PT):** 2026-09-15  
**Holdout window under D9:** calendar **2025-01-01 ≤ t < 2026-01-01**  
**Dataset ID (planned):** `yf_statarb_sector_equities_daily_2015_2025_v1`

## Verdict

**CLEAR** — no evidence that calendar-year **2025** market data was previously inspected, tuned against, or used for empirical evaluation / performance claims in this repository.

## Scope searched

| Surface | Method | Finding |
|---|---|---|
| Working tree (src, scripts, tests, docs, research, results, README) | recursive text search for `2025` / `holdout` | Only this audit scaffold / D9 placeholders; no empirical 2025 metrics. |
| Results tree | listed `results/*` | Placeholder READMEs under `synthetic/`, `walk_forward/`, `pair_selection/`, `execution_costs/`, `sensitivity/`, `factor_attribution/`. No CSV/JSON OOS packs. |
| Research doc | `research/statistical-arbitrage-validation.md` | States “Results pending reproducible OOS run.” No 2025 evaluation. |
| PairFinder / live bridges | `pair_finder.py`, `live_feed.py`, `live_trading.py` | Convenience live/Alpaca paths; no pinned 2015–2025 study window; no committed 2025 OOS metrics. |
| Tests | `tests/` | Synthetic/unit fixtures for D3 gates and research helpers only. |
| GitHub code search (prior session notes) | empirical 2025 evaluation artifacts | **null** |

## Classification rules applied

- **CLEAR:** holdout calendar window not used for model selection, hyperparameter tuning, benchmark cherry-picking, or reported historical performance.
- **PREVIOUSLY INSPECTED:** any committed notebook/result/config that evaluates or plots 2025 pair returns for research decisions.

Copyright / changelog year strings do **not** count as empirical holdout inspection.

## Universe choice (explicit; not expanded mid-study)

Documented convenience universe from `pair_finder.PairFinder.SECTOR_PAIRS` (liquid US sector equities). Survivorship / convenience bias remains an explicit limitation (README + research doc). Frozen symbol list is locked in the dataset manifest after acquisition; mid-study expansion is forbidden.

## Restrictions until FINAL CONFIGURATION FROZEN

1. Do **not** evaluate models on 2025 for final claims.
2. Development / validation work uses **2015–2023** (formation/dev) and **2024** (validation) only, once data is frozen.
3. Pre-registered experiment configs under `configs/experiments/` remain **`not-yet-frozen-for-holdout`** until development+validation complete and a tracker `FINAL CONFIGURATION FROZEN` record exists on Issue #3.

## Sign-off

| Field | Value |
|---|---|
| Holdout status | **CLEAR** |
| Prior empirical 2025 evaluation artifacts | **null** (none found) |
| Ready for acquisition + pre-registration | **yes** |


## Post-freeze holdout execution (v1, superseded universe)

YAML status was set to `frozen-for-holdout` after formation+validation; calendar-2025 evaluation was then run once under frozen config `statarb_hist_oos_v1` (experiment `statarb_hist_oos_v1_holdout_2025`). Tracker confirmation: **FINAL CONFIGURATION FROZEN — Stat-Arb D9-B** on Issue #3 (https://github.com/jmiaie/quant-research-portfolio/issues/3#issuecomment-5691711387). Formation-time BH/FDR still selected **0** pairs, so holdout trading metrics are **null** (reported honestly). No retune after freeze.

**This section covers only v1's `PairFinder.SECTOR_PAIRS` universe** (individual equities: energy/bank/tech/retail/airline/miner/utility/REIT/pharma names). It does **not** cover the v3 authoritative ETF universe below.

## D9-B v3 holdout audit (authoritative ETF universe, dataset `yf_stat_arb_etfs_daily_2015_2025_v1`)

**Audit date (PT):** 2026-09-16

**Within-repo:** no code path in this repository's `wf_v3_*` pipeline reads, references, or was written with knowledge of 2025 values for the v3 universe (SPY, QQQ, DIA, IWM, XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, SHY, IEF, TLT, GLD, SLV). The v1 universe above shares zero symbols with v3, so v1's already-executed 2025 holdout cannot leak into v3 directly. DEV (2015-2023) + 2024-validation execution (see `research/experiment-ledger.csv`, `statarb_hist_etf_wf_v3_dev_formation` / `_val_2024`) used a panel with all rows dated 2025-01-01 or later structurally sliced out before the study ran (`scripts/run_statarb_wf_v3_study.py`) — verified via an explicit refusal test, not just a status check.

**Cross-repository exposure (disclosed, not "clear"):** 5 of the 19 v3 symbols — **SPY, QQQ, IWM, TLT, GLD** — are also the entire universe of `financial-dynamics-model`'s D9-A regime study, whose 2025 holdout (`fdm_hist_regime_v1_holdout_2025`) has **already been executed and its results discussed/reported** (e.g. SPY 2025 `mean_return_h1 = -0.000293`; FDM PR #12, Issue #3). This is a genuine cross-program information channel: someone reviewing this program has already seen how SPY/QQQ/IWM/TLT/GLD behaved in 2025, before D9-B v3's holdout stage.

No code in this repository's pipeline consumes FDM's results — pair selection and parameter selection here are fully mechanical (Engle-Granger + BH-FDR + a pre-specified grid with a pre-specified fallback), with no step that could read or be hand-tuned against FDM's reported 2025 numbers. So the *procedure* itself has no direct leakage path. But this audit does **not** claim full independence at the program level, and does not classify this as blanket **CLEAR** — it discloses the exposure so whoever authorizes the D9-B v3 holdout can judge whether it matters, rather than asserting an absence of any prior 2025 contact with these symbols anywhere in the program.

**Verdict for v3:** CLEAR within this repository's own pipeline; cross-program exposure for 5/19 symbols via FDM disclosed above, not resolved by this document alone.
