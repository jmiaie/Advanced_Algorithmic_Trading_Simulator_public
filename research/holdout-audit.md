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


## Post-freeze holdout execution

YAML status was set to `frozen-for-holdout` after formation+validation; calendar-2025 evaluation was then run once under frozen config `statarb_hist_oos_v1` (experiment `statarb_hist_oos_v1_holdout_2025`). Tracker confirmation: **FINAL CONFIGURATION FROZEN — Stat-Arb D9-B** on Issue #3 (https://github.com/jmiaie/quant-research-portfolio/issues/3#issuecomment-5691711387). Formation-time BH/FDR still selected **0** pairs, so holdout trading metrics are **null** (reported honestly). No retune after freeze.
