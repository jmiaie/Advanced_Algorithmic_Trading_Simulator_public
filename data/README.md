# Data

- Raw vendor snapshots live under `data/raw/` and are **gitignored**.
- Manifests under `data/manifests/` are committed after **DATA FROZEN**.

## Conforming dataset (v3/v4 ETF study)

- **ID:** `yf_stat_arb_etfs_daily_2015_2025_v1`
- **Acquire:** `python scripts/acquire_yf_stat_arb_etfs_daily.py` (local only; **never CI**)
- Universe: ETF groups A–D (see manifest `universe_groups`)

## Superseded exploratory dataset

- `yf_statarb_sector_equities_daily_2015_2025_v1` — PairFinder.SECTOR_PAIRS equities; **non-conforming**; retain manifest only for audit trail.
