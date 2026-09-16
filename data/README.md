# Data (Directive #9)

- `raw/` — vendor snapshots (yfinance CSVs). **Gitignored.** Local/agent acquisition only; never fetch from CI.
- `manifests/` — committed dataset manifests with SHA-256 after freeze.

Dataset for Stat-Arb D9-B: `yf_statarb_sector_equities_daily_2015_2025_v1`.

Acquire locally:

```bash
python scripts/acquire_yf_statarb_sector_equities_daily.py
```
