# Statistical Arbitrage & Execution Research Engine

**Cointegration • Dynamic Hedge Estimation • Walk-Forward Validation • Execution & TCA**

A statistical arbitrage research repository focused on **research validity, temporal discipline, execution-cost awareness, portfolio accounting, reproducibility, and truthful claims**. The repository preserves the original event-driven pairs prototype, but now positions it explicitly as a **research engine** rather than a finished production trading stack.

## Headline result

The pre-registered 2025 walk-forward evaluation (`statarb_hist_etf_wf_v4`, 19 US ETFs in four within-group buckets) ran once over **three pure-2025 test windows** (2025-01-08 → 2025-10-09). Each window tested **55 candidate pairs** (165 in total); **zero survived Benjamini-Hochberg FDR control**, so **zero pairs qualified and zero trades were placed**.

This is the intended behavior of multiple-testing control, not a pipeline failure: the same screen admitted four pairs in the earlier development windows, and in 2025 it declined to trade rather than admit pairs that did not clear the pre-specified threshold. Because no positions were taken, **no 2025 Sharpe ratio, drawdown, return or turnover figure exists**, and none is estimated or implied.

- Technical write-up: [`publication/stat-arb-study/TECHNICAL-PAPER.md`](publication/stat-arb-study/TECHNICAL-PAPER.md) (short version: [`CASE-STUDY.md`](publication/stat-arb-study/CASE-STUDY.md))
- Result artifacts: [`results/historical_oos/`](results/historical_oos/) (primary: `statarb_hist_etf_wf_v4_holdout_2025.json`)
- Offline verification: `python3 publication/stat-arb-study/scripts/publication_pack.py check` and `python3 publication/stat-arb-study/scripts/claim_crosscheck.py`

## Research hypothesis and architecture

The working hypothesis is transparent mean-reversion in cointegrated pairs, evaluated with:
- **Static OLS Hedge Ratio** as the explicit baseline
- **Sequential Kalman hedge estimation** as the dynamic alternative
- **Formation / validation / test chronology** with no random splits
- **Formation-only pair selection** with **Benjamini-Hochberg / FDR** control on Engle-Granger scan p-values
- **Decision-time-safe trailing normalization** for z-score signals, excluding the current spread from its own mean/std estimate
- **Stylized cost-aware execution simulation** with commissions, spread, slippage, and impact scenarios
- **Portfolio ledger accounting** for cash, positions, realized PnL, unrealized PnL, exposure, and NAV

## What is implemented vs. what is not

Implemented in this public repo:
- event-time synchronization so pair signals wait for **same-timestamp** observations from both legs
- chronological split helpers and walk-forward window generation
- static OLS baseline diagnostics and sequential Kalman beta path estimation
- FDR-controlled pair selection on formation data
- stylized partial-fill / VWAP execution simulation with configurable scenario costs
- portfolio accounting and conservative performance-analytics helpers

Explicit limitations and truthful positioning:
- the execution simulator is **stylized** and **not an empirical L2 market microstructure reconstruction**
- the static baseline is **not dynamic**; previous README language claiming a dynamic hedge ratio from full-history `polyfit` was incorrect and has been corrected
- the pre-registered 2025 evaluation produced no trades (see **Headline result**), so this repository makes no out-of-sample performance claim
- the universe examples are still static convenience lists and therefore remain exposed to survivorship / selection bias if treated as a production universe
- static OLS vs Kalman nuance: `fit_dynamic_kalman_hedge_ratio` (`src/stat_arb_engine/research.py`) computes its residual at bar t from the state *after* the update with y_t, so that spread is an in-sample filtered residual; the v4 walk-forward backtest (`lag_for_execution` in `src/stat_arb_engine/wf_v4_backtest.py`) instead executes each bar with the hedge ratio (and position) decided at t-1
- live/paper Alpaca integration remains secondary to the default research/backtest workflow and now requires explicit live opt-in

## Repository layout

```text
src/stat_arb_engine/        # Research engine package
main.py                     # Synthetic offline demonstration
pair_finder.py              # Live-data pair scan bridge using formation-only research logic
live_trading.py             # Secondary Alpaca bridge (explicit live opt-in)
live_feed.py                # Alpaca data feed used by the legacy entry points
research/                   # Experiment ledger and study records
publication/stat-arb-study/ # Technical write-up + offline reproducibility bundle
results/
```

`main.py`, `pair_finder.py`, `live_trading.py` and `live_feed.py` are legacy root-level entry points kept for the original prototype workflow; the research code lives in `src/stat_arb_engine/` and `scripts/`.

## Setup

Core research environment:

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
mypy
```

Optional live integration extras:

```bash
pip install -e ".[dev,live]"
cp .env.example .env
# edit .env with Alpaca credentials
```

## Research workflow summary

1. Build a formation window.
2. Scan candidate pairs on formation data only.
3. Apply BH/FDR control to Engle-Granger scan p-values.
4. Use residual ADF as a secondary diagnostic filter rather than double-counted evidence.
5. Estimate the baseline **Static OLS Hedge Ratio** and compare against the sequential Kalman model.
6. Generate z-score signals from prior-window statistics only; do not let the current spread normalize itself.
7. Evaluate on untouched validation / test windows or walk-forward slices.
8. Report assumptions, cost scenarios, and limitations alongside results.

## Reproducible artifacts

The executed historical study lives in `results/historical_oos/` and is documented in `publication/stat-arb-study/`. Additional artifact directories are reserved under:
- `results/synthetic/`
- `results/walk_forward/`
- `results/pair_selection/`
- `results/execution_costs/`
- `results/factor_attribution/`
- `results/sensitivity/`

Directories without a committed run contain placeholder READMEs rather than fabricated metrics.

## Consolidation note

This repository is the maintained public stat-arb research repository. A read-only audit of `jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst` found broader unsupported public claims around institutional realism, factor-isolated alpha, and dynamic/OOS evidence. Consolidate future public claims here unless that overlapping repository is narrowed and brought to the same validation standard.

## Live trading safety

Research/backtesting is the default path.

Live order submission is intentionally secondary and requires `--live` explicit opt-in. The Alpaca paper account is the default; `--no-paper` selects the live account, and the selected account and order mode are logged at startup. Even then, this repository should be treated as a research environment with stylized execution assumptions rather than production deployment evidence.

## Author

Jeff Milam | [github.com/jmiaie](https://github.com/jmiaie)
