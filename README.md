# Statistical Arbitrage & Execution Research Engine

**Cointegration • Dynamic Hedge Estimation • Walk-Forward Validation • Execution & TCA**

A recruiter-facing statistical arbitrage research repository focused on **research validity, temporal discipline, execution-cost awareness, portfolio accounting, reproducibility, and truthful claims**. The repository preserves the original event-driven pairs prototype, but now positions it explicitly as a **research engine** rather than a finished production trading stack.

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
- portfolio accounting and recruiter-safe analytics scaffolding

Explicit limitations and truthful positioning:
- the execution simulator is **stylized** and **not an empirical L2 market microstructure reconstruction**
- the static baseline is **not dynamic**; previous README language claiming a dynamic hedge ratio from full-history `polyfit` was incorrect and has been corrected
- no out-of-sample performance claims are asserted here without reproducible runs
- the universe examples are still static convenience lists and therefore remain exposed to survivorship / selection bias if treated as a production universe
- live/paper Alpaca integration remains secondary to the default research/backtest workflow and now requires explicit live opt-in

## Repository layout

```text
src/stat_arb_engine/        # Research engine package
main.py                     # Synthetic offline demonstration
pair_finder.py              # Live-data pair scan bridge using formation-only research logic
live_trading.py             # Secondary Alpaca bridge (explicit live opt-in)
research/statistical-arbitrage-validation.md
results/
```

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

Artifact directories are reserved under:
- `results/synthetic/`
- `results/walk_forward/`
- `results/pair_selection/`
- `results/execution_costs/`
- `results/factor_attribution/`
- `results/sensitivity/`

Until reproducible OOS experiments are run, these artifacts use explicit **Results pending reproducible OOS run** placeholders rather than fabricated metrics.

## Consolidation note

This repository is the maintained recruiter-facing stat-arb research surface. A read-only audit of `jmiaie/Statistical_Arbitrage_and_Conintegration_Strategic_Analyst` found broader unsupported public claims around institutional realism, factor-isolated alpha, and dynamic/OOS evidence. Consolidate future public claims here unless that overlapping repository is narrowed and brought to the same validation standard.

## Live trading safety

Research/backtesting is the default path.

Live order submission is intentionally secondary and requires `--live` explicit opt-in. Even then, this repository should be treated as a research environment with stylized execution assumptions rather than production deployment evidence.

## Author

Jeff Milam | [github.com/jmiaie](https://github.com/jmiaie)
