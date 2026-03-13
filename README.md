# Advanced Algorithmic Trading Simulator

Event-driven pairs trading engine with live market integration via Alpaca.

## Architecture

```
├── main.py           # Backtest with synthetic data
├── strategies.py     # Pairs trading strategy (z-score based)
├── execution.py      # Limit order book + Numba-optimized matching
├── engine.py         # Event-driven backtester
├── analytics.py      # PnL tracking and tearsheets
├── live_feed.py      # Alpaca data feed + cointegration pair finder
├── live_trading.py   # Live/paper trading bridge
├── Dockerfile
└── requirements.txt
```

## Two Modes

### 1. Backtest (Synthetic Data)
```bash
python main.py
```

### 2. Live/Paper Trading (Alpaca)

**Step 1: Find cointegrated pairs**
```bash
python live_trading.py --scan --sector Energy
python live_trading.py --scan --sector Banks
```

**Step 2: Analyze a specific pair**
```bash
python live_trading.py --scan  # Shows z-scores, half-life, signals
```

**Step 3: Paper trade**
```bash
# Dry run first
python live_trading.py --pair XOM/CVX --dry-run --once

# Paper trading (continuous)
python live_trading.py --pair XOM/CVX --dry-run --interval 300
```

**Step 4: Live trade** ⚠️
```bash
python live_trading.py --pair XOM/CVX --qty 50 --interval 300
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scan` | - | Scan for cointegrated pairs |
| `--sector` | mixed | Sector to scan (Energy, Banks, Gold_Miners, etc.) |
| `--pair` | - | Pair to trade: `TICKER_A/TICKER_B` |
| `--hedge-ratio` | auto | Override hedge ratio (auto-calculated from cointegration) |
| `--qty` | 100 | Shares per leg |
| `--entry-z` | 2.0 | Z-score threshold to enter |
| `--exit-z` | 0.5 | Z-score threshold to exit |
| `--interval` | 300 | Seconds between checks |
| `--dry-run` | off | Log signals without trading |
| `--once` | off | Single cycle then exit |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Alpaca API keys (free at https://alpaca.markets)
```

## Strategy

**Pairs Trading via Z-Score Mean Reversion:**
1. Find cointegrated pairs (Engle-Granger + ADF test)
2. Calculate dynamic hedge ratio
3. Compute spread z-score over rolling window
4. Enter when z-score > 2σ (spread diverges)
5. Exit when z-score < 0.5σ (spread reverts)

**Risk**: Spread can diverge further (structural break). Half-life analysis helps gauge reversion speed.

## Author

Jeff Milam | [github.com/jmiaie](https://github.com/jmiaie)
