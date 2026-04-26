"""
Live Trading Bridge
Connects the backtest engine's PairsTradingStrategy to Alpaca for live/paper trading.
"""
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

import pandas as pd

from live_feed import AlpacaDataFeed
from pair_finder import PairFinder
from strategies import PairsTradingStrategy

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_trading.log"),
    ],
)


class LiveTradingEngine:
    """
    Runs the pairs trading strategy on live market data via Alpaca.
    
    Modes:
      - paper: Paper trades on Alpaca (no real money)
      - live: Real money (use with caution)
      - scan: Just scan for pairs and report signals, no trading
    """

    def __init__(
        self,
        feed: AlpacaDataFeed,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float = 1.0,
        position_size: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        window: int = 20,
        max_loss_dollars: float = 500.0,
        dry_run: bool = True,
    ):
        self.feed = feed
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.position_size = position_size
        self.max_loss = max_loss_dollars
        self.dry_run = dry_run

        self.strategy = PairsTradingStrategy(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=hedge_ratio,
            window=window,
            entry_z=entry_z,
            exit_z=exit_z,
        )
        # Override qty
        self.strategy_qty = position_size

    def _get_latest_prices(self) -> Optional[dict]:
        """Get most recent closing prices for the pair."""
        try:
            bars_a = self.feed.get_bars(self.ticker_a, timeframe="1Day", limit=1)
            bars_b = self.feed.get_bars(self.ticker_b, timeframe="1Day", limit=1)

            if bars_a.empty or bars_b.empty:
                return None

            return {
                self.ticker_a: bars_a["Close"].iloc[-1],
                self.ticker_b: bars_b["Close"].iloc[-1],
                "timestamp": bars_a.index[-1],
            }
        except Exception:
            logger.error("Failed to get prices", exc_info=True)
            return None

    def _warm_up(self, lookback_days: int = 60):
        """Load historical data to warm up the strategy's spread history."""
        start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        data = self.feed.get_multi_bars(
            [self.ticker_a, self.ticker_b], start=start
        )

        if self.ticker_a not in data or self.ticker_b not in data:
            logger.error("Cannot warm up — missing data")
            return 0

        prices = pd.DataFrame({
            self.ticker_a: data[self.ticker_a]["Close"],
            self.ticker_b: data[self.ticker_b]["Close"],
        }).dropna()

        fed = 0
        for idx, row in prices.iterrows():
            for sym in [self.ticker_a, self.ticker_b]:
                event = {"type": "MARKET", "symbol": sym, "price": row[sym], "timestamp": idx}
                self.strategy.calculate_signals(event)
            fed += 1

        logger.info("Warmed up strategy with %d days of data", fed)
        return fed

    def _execute_orders(self, orders: list):
        """Execute order signals through Alpaca."""
        for order in orders:
            symbol = order["symbol"]
            side = order["side"]
            qty = self.strategy_qty

            if self.dry_run:
                logger.info("[DRY RUN] %s %d %s", side.upper(), qty, symbol)
                continue

            try:
                result = self.feed.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market",
                )
                logger.info(
                    "ORDER FILLED: %s %d %s @ market (id=%s)",
                    side.upper(), qty, symbol, result.get("id", ""),
                )
            except Exception:
                logger.error("Order failed for %s", symbol, exc_info=True)

    def run_once(self):
        """Single cycle: get prices → generate signals → execute."""
        prices = self._get_latest_prices()
        if not prices:
            logger.warning("No price data available")
            return

        logger.info(
            "Prices: %s=$%.2f, %s=$%.2f",
            self.ticker_a, prices[self.ticker_a],
            self.ticker_b, prices[self.ticker_b],
        )

        # Feed both prices to strategy
        all_orders = []
        for sym in [self.ticker_a, self.ticker_b]:
            event = {
                "type": "MARKET",
                "symbol": sym,
                "price": prices[sym],
                "timestamp": prices["timestamp"],
            }
            orders = self.strategy.calculate_signals(event)
            all_orders.extend(orders)

        if all_orders:
            logger.info("SIGNAL: %d orders generated", len(all_orders))
            self._execute_orders(all_orders)
        else:
            logger.info("No signal (position=%d, waiting...)", self.strategy.invested)

    def run_loop(self, interval_seconds: int = 300):
        """Continuous loop — check every N seconds."""
        logger.info("=" * 60)
        logger.info(
            "Live Trading: %s/%s | Position size: %d | Mode: %s",
            self.ticker_a, self.ticker_b, self.position_size,
            "DRY RUN" if self.dry_run else "LIVE",
        )
        logger.info("=" * 60)

        # Warm up
        self._warm_up()

        # Account info
        if not self.dry_run:
            try:
                account = self.feed.get_account()
                logger.info("Account equity: $%s", account.get("equity", "?"))
            except Exception as e:
                logger.warning("Could not fetch account: %s", e)

        cycle = 0
        try:
            while True:
                cycle += 1
                logger.info("--- Cycle %d at %s ---", cycle, datetime.utcnow().isoformat())
                self.run_once()
                logger.info("Sleeping %ds...", interval_seconds)
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Live Pairs Trading Engine")
    parser.add_argument("--scan", action="store_true", help="Scan for pairs only, don't trade")
    parser.add_argument("--sector", type=str, default=None, help="Sector to scan (Energy, Banks, etc.)")
    parser.add_argument("--pair", type=str, default=None, help="Pair to trade: TICKER_A/TICKER_B")
    parser.add_argument("--hedge-ratio", type=float, default=None, help="Override hedge ratio")
    parser.add_argument("--qty", type=int, default=100, help="Shares per leg (default: 100)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry z-score threshold")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit z-score threshold")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between checks (default: 300)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Simulate without trading")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--paper", action="store_true", default=True, help="Use paper trading API")
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    feed = AlpacaDataFeed(api_key=api_key, secret_key=secret_key, paper=args.paper)

    # --- Scan mode ---
    if args.scan:
        logger.info("=== PAIR SCANNING MODE ===")
        finder = PairFinder(feed)
        pairs = finder.find_pairs(sector=args.sector)

        if not pairs:
            logger.info("No cointegrated pairs found.")
            return

        logger.info("\n=== TOP COINTEGRATED PAIRS ===")
        for i, (a, b, pval, hr) in enumerate(pairs, 1):
            analysis = finder.analyze_pair(a, b)
            logger.info(
                "%d. %s/%s — coint_p=%.4f, hedge=%.3f, z=%.2f, half_life=%.1fd, signal=%s",
                i, a, b, pval, hr, analysis["z_score_current"],
                analysis["half_life_days"], analysis["signal"],
            )
        return

    # --- Trading mode ---
    if not args.pair:
        logger.error("Specify --pair TICKER_A/TICKER_B or use --scan to find pairs")
        sys.exit(1)

    parts = args.pair.split("/")
    if len(parts) != 2:
        logger.error("Pair format: TICKER_A/TICKER_B (e.g., XOM/CVX)")
        sys.exit(1)

    ticker_a, ticker_b = parts

    # Auto-calculate hedge ratio if not provided
    hedge_ratio = args.hedge_ratio
    if hedge_ratio is None:
        logger.info("Calculating hedge ratio for %s/%s...", ticker_a, ticker_b)
        finder = PairFinder(feed)
        analysis = finder.analyze_pair(ticker_a, ticker_b)
        hedge_ratio = analysis.get("hedge_ratio", 1.0)
        logger.info("Auto hedge ratio: %.4f (coint_p=%.4f, z=%.2f)",
                     hedge_ratio, analysis.get("coint_pvalue", 0), analysis.get("z_score_current", 0))

    engine = LiveTradingEngine(
        feed=feed,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=hedge_ratio,
        position_size=args.qty,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        dry_run=args.dry_run,
    )

    if args.once:
        engine._warm_up()
        engine.run_once()
    else:
        engine.run_loop(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
