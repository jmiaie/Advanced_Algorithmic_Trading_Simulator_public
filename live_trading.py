"""
Live Trading Bridge
Connects the backtest engine's PairsTradingStrategy to Alpaca for live/paper trading.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from live_feed import AlpacaDataFeed
from pair_finder import PairFinder
from stat_arb_engine.strategies import PairsTradingStrategy, PositionSizer

load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("live_trading.log")],
)


class LiveTradingEngine:
    """
    Runs the pairs trading strategy on live market data via Alpaca.

    Default mode is paper/dry-run research. Live submission requires explicit opt-in.
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
            position_sizer=PositionSizer(mode="fixed_shares", quantity=position_size),
            missing_bar_policy="drop",
        )

    def _get_latest_prices(self) -> Optional[dict]:
        try:
            bars_a = self.feed.get_bars(self.ticker_a, timeframe="1Day", limit=1)
            bars_b = self.feed.get_bars(self.ticker_b, timeframe="1Day", limit=1)
            if bars_a.empty or bars_b.empty:
                return None
            timestamp = min(bars_a.index[-1], bars_b.index[-1])
            if bars_a.index[-1] != bars_b.index[-1]:
                logger.warning("Latest bars are asynchronous; waiting for a common timestamp")
                return None
            return {
                self.ticker_a: float(bars_a["Close"].iloc[-1]),
                self.ticker_b: float(bars_b["Close"].iloc[-1]),
                "timestamp": timestamp,
            }
        except Exception:
            logger.error("Failed to get prices", exc_info=True)
            return None

    def _warm_up(self, lookback_days: int = 60):
        start = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        data = self.feed.get_multi_bars([self.ticker_a, self.ticker_b], start=start)
        if self.ticker_a not in data or self.ticker_b not in data:
            logger.error("Cannot warm up — missing data")
            return 0
        prices = pd.DataFrame(
            {
                self.ticker_a: data[self.ticker_a]["Close"],
                self.ticker_b: data[self.ticker_b]["Close"],
            }
        ).dropna()
        fed = 0
        for idx, row in prices.iterrows():
            self.strategy.calculate_signals(
                {
                    "type": "MARKET_BATCH",
                    "timestamp": idx,
                    "prices": {
                        self.ticker_a: float(row[self.ticker_a]),
                        self.ticker_b: float(row[self.ticker_b]),
                    },
                }
            )
            fed += 1
        logger.info("Warmed up strategy with %d synchronized observations", fed)
        return fed

    def _execute_orders(self, orders: list):
        for order in orders:
            symbol = order["symbol"]
            side = order["side"]
            qty = int(order["qty"])
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
                    side.upper(),
                    qty,
                    symbol,
                    result.get("id", ""),
                )
            except Exception:
                logger.error("Order failed for %s", symbol, exc_info=True)

    def run_once(self):
        prices = self._get_latest_prices()
        if not prices:
            logger.warning("No synchronized price data available")
            return
        logger.info(
            "Prices: %s=$%.2f, %s=$%.2f",
            self.ticker_a,
            prices[self.ticker_a],
            self.ticker_b,
            prices[self.ticker_b],
        )
        orders = self.strategy.calculate_signals(
            {
                "type": "MARKET_BATCH",
                "symbol": "PAIR",
                "timestamp": prices["timestamp"],
                "prices": {
                    self.ticker_a: prices[self.ticker_a],
                    self.ticker_b: prices[self.ticker_b],
                },
            }
        )
        if orders:
            logger.info("SIGNAL: %d orders generated", len(orders))
            self._execute_orders(orders)
        else:
            logger.info("No signal (position=%d, waiting...)", self.strategy.invested)

    def run_loop(self, interval_seconds: int = 300):
        logger.info("=" * 60)
        logger.info(
            "Live Trading: %s/%s | Position size: %d | Mode: %s",
            self.ticker_a,
            self.ticker_b,
            self.position_size,
            "DRY RUN" if self.dry_run else "LIVE",
        )
        logger.info("=" * 60)
        self._warm_up()
        if not self.dry_run:
            try:
                account = self.feed.get_account()
                logger.info("Account equity: $%s", account.get("equity", "?"))
            except Exception as exc:
                logger.warning("Could not fetch account: %s", exc)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Pairs Trading Engine")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for pairs only, don't trade",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Sector to scan (Energy, Banks, etc.)",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Pair to trade: TICKER_A/TICKER_B",
    )
    parser.add_argument(
        "--hedge-ratio",
        type=float,
        default=None,
        help="Override static OLS hedge ratio",
    )
    parser.add_argument("--qty", type=int, default=100, help="Base share quantity")
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Entry z-score threshold",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Exit z-score threshold",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Simulate without trading (default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly opt in to live order submission",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument(
        "--paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Alpaca paper trading account (default); --no-paper uses the live account",
    )
    return parser


def main():
    args = build_parser().parse_args()
    logger.info(
        "Alpaca account: %s | order mode: %s",
        "PAPER" if args.paper else "LIVE",
        "LIVE SUBMISSION" if args.live else "DRY RUN",
    )

    if args.live:
        args.dry_run = False
        logger.warning(
            "LIVE ORDER SUBMISSION ENABLED. This path is secondary to the default "
            "research workflow."
        )

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    feed = AlpacaDataFeed(api_key=api_key, secret_key=secret_key, paper=args.paper)
    if args.scan:
        logger.info("=== PAIR SCANNING MODE ===")
        finder = PairFinder(feed)
        pairs = finder.find_pairs(sector=args.sector)
        if not pairs:
            logger.info("No cointegrated pairs found after FDR control.")
            return
        logger.info("\n=== TOP COINTEGRATED PAIRS ===")
        for i, (a, b, pval, hr) in enumerate(pairs, 1):
            analysis = finder.analyze_pair(a, b)
            logger.info(
                "%d. %s/%s — EG p=%.4f, static_hr=%.3f, q? see analysis, z=%.2f, "
                "half_life=%.1fd, signal=%s",
                i,
                a,
                b,
                pval,
                hr,
                analysis["z_score_current"],
                analysis["half_life_days"],
                analysis["signal"],
            )
        return

    if not args.pair:
        logger.error("Specify --pair TICKER_A/TICKER_B or use --scan to find pairs")
        sys.exit(1)
    parts = args.pair.split("/")
    if len(parts) != 2:
        logger.error("Pair format: TICKER_A/TICKER_B (e.g., XOM/CVX)")
        sys.exit(1)
    ticker_a, ticker_b = parts

    hedge_ratio = args.hedge_ratio
    if hedge_ratio is None:
        logger.info("Calculating static OLS hedge ratio for %s/%s...", ticker_a, ticker_b)
        finder = PairFinder(feed)
        analysis = finder.analyze_pair(ticker_a, ticker_b)
        hedge_ratio = analysis.get("hedge_ratio", 1.0)
        logger.info(
            "Static OLS hedge ratio: %.4f (EG p=%.4f, z=%.2f, dynamic beta current=%.4f)",
            hedge_ratio,
            analysis.get("coint_pvalue", 0),
            analysis.get("z_score_current", 0),
            analysis.get("dynamic_beta_current", 0),
        )

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
