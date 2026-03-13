"""
Live Data Feed — Alpaca Markets Integration
Fetches real market data and streams it into the event-driven engine.
Also includes a cointegration-based pair finder.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import requests
from statsmodels.tsa.stattools import coint, adfuller

logger = logging.getLogger(__name__)

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class AlpacaDataFeed:
    """Fetches historical and real-time data from Alpaca."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
        self.data_url = ALPACA_DATA_URL
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    def get_account(self) -> Dict:
        """Get account info."""
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical bars for a symbol."""
        params = {"timeframe": timeframe, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        resp = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/bars",
            headers=self.headers,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        bars = data.get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_multi_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols."""
        result = {}
        for sym in symbols:
            try:
                df = self.get_bars(sym, timeframe=timeframe, start=start, limit=limit)
                if not df.empty:
                    result[sym] = df
                    logger.info("Fetched %d bars for %s", len(df), sym)
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", sym, e)
        return result

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
    ) -> Dict:
        """Submit an order."""
        order = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        resp = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order,
        )
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def close_all_positions(self) -> Dict:
        """Liquidate all positions."""
        resp = requests.delete(f"{self.base_url}/v2/positions", headers=self.headers)
        resp.raise_for_status()
        return resp.json()


class PairFinder:
    """
    Finds cointegrated pairs from a universe of stocks.
    Uses Engle-Granger test with ADF stationarity check on residuals.
    """

    def __init__(self, data_feed: AlpacaDataFeed, min_period_days: int = 252):
        self.data_feed = data_feed
        self.min_period_days = min_period_days

    # Classic sector-neutral pair candidates
    SECTOR_PAIRS = {
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
        "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
        "Tech_Hardware": ["AAPL", "MSFT", "DELL", "HPQ", "IBM"],
        "Retail": ["WMT", "TGT", "COST", "DG", "DLTR"],
        "Airlines": ["DAL", "UAL", "LUV", "AAL", "ALK"],
        "Gold_Miners": ["NEM", "GOLD", "AEM", "FNV", "WPM"],
        "Oil_Services": ["SLB", "HAL", "BKR", "FTI"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
        "REITs": ["PLD", "AMT", "CCI", "EQIX", "SPG", "O"],
        "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
    }

    def find_pairs(
        self,
        universe: Optional[List[str]] = None,
        sector: Optional[str] = None,
        p_value_threshold: float = 0.05,
        top_n: int = 10,
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find cointegrated pairs.
        Returns: [(ticker_a, ticker_b, p_value, hedge_ratio), ...]
        """
        if universe is None:
            if sector and sector in self.SECTOR_PAIRS:
                universe = self.SECTOR_PAIRS[sector]
            else:
                # Default: combine a few liquid sectors
                universe = (
                    self.SECTOR_PAIRS["Energy"][:5]
                    + self.SECTOR_PAIRS["Banks"][:5]
                    + self.SECTOR_PAIRS["Gold_Miners"][:4]
                )

        start_date = (datetime.utcnow() - timedelta(days=self.min_period_days)).strftime("%Y-%m-%d")
        logger.info("Fetching data for %d symbols from %s...", len(universe), start_date)

        data = self.data_feed.get_multi_bars(universe, start=start_date)

        # Align all series to common dates
        close_prices = {}
        for sym, df in data.items():
            if len(df) >= self.min_period_days * 0.7:  # allow some missing days
                close_prices[sym] = df["Close"]

        if len(close_prices) < 2:
            logger.warning("Not enough data for pair finding")
            return []

        price_df = pd.DataFrame(close_prices).dropna()
        symbols = list(price_df.columns)
        logger.info("Testing %d symbols (%d pairs)...", len(symbols), len(list(combinations(symbols, 2))))

        results = []
        for sym_a, sym_b in combinations(symbols, 2):
            try:
                score, pvalue, _ = coint(price_df[sym_a], price_df[sym_b])

                if pvalue < p_value_threshold:
                    # Calculate hedge ratio via OLS
                    hedge_ratio = np.polyfit(price_df[sym_b], price_df[sym_a], 1)[0]

                    # Verify spread stationarity
                    spread = price_df[sym_a] - hedge_ratio * price_df[sym_b]
                    adf_stat, adf_pvalue, *_ = adfuller(spread)

                    if adf_pvalue < p_value_threshold:
                        results.append((sym_a, sym_b, pvalue, hedge_ratio))
                        logger.info(
                            "PAIR FOUND: %s/%s (coint p=%.4f, hedge=%.3f, ADF p=%.4f)",
                            sym_a, sym_b, pvalue, hedge_ratio, adf_pvalue,
                        )
            except Exception as e:
                logger.debug("Error testing %s/%s: %s", sym_a, sym_b, e)

        # Sort by p-value (strongest cointegration first)
        results.sort(key=lambda x: x[2])
        return results[:top_n]

    def analyze_pair(self, sym_a: str, sym_b: str) -> Dict:
        """Deep analysis of a specific pair."""
        start_date = (datetime.utcnow() - timedelta(days=self.min_period_days)).strftime("%Y-%m-%d")
        data = self.data_feed.get_multi_bars([sym_a, sym_b], start=start_date)

        if sym_a not in data or sym_b not in data:
            return {"error": "Missing data"}

        prices = pd.DataFrame({
            sym_a: data[sym_a]["Close"],
            sym_b: data[sym_b]["Close"],
        }).dropna()

        # Cointegration test
        _, coint_pvalue, _ = coint(prices[sym_a], prices[sym_b])

        # Hedge ratio
        hedge_ratio = np.polyfit(prices[sym_b], prices[sym_a], 1)[0]

        # Spread
        spread = prices[sym_a] - hedge_ratio * prices[sym_b]
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score_current = (spread.iloc[-1] - spread_mean) / spread_std

        # Half-life of mean reversion (Ornstein-Uhlenbeck)
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
        aligned.columns = ["lag", "diff"]
        beta = np.polyfit(aligned["lag"], aligned["diff"], 1)[0]
        half_life = -np.log(2) / beta if beta < 0 else float("inf")

        # ADF
        adf_stat, adf_pvalue, *_ = adfuller(spread)

        return {
            "pair": f"{sym_a}/{sym_b}",
            "coint_pvalue": round(coint_pvalue, 4),
            "hedge_ratio": round(hedge_ratio, 4),
            "spread_mean": round(spread_mean, 4),
            "spread_std": round(spread_std, 4),
            "z_score_current": round(z_score_current, 4),
            "half_life_days": round(half_life, 1),
            "adf_pvalue": round(adf_pvalue, 4),
            "data_points": len(prices),
            "signal": "LONG_SPREAD" if z_score_current < -2 else "SHORT_SPREAD" if z_score_current > 2 else "NEUTRAL",
        }
