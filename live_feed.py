"""
Live Data Feed — Alpaca Markets Integration
Fetches real market data and streams it into the event-driven engine.
Also includes a cointegration-based pair finder.
"""
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
            except requests.RequestException as e:
                logger.warning("Network error fetching %s: %s", sym, e)
            except Exception:
                logger.error("Failed to fetch %s", sym, exc_info=True)
        return result

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict:
        """Submit an order."""
        order = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
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
