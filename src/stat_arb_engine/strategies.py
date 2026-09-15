from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .engine import SynchronizedMarketBuffer


@dataclass(frozen=True)
class PositionSizer:
    mode: str = "fixed_shares"
    quantity: int = 100
    gross_notional: float = 10_000.0

    def size(
        self,
        prices: Mapping[str, float],
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
        spread_history: Optional[deque[float]] = None,
    ) -> Dict[str, int]:
        price_a = float(prices[ticker_a])
        price_b = float(prices[ticker_b])
        if self.mode == "fixed_shares":
            qty_a = self.quantity
            qty_b = max(1, int(round(abs(hedge_ratio) * self.quantity)))
        elif self.mode in {"fixed_gross_notional", "dollar_neutral"}:
            leg_notional = self.gross_notional / 2.0
            qty_a = max(1, int(leg_notional / price_a))
            qty_b = max(1, int(leg_notional / price_b))
        elif self.mode == "hedge_ratio_neutral":
            qty_a = self.quantity
            qty_b = max(1, int(round(abs(hedge_ratio) * self.quantity)))
        elif self.mode == "vol_scaled_gross":
            if spread_history is None or len(spread_history) < 5:
                raise ValueError("vol_scaled_gross sizing requires spread history")
            spread_vol = float(np.std(np.array(spread_history, dtype=float)))
            if spread_vol <= 0:
                raise ValueError("vol_scaled_gross sizing requires positive spread volatility")
            leg_notional = self.gross_notional / 2.0
            scale = min(2.0, 1.0 / spread_vol)
            qty_a = max(1, int((leg_notional * scale) / price_a))
            qty_b = max(1, int((leg_notional * scale) / price_b))
        else:
            raise ValueError(f"Unsupported sizing mode: {self.mode}")
        return {ticker_a: qty_a, ticker_b: qty_b}


class PairsTradingStrategy:
    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float = 1.0,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        position_sizer: PositionSizer | None = None,
        max_holding_period: int | None = None,
        structural_break_z: float | None = None,
        missing_bar_policy: str = "error",
    ) -> None:
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = float(hedge_ratio)
        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.position_sizer = position_sizer or PositionSizer()
        self.max_holding_period = max_holding_period
        self.structural_break_z = structural_break_z
        self.spread_history: deque[float] = deque(maxlen=window)
        self.invested = 0
        self.position_age = 0
        self.signal_history: List[Dict[str, Any]] = []
        self.sync_buffer = SynchronizedMarketBuffer(
            [ticker_a, ticker_b],
            missing_bar_policy=missing_bar_policy,
        )

    def _build_orders(
        self,
        side_a: str,
        side_b: str,
        timestamp: pd.Timestamp,
        prices: Mapping[str, float],
    ) -> List[Dict[str, Any]]:
        sizes = self.position_sizer.size(
            prices,
            self.ticker_a,
            self.ticker_b,
            hedge_ratio=self.hedge_ratio,
            spread_history=self.spread_history,
        )
        return [
            {
                "type": "MARKET",
                "symbol": self.ticker_a,
                "side": side_a,
                "qty": sizes[self.ticker_a],
                "timestamp": timestamp,
            },
            {
                "type": "MARKET",
                "symbol": self.ticker_b,
                "side": side_b,
                "qty": sizes[self.ticker_b],
                "timestamp": timestamp,
            },
        ]

    def _to_batch(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if event.get("type") == "MARKET_BATCH":
            prices = {
                key: float(value)
                for key, value in event["prices"].items()
            }
            missing = [
                symbol
                for symbol in (self.ticker_a, self.ticker_b)
                if symbol not in prices
            ]
            if missing:
                raise ValueError(
                    "MARKET_BATCH missing same-timestamp pair legs: "
                    + ", ".join(missing)
                )
            return {
                "type": "MARKET_BATCH",
                "timestamp": pd.Timestamp(event["timestamp"]),
                "prices": prices,
            }
        return self.sync_buffer.push(event)

    def _flat_orders(
        self,
        timestamp: pd.Timestamp,
        prices: Mapping[str, float],
    ) -> List[Dict[str, Any]]:
        side_a = "sell" if self.invested == 1 else "buy"
        side_b = "buy" if self.invested == 1 else "sell"
        return self._build_orders(side_a, side_b, timestamp, prices)

    def calculate_signals(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch = self._to_batch(event)
        if batch is None:
            return []

        prices = batch["prices"]
        spread = float(prices[self.ticker_a] - (self.hedge_ratio * prices[self.ticker_b]))
        timestamp = pd.Timestamp(batch["timestamp"])
        if len(self.spread_history) < self.window:
            self.spread_history.append(spread)
            return []

        history = np.array(self.spread_history, dtype=float)
        spread_mean = float(history.mean())
        spread_std = float(history.std())
        z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0.0
        orders: List[Dict[str, Any]] = []

        if self.invested != 0:
            self.position_age += 1

        structural_break = (
            self.structural_break_z is not None
            and self.invested != 0
            and abs(z_score) >= self.structural_break_z
        )
        max_holding_exit = (
            self.max_holding_period is not None
            and self.invested != 0
            and self.position_age >= self.max_holding_period
        )

        if structural_break or max_holding_exit:
            orders = self._flat_orders(timestamp, prices)
            self.invested = 0
            self.position_age = 0
        elif self.invested == 0 and z_score < -self.entry_z:
            orders = self._build_orders("buy", "sell", timestamp, prices)
            self.invested = 1
            self.position_age = 0
        elif self.invested == 0 and z_score > self.entry_z:
            orders = self._build_orders("sell", "buy", timestamp, prices)
            self.invested = -1
            self.position_age = 0
        elif self.invested == 1 and z_score > -self.exit_z:
            orders = self._build_orders("sell", "buy", timestamp, prices)
            self.invested = 0
            self.position_age = 0
        elif self.invested == -1 and z_score < self.exit_z:
            orders = self._build_orders("buy", "sell", timestamp, prices)
            self.invested = 0
            self.position_age = 0

        self.signal_history.append(
            {
                "timestamp": timestamp,
                "spread": spread,
                "z_score": z_score,
                "invested": self.invested,
            }
        )
        self.spread_history.append(spread)
        return orders
