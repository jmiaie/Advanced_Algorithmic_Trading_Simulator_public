from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence

import pandas as pd


@dataclass
class SynchronizedMarketBuffer:
    symbols: Sequence[str]
    missing_bar_policy: str = "error"
    last_emitted_timestamp: Optional[pd.Timestamp] = None
    pending_timestamp: Optional[pd.Timestamp] = None
    pending_prices: Dict[str, float] = field(default_factory=dict)

    def push(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if event.get("type") != "MARKET":
            return None

        symbol = event["symbol"]
        timestamp = pd.Timestamp(event["timestamp"])
        price = float(event["price"])

        if symbol not in self.symbols:
            raise ValueError(f"Unexpected symbol for synchronized buffer: {symbol}")
        if self.last_emitted_timestamp is not None:
            if timestamp < self.last_emitted_timestamp:
                raise ValueError("Non-monotonic market event timestamp received")
            if timestamp == self.last_emitted_timestamp:
                raise ValueError(f"Duplicate synchronized timestamp received at {timestamp}")

        if self.pending_timestamp is None:
            self.pending_timestamp = timestamp
        elif timestamp < self.pending_timestamp:
            raise ValueError("Non-monotonic pending timestamp received")
        elif timestamp > self.pending_timestamp:
            if len(self.pending_prices) != len(self.symbols):
                missing = sorted(set(self.symbols) - set(self.pending_prices))
                if self.missing_bar_policy == "error":
                    raise ValueError(
                        f"Missing synchronized bar(s) {missing} at {self.pending_timestamp}"
                    )
                if self.missing_bar_policy not in {"drop", "skip"}:
                    raise ValueError(f"Unsupported missing_bar_policy: {self.missing_bar_policy}")
            self.pending_timestamp = timestamp
            self.pending_prices = {}

        if symbol in self.pending_prices:
            raise ValueError(f"Duplicate timestamp for symbol {symbol} at {timestamp}")

        self.pending_prices[symbol] = price
        if len(self.pending_prices) < len(self.symbols):
            return None

        batch = {
            "type": "MARKET_BATCH",
            "timestamp": self.pending_timestamp,
            "prices": dict(self.pending_prices),
        }
        self.last_emitted_timestamp = self.pending_timestamp
        self.pending_timestamp = None
        self.pending_prices = {}
        return batch


class DataStreamer:
    def __init__(self, df_dict: Dict[str, pd.DataFrame], missing_bar_policy: str = "error") -> None:
        if not df_dict:
            raise ValueError("df_dict must not be empty")
        self.data = df_dict
        self.symbols = list(df_dict.keys())
        self.buffer = SynchronizedMarketBuffer(
            self.symbols,
            missing_bar_policy=missing_bar_policy,
        )
        self._validate_frames()

    def _validate_frames(self) -> None:
        for symbol, df in self.data.items():
            if "Close" not in df.columns:
                raise ValueError(f"Missing Close column for {symbol}")
            if df.index.has_duplicates:
                raise ValueError(f"Duplicate timestamps detected for {symbol}")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"Non-monotonic timestamps detected for {symbol}")
            if df["Close"].isna().any():
                raise ValueError(f"Missing Close values detected for {symbol}")

    def stream_batches(self) -> Generator[Dict[str, Any], None, None]:
        symbol_order = {symbol: index for index, symbol in enumerate(self.symbols)}
        events: List[Dict[str, Any]] = []
        for symbol, df in self.data.items():
            for timestamp, row in df.iterrows():
                events.append(
                    {
                        "type": "MARKET",
                        "symbol": symbol,
                        "price": float(row["Close"]),
                        "timestamp": pd.Timestamp(timestamp),
                    }
                )
        events.sort(
            key=lambda item: (
                pd.Timestamp(item["timestamp"]),
                symbol_order[item["symbol"]],
            )
        )

        for event in events:
            batch = self.buffer.push(event)
            if batch is not None:
                yield batch

    def stream_next(self) -> Generator[List[Dict[str, Any]], None, None]:
        for batch in self.stream_batches():
            yield [
                {
                    "type": "MARKET",
                    "symbol": symbol,
                    "price": price,
                    "timestamp": batch["timestamp"],
                }
                for symbol, price in batch["prices"].items()
            ]


class EventDrivenBacktester:
    def __init__(
        self,
        data_streamer: DataStreamer,
        strategy: Any,
        execution: Any,
        portfolio: Any | None = None,
    ) -> None:
        self.data_streamer = data_streamer
        self.strategy = strategy
        self.execution = execution
        self.portfolio = portfolio
        self.portfolio_history: List[Dict[str, Any]] = []

    def run(self) -> None:
        print("--- Starting Backtest ---")
        for market_event in self.data_streamer.stream_batches():
            timestamp = market_event["timestamp"]
            orders = self.strategy.calculate_signals(market_event)
            for order in orders:
                current_price = float(market_event["prices"][order["symbol"]])
                self.execution.lob.update(current_price - 0.01, current_price + 0.01)
                fill = self.execution.submit_order(order)
                if fill is not None and self.portfolio is not None:
                    self.portfolio.process_fill(fill)
            if self.portfolio is not None:
                snapshot = self.portfolio.mark_to_market(
                    market_event["prices"],
                    timestamp=timestamp,
                )
                self.portfolio_history.append(snapshot)
        print(f"--- Backtest Complete. Total Trades: {len(self.execution.fills)} ---")
