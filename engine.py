from typing import Dict, Any, Generator, List
import pandas as pd

class DataStreamer:
    def __init__(self, df_dict: Dict[str, pd.DataFrame]) -> None:
        self.data = df_dict
        self.max_len = len(list(df_dict.values())[0])

    def stream_next(self) -> Generator[List[Dict[str, Any]], None, None]:
        for i in range(self.max_len):
            events = []
            for sym, df in self.data.items():
                if i < len(df):
                    row = df.iloc[i]
                    events.append({'type': 'MARKET', 'symbol': sym, 'price': row['Close'], 'timestamp': row.name})
            yield events

class EventDrivenBacktester:
    def __init__(self, data_streamer: DataStreamer, strategy: Any, execution: Any) -> None:
        self.data_streamer = data_streamer
        self.strategy = strategy
        self.execution = execution

    def run(self) -> None:
        print("--- Starting Backtest ---")
        for market_events in self.data_streamer.stream_next():
            for event in market_events:
                self.execution.lob.update(event['price'], event['price']+0.01)
                orders = self.strategy.calculate_signals(event)
                for order in orders:
                    self.execution.submit_order(order)
        print(f"--- Backtest Complete. Total Trades: {len(self.execution.fills)} ---")
