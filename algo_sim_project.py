import os

project_name = "algo_sim_project"
files = {}

# --- FILE: execution.py ---
files["execution.py"] = r'''import numpy as np
import pandas as pd
from numba import jit

# 1. Numba Optimized Matching Engine
@jit(nopython=True)
def match_order_numba(book_prices, book_qtys, order_qty):
    fill_cost = 0.0
    filled_qty = 0.0
    remaining = order_qty
    n_levels = len(book_prices)
    for i in range(n_levels):
        if remaining <= 0: break
        price = book_prices[i]
        available = book_qtys[i]
        take = available if remaining >= available else remaining
        fill_cost += take * price
        filled_qty += take
        remaining -= take
    if filled_qty == 0: return 0.0
    return fill_cost / filled_qty

# 2. Limit Order Book Simulator
class LimitOrderBook:
    def __init__(self):
        self.bids = [] 
        self.asks = [] 

    def update(self, bid_price, ask_price, depth_qty=1000):
        self.bids = [(bid_price, depth_qty), (bid_price - 0.01, depth_qty)]
        self.asks = [(ask_price, depth_qty), (ask_price + 0.01, depth_qty)]

    def match_market_order(self, side, quantity):
        if side == 'buy':
            prices = np.array([x[0] for x in self.asks], dtype=np.float64)
            qtys = np.array([x[1] for x in self.asks], dtype=np.float64)
        else:
            prices = np.array([x[0] for x in self.bids], dtype=np.float64)
            qtys = np.array([x[1] for x in self.bids], dtype=np.float64)
        return match_order_numba(prices, qtys, quantity)

# 3. Execution Handler
class ExecutionHandler:
    def __init__(self, lob):
        self.lob = lob
        self.fills = []
        self.order_id_counter = 0

    def submit_order(self, event):
        self.order_id_counter += 1
        oid = self.order_id_counter
        if event['type'] == 'MARKET':
            fill_price = self.lob.match_market_order(event['side'], event['qty'])
            if fill_price > 0:
                self._record_fill(oid, event, fill_price)

    def _record_fill(self, oid, event, price):
        commission = max(1.0, event['qty'] * 0.005)
        fill = {
            'timestamp': event.get('timestamp', pd.Timestamp.now()),
            'symbol': event['symbol'],
            'side': event['side'],
            'qty': event['qty'],
            'price': price,
            'commission': commission,
            'net_cash_flow': (-1 if event['side'] == 'buy' else 1) * (event['qty'] * price) - commission
        }
        self.fills.append(fill)
'''

# --- FILE: strategies.py ---
files["strategies.py"] = r'''import numpy as np
from collections import deque

class PairsTradingStrategy:
    def __init__(self, ticker_a, ticker_b, hedge_ratio=1.0, window=20, entry_z=2.0, exit_z=0.5):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.prices = {ticker_a: None, ticker_b: None}
        self.spread_history = deque(maxlen=window)
        self.invested = 0 

    def calculate_signals(self, event):
        if event['type'] != 'MARKET': return []
        self.prices[event['symbol']] = event['price']
        p_a = self.prices[self.ticker_a]
        p_b = self.prices[self.ticker_b]
        if p_a is None or p_b is None: return []

        spread = p_a - (self.hedge_ratio * p_b)
        self.spread_history.append(spread)
        if len(self.spread_history) < self.window: return []

        arr = np.array(self.spread_history)
        std = np.std(arr)
        z_score = (spread - np.mean(arr)) / std if std > 0 else 0
        
        orders = []
        timestamp = event.get('timestamp')
        
        if self.invested == 0 and z_score < -self.entry_z:
            orders.append({'type': 'MARKET', 'symbol': self.ticker_a, 'side': 'buy', 'qty': 100, 'timestamp': timestamp})
            orders.append({'type': 'MARKET', 'symbol': self.ticker_b, 'side': 'sell', 'qty': int(100*self.hedge_ratio), 'timestamp': timestamp})
            self.invested = 1
        elif self.invested == 0 and z_score > self.entry_z:
            orders.append({'type': 'MARKET', 'symbol': self.ticker_a, 'side': 'sell', 'qty': 100, 'timestamp': timestamp})
            orders.append({'type': 'MARKET', 'symbol': self.ticker_b, 'side': 'buy', 'qty': int(100*self.hedge_ratio), 'timestamp': timestamp})
            self.invested = -1
        elif self.invested == 1 and z_score > -self.exit_z:
            orders.append({'type': 'MARKET', 'symbol': self.ticker_a, 'side': 'sell', 'qty': 100, 'timestamp': timestamp})
            orders.append({'type': 'MARKET', 'symbol': self.ticker_b, 'side': 'buy', 'qty': int(100*self.hedge_ratio), 'timestamp': timestamp})
            self.invested = 0
        elif self.invested == -1 and z_score < self.exit_z:
            orders.append({'type': 'MARKET', 'symbol': self.ticker_a, 'side': 'buy', 'qty': 100, 'timestamp': timestamp})
            orders.append({'type': 'MARKET', 'symbol': self.ticker_b, 'side': 'sell', 'qty': int(100*self.hedge_ratio), 'timestamp': timestamp})
            self.invested = 0
        return orders
'''

# --- FILE: engine.py ---
files["engine.py"] = r'''import queue

class DataStreamer:
    def __init__(self, df_dict):
        self.data = df_dict
        self.max_len = len(list(df_dict.values())[0])

    def stream_next(self):
        for i in range(self.max_len):
            events = []
            for sym, df in self.data.items():
                if i < len(df):
                    row = df.iloc[i]
                    events.append({'type': 'MARKET', 'symbol': sym, 'price': row['Close'], 'timestamp': row.name})
            yield events

class EventDrivenBacktester:
    def __init__(self, data_streamer, strategy, execution):
        self.data_streamer = data_streamer
        self.strategy = strategy
        self.execution = execution

    def run(self):
        print("--- Starting Backtest ---")
        for market_events in self.data_streamer.stream_next():
            for event in market_events:
                self.execution.lob.update(event['price'], event['price']+0.01)
                orders = self.strategy.calculate_signals(event)
                for order in orders:
                    self.execution.submit_order(order)
        print(f"--- Backtest Complete. Total Trades: {len(self.execution.fills)} ---")
'''

# --- FILE: analytics.py ---
files["analytics.py"] = r'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceMonitor:
    def __init__(self, fills_list):
        self.fills = pd.DataFrame(fills_list)
        
    def generate_tearsheet(self):
        if self.fills.empty:
            print("No trades executed.")
            return
        self.fills['cum_pnl'] = self.fills['net_cash_flow'].cumsum()
        print(f"Total PnL: ${self.fills['cum_pnl'].iloc[-1]:.2f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.fills['timestamp'], self.fills['cum_pnl'], label='Strategy Equity')
        plt.title("Cumulative PnL")
        plt.legend()
        plt.grid(True)
        plt.show()
'''

# --- FILE: main.py ---
files["main.py"] = r'''import pandas as pd
import numpy as np
from execution import LimitOrderBook, ExecutionHandler
from strategies import PairsTradingStrategy
from engine import DataStreamer, EventDrivenBacktester
from analytics import PerformanceMonitor

def generate_synthetic_data(length=500):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    noise = np.random.normal(0, 1, length)
    price_a = 100 + np.cumsum(noise)
    spread = np.random.normal(0, 2, length) 
    price_b = price_a - spread 
    return {'ASSET_A': pd.DataFrame({'Close': price_a}, index=dates),
            'ASSET_B': pd.DataFrame({'Close': price_b}, index=dates)}

if __name__ == "__main__":
    print("Generating Synthetic Data...")
    data_map = generate_synthetic_data()
    streamer = DataStreamer(data_map)
    lob = LimitOrderBook()
    execution = ExecutionHandler(lob)
    strategy = PairsTradingStrategy('ASSET_A', 'ASSET_B', hedge_ratio=1.0)
    backtester = EventDrivenBacktester(streamer, strategy, execution)
    backtester.run()
    monitor = PerformanceMonitor(execution.fills)
    monitor.generate_tearsheet()
'''

# --- FILE: requirements.txt ---
files["requirements.txt"] = "numpy>=1.21.0\npandas>=1.3.0\nmatplotlib>=3.4.0\nnumba>=0.53.0"

# --- FILE: Dockerfile ---
files["Dockerfile"] = r'''FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean
COPY . .
CMD ["python", "main.py"]
'''

def create_project():
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    for filename, content in files.items():
        path = os.path.join(project_name, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"Project '{project_name}' successfully created!")

if __name__ == "__main__":
    create_project()