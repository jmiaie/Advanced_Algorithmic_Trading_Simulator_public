import numpy as np
from collections import deque
from typing import Dict, Any, List

class PairsTradingStrategy:
    def __init__(self, ticker_a: str, ticker_b: str, hedge_ratio: float = 1.0, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5) -> None:
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.hedge_ratio = hedge_ratio
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.prices = {ticker_a: None, ticker_b: None}
        self.spread_history = deque(maxlen=window)
        self.invested = 0 

    def calculate_signals(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
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
