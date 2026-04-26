import numpy as np
import pandas as pd
from numba import jit
from typing import List, Dict, Any, Tuple

@jit(nopython=True)
def match_order_numba(book_prices: np.ndarray, book_qtys: np.ndarray, order_qty: float) -> float:
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

class LimitOrderBook:
    def __init__(self) -> None:
        self.bids: List[Tuple[float, float]] = [] 
        self.asks: List[Tuple[float, float]] = [] 

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

class ExecutionHandler:
    def __init__(self, lob: LimitOrderBook) -> None:
        self.lob = lob
        self.fills: List[Dict[str, Any]] = []
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
