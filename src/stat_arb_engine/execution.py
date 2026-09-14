from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from numba import jit
except Exception:  # pragma: no cover - optional acceleration only
    def jit(*_args: Any, **_kwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator


@jit(nopython=True)
def match_order_numba(
    book_prices: np.ndarray,
    book_qtys: np.ndarray,
    order_qty: float,
) -> Tuple[float, float]:
    fill_cost = 0.0
    filled_qty = 0.0
    remaining = order_qty
    n_levels = len(book_prices)
    for i in range(n_levels):
        if remaining <= 0:
            break
        price = book_prices[i]
        available = book_qtys[i]
        take = available if remaining >= available else remaining
        fill_cost += take * price
        filled_qty += take
        remaining -= take
    if filled_qty == 0:
        return 0.0, 0.0
    return fill_cost / filled_qty, filled_qty


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    borrow_bps_per_day: float = 0.0
    scenario_label: str = "stylized"

    def calculate(self, side: str, quantity: float, price: float) -> Dict[str, float | str]:
        _ = side
        notional = float(quantity) * float(price)
        commission = max(self.min_commission, float(quantity) * self.commission_per_share)
        spread_cost = notional * (self.spread_bps / 10_000.0)
        slippage_cost = notional * (self.slippage_bps / 10_000.0)
        impact_cost = notional * (self.impact_bps / 10_000.0)
        borrow_cost = 0.0
        return {
            "commission": commission,
            "spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "impact_cost": impact_cost,
            "borrow_cost": borrow_cost,
            "execution_cost": spread_cost + slippage_cost + impact_cost + borrow_cost,
            "total_cost": commission + spread_cost + slippage_cost + impact_cost + borrow_cost,
            "scenario_label": self.scenario_label,
        }


class LimitOrderBook:
    def __init__(self) -> None:
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []

    def update(
        self,
        bid_price: float,
        ask_price: float,
        depth_qty: float = 1000.0,
        levels: int = 2,
    ) -> None:
        self.bids = [(bid_price - (0.01 * i), depth_qty) for i in range(levels)]
        self.asks = [(ask_price + (0.01 * i), depth_qty) for i in range(levels)]

    def match_market_order(self, side: str, quantity: float) -> Dict[str, float]:
        if side == "buy":
            prices = np.array([x[0] for x in self.asks], dtype=np.float64)
            qtys = np.array([x[1] for x in self.asks], dtype=np.float64)
        else:
            prices = np.array([x[0] for x in self.bids], dtype=np.float64)
            qtys = np.array([x[1] for x in self.bids], dtype=np.float64)
        average_price, filled_qty = match_order_numba(prices, qtys, float(quantity))
        return {
            "average_price": float(average_price),
            "filled_qty": float(filled_qty),
            "remaining_qty": float(quantity - filled_qty),
        }


class ExecutionHandler:
    def __init__(self, lob: LimitOrderBook, cost_model: CostModel | None = None) -> None:
        self.lob = lob
        self.cost_model = cost_model or CostModel()
        self.fills: List[Dict[str, Any]] = []
        self.order_id_counter = 0

    def submit_order(self, event: Dict[str, Any]) -> Dict[str, Any] | None:
        self.order_id_counter += 1
        oid = self.order_id_counter
        if event["type"] != "MARKET":
            return None

        match = self.lob.match_market_order(event["side"], float(event["qty"]))
        if match["filled_qty"] <= 0:
            return None
        return self._record_fill(oid, event, match)

    def _record_fill(
        self,
        oid: int,
        event: Dict[str, Any],
        match: Dict[str, float],
    ) -> Dict[str, Any]:
        quantity = match["filled_qty"]
        price = match["average_price"]
        costs = self.cost_model.calculate(event["side"], quantity, price)
        total_cost = float(costs["total_cost"])
        signed_qty = quantity if event["side"] == "buy" else -quantity
        gross_cash_flow = -signed_qty * price
        fill = {
            "order_id": oid,
            "timestamp": pd.Timestamp(event.get("timestamp", pd.Timestamp.now())),
            "symbol": event["symbol"],
            "side": event["side"],
            "requested_qty": float(event["qty"]),
            "qty": float(quantity),
            "remaining_qty": float(match["remaining_qty"]),
            "price": float(price),
            "gross_cash_flow": float(gross_cash_flow),
            "net_cash_flow": float(gross_cash_flow - total_cost),
            **costs,
        }
        self.fills.append(fill)
        return fill
