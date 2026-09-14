from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class PositionState:
    quantity: float = 0.0
    average_cost: float = 0.0


class PortfolioLedger:
    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.positions: Dict[str, PositionState] = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.commissions = 0.0
        self.transaction_costs = 0.0
        self.market_prices: Dict[str, float] = {}

    def process_fill(self, fill: Dict[str, Any]) -> None:
        symbol = fill["symbol"]
        position = self.positions.setdefault(symbol, PositionState())
        quantity = float(fill["qty"])
        price = float(fill["price"])
        signed_fill = quantity if fill["side"] == "buy" else -quantity

        self.cash += float(fill["net_cash_flow"])
        self.commissions += float(fill.get("commission", 0.0))
        self.transaction_costs += float(fill.get("execution_cost", 0.0))

        current_qty = position.quantity
        if current_qty == 0 or current_qty * signed_fill > 0:
            new_qty = current_qty + signed_fill
            if new_qty != 0:
                weighted_cost = (
                    abs(current_qty) * position.average_cost
                    + abs(signed_fill) * price
                )
                position.average_cost = weighted_cost / abs(new_qty)
            else:
                position.average_cost = 0.0
            position.quantity = new_qty
            return

        closing_qty = min(abs(current_qty), abs(signed_fill))
        if current_qty > 0:
            self.realized_pnl += (price - position.average_cost) * closing_qty
        else:
            self.realized_pnl += (position.average_cost - price) * closing_qty

        new_qty = current_qty + signed_fill
        if new_qty == 0:
            position.quantity = 0.0
            position.average_cost = 0.0
        elif current_qty * new_qty > 0:
            position.quantity = new_qty
        else:
            position.quantity = new_qty
            position.average_cost = price

    def mark_to_market(
        self,
        prices: Dict[str, float],
        timestamp: Any | None = None,
    ) -> Dict[str, Any]:
        self.market_prices.update({symbol: float(price) for symbol, price in prices.items()})
        position_market_values: Dict[str, float] = {}
        position_cost_bases: Dict[str, float] = {}
        unrealized = 0.0
        for symbol, position in self.positions.items():
            price = self.market_prices.get(symbol)
            if price is None:
                continue
            market_value = position.quantity * price
            cost_basis = position.quantity * position.average_cost
            position_market_values[symbol] = market_value
            position_cost_bases[symbol] = cost_basis
            if position.quantity > 0:
                unrealized += (price - position.average_cost) * position.quantity
            elif position.quantity < 0:
                unrealized += (position.average_cost - price) * abs(position.quantity)

        self.unrealized_pnl = unrealized
        net_market_value = sum(position_market_values.values())
        gross_market_value = sum(abs(value) for value in position_market_values.values())
        net_cost_basis = sum(position_cost_bases.values())
        gross_cost_basis = sum(abs(value) for value in position_cost_bases.values())
        gross_pnl = self.realized_pnl + self.unrealized_pnl
        net_pnl = gross_pnl - self.commissions - self.transaction_costs
        equity = self.cash + net_market_value
        snapshot = {
            "timestamp": (
                pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.now()
            ),
            "cash": self.cash,
            "positions": {
                symbol: {
                    "quantity": state.quantity,
                    "average_cost": state.average_cost,
                    "cost_basis": position_cost_bases.get(symbol, 0.0),
                    "market_value": position_market_values.get(symbol, 0.0),
                }
                for symbol, state in self.positions.items()
            },
            "market_prices": dict(self.market_prices),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "commissions": self.commissions,
            "transaction_costs": self.transaction_costs,
            "gross_cost_basis": gross_cost_basis,
            "net_cost_basis": net_cost_basis,
            "gross_market_value": gross_market_value,
            "net_market_value": net_market_value,
            "gross_exposure": gross_market_value,
            "net_exposure": net_market_value,
            "equity": equity,
            "nav": equity,
        }
        return snapshot
