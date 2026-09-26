"""V3 cost engine: GROSS / BASE / STRESS scenarios for the conforming
walk-forward stat-arb study. These are scenario assumptions, not measured
historical execution costs.

Borrow accrues daily on the actual short market value held that day
(annualized_bps / 252), not just at trade time -- unlike the existing
engine's CostModel (execution.py), whose borrow_cost is a stub that always
returns 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostScenario:
    label: str
    commission_per_share: float
    min_commission: float
    spread_bps_per_side: float
    slip_bps_per_side: float
    impact_bps_per_side: float
    borrow_bps_annualized: float

    def fill_cost(self, quantity: float, price: float) -> float:
        """One-sided execution cost (commission + spread + slippage + impact) for a fill."""
        qty = abs(float(quantity))
        if qty == 0.0:
            return 0.0
        notional = qty * abs(float(price))
        commission = max(self.min_commission, qty * self.commission_per_share)
        spread_cost = notional * (self.spread_bps_per_side / 10_000.0)
        slip_cost = notional * (self.slip_bps_per_side / 10_000.0)
        impact_cost = notional * (self.impact_bps_per_side / 10_000.0)
        return commission + spread_cost + slip_cost + impact_cost

    def daily_borrow_cost(self, short_market_value: float) -> float:
        """Accrued borrow cost for one trading day on the actual short market value held."""
        smv = abs(float(short_market_value))
        if smv == 0.0:
            return 0.0
        return smv * (self.borrow_bps_annualized / 10_000.0) / TRADING_DAYS_PER_YEAR


# Frozen per the pre-registered spec (Sec "Costs"): exact values, not tunable.
GROSS = CostScenario(
    label="analytical_zero",
    commission_per_share=0.0,
    min_commission=0.0,
    spread_bps_per_side=0.0,
    slip_bps_per_side=0.0,
    impact_bps_per_side=0.0,
    borrow_bps_annualized=0.0,
)
BASE = CostScenario(
    label="BASE",
    commission_per_share=0.005,
    min_commission=1.0,
    spread_bps_per_side=1.0,
    slip_bps_per_side=1.0,
    impact_bps_per_side=0.5,
    borrow_bps_annualized=50.0,
)
STRESS = CostScenario(
    label="STRESS",
    commission_per_share=0.005,
    min_commission=1.0,
    spread_bps_per_side=3.0,
    slip_bps_per_side=3.0,
    impact_bps_per_side=2.0,
    borrow_bps_annualized=200.0,
)

SCENARIOS: Mapping[str, CostScenario] = {"GROSS": GROSS, "BASE": BASE, "STRESS": STRESS}


def scenario_by_label(label: str) -> CostScenario:
    key = label.strip().upper()
    if key not in SCENARIOS:
        raise ValueError(f"Unknown cost scenario '{label}'; expected one of {list(SCENARIOS)}")
    return SCENARIOS[key]


def all_scenarios() -> Dict[str, CostScenario]:
    return dict(SCENARIOS)
