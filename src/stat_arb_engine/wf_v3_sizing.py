"""V3 position sizing: fixed_gross_notional, primary gross leverage <= 1.0x
allocated NAV. Leg quantities are normalized by the hedge ratio so total
absolute gross exposure respects the specified notional -- not the v1
fixed-100-share sizing, which the authoritative spec bans for the primary
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegSizes:
    qty_y: float
    qty_x: float
    gross_notional_target: float
    gross_notional_actual: float


def size_pair_legs(
    *,
    direction: int,
    hedge_ratio: float,
    price_y: float,
    price_x: float,
    allocated_nav: float,
    gross_notional_multiple: float = 1.0,
) -> LegSizes:
    """Size a spread position: long the spread (direction=+1) buys y and sells
    hedge_ratio*x; short the spread (direction=-1) is the reverse. Weights are
    split between legs in proportion to their dollar contribution to one unit
    of spread (1 : |hedge_ratio|) so total absolute gross notional equals
    gross_notional_multiple * allocated_nav exactly (respecting the hedge ratio).
    """
    if direction not in (-1, 0, 1):
        raise ValueError(f"direction must be -1, 0, or 1, got {direction}")
    if price_y <= 0 or price_x <= 0:
        raise ValueError("prices must be positive")
    if allocated_nav <= 0:
        raise ValueError("allocated_nav must be positive")
    if gross_notional_multiple > 1.0:
        raise ValueError(
            f"gross_notional_multiple {gross_notional_multiple} exceeds primary "
            "experiment's maximum gross leverage of 1.0x allocated NAV"
        )

    target_gross = gross_notional_multiple * allocated_nav
    if direction == 0:
        return LegSizes(0.0, 0.0, target_gross, 0.0)

    beta = abs(float(hedge_ratio))
    total_weight = 1.0 + beta
    y_notional = target_gross * (1.0 / total_weight)
    x_notional = target_gross * (beta / total_weight)

    # Sign convention: spread = y - hedge_ratio * x. Long the spread (direction=+1)
    # => long y, short hedge_ratio*x (same sign as hedge_ratio); short the spread
    # is the mirror image.
    qty_y = direction * (y_notional / price_y)
    qty_x = -direction * (1.0 if hedge_ratio >= 0 else -1.0) * (x_notional / price_x)

    gross_actual = abs(qty_y * price_y) + abs(qty_x * price_x)
    return LegSizes(
        qty_y=qty_y,
        qty_x=qty_x,
        gross_notional_target=target_gross,
        gross_notional_actual=gross_actual,
    )
