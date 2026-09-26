"""V3 backtest: fair OLS-vs-Kalman comparison on one selected pair over
one walk-forward test window, under fixed_gross_notional sizing and the
GROSS/BASE/STRESS cost scenarios.

Both hedge models are traded (not diagnostics-only, unlike v1/v2): the
spread is built continuously across formation+eval bars so the very first
eval-window bars still have a well-defined trailing z-score without ever
touching data past that bar (rolling stats are backward-looking, and are
additionally shifted by one bar before being applied to that bar's own
signal -- so the current observation never contaminates the z-score used
to decide on it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd

from .research import fit_dynamic_kalman_hedge_ratio, fit_static_ols_hedge_ratio
from .wf_v3_costs import CostScenario
from .wf_v3_sizing import size_pair_legs

Direction = Literal[-1, 0, 1]


@dataclass(frozen=True)
class SpreadSeries:
    spread: pd.Series
    hedge_ratio: pd.Series  # constant for static OLS, time-varying for Kalman
    intercept: pd.Series
    estimation_mode: str


def build_static_spread(
    formation_fit_prices: pd.DataFrame,
    combined_prices: pd.DataFrame,
    sym_a: str,
    sym_b: str,
) -> SpreadSeries:
    """Formation-fit static OLS (frozen intercept+slope) applied unchanged
    across the full formation+eval series. No refit on eval data."""
    fit = fit_static_ols_hedge_ratio(
        formation_fit_prices[sym_a], formation_fit_prices[sym_b]
    )
    y = combined_prices[sym_a]
    x = combined_prices[sym_b]
    spread = y - (fit.intercept + fit.slope * x)
    return SpreadSeries(
        spread=spread,
        hedge_ratio=pd.Series(fit.slope, index=combined_prices.index),
        intercept=pd.Series(fit.intercept, index=combined_prices.index),
        estimation_mode="static_batch_ols",
    )


def build_kalman_spread(
    combined_prices: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    *,
    process_variance: float,
    observation_variance: float,
) -> SpreadSeries:
    """Sequential forward-only Kalman filter run continuously across the full
    formation+eval series (no smoothing; state at bar t uses only bars <= t)."""
    result = fit_dynamic_kalman_hedge_ratio(
        combined_prices[sym_a],
        combined_prices[sym_b],
        process_variance=process_variance,
        observation_variance=observation_variance,
    )
    return SpreadSeries(
        spread=result.fitted_spread,
        hedge_ratio=result.beta_path,
        intercept=result.alpha_path,
        estimation_mode="sequential_filtered_kalman",
    )


def causal_zscore(spread: pd.Series, window: int) -> pd.Series:
    """z_t = (spread_t - rolling_mean_{t-1}) / rolling_std_{t-1}: the trailing
    window ends at t-1, so bar t's own value never enters the statistic used
    to score bar t (avoids the self-contamination the spec explicitly bans)."""
    rolling_mean = spread.rolling(window, min_periods=window).mean().shift(1)
    rolling_std = spread.rolling(window, min_periods=window).std(ddof=1).shift(1)
    z = (spread - rolling_mean) / rolling_std
    return z.replace([float("inf"), float("-inf")], pd.NA)


def generate_positions(zscore: pd.Series, entry_z: float, exit_abs_z: float) -> pd.Series:
    """State machine over z-score: enter short-spread when z >= entry_z,
    long-spread when z <= -entry_z, exit to flat when |z| <= exit_abs_z.
    Same-timestamp: the position decided from z_t is the position held at t
    (matches the program's existing same-timestamp-signal convention)."""
    positions: List[int] = []
    state = 0
    for z in zscore:
        if pd.isna(z):
            positions.append(state if state != 0 else 0)
            # Cannot evaluate signal with NaN z; hold prior state as-is if
            # already in a position (do not force a costless exit on missing
            # data), otherwise stay flat.
            continue
        if state == 0:
            if z >= entry_z:
                state = -1
            elif z <= -entry_z:
                state = 1
        else:
            if abs(z) <= exit_abs_z:
                state = 0
        positions.append(state)
    return pd.Series(positions, index=zscore.index, dtype=int)


@dataclass
class TradeFill:
    date: pd.Timestamp
    reason: str  # "entry" | "exit" | "reversal"


@dataclass
class BacktestResult:
    estimation_mode: str
    pair: str
    n_bars: int
    n_trades: int
    fills: List[TradeFill] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gross_return: float | None = None
    net_return: float | None = None
    annualized_vol: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    turnover: float | None = None
    cost_drag: float | None = None
    avg_holding_period_days: float | None = None
    gross_exposure_mean: float | None = None
    net_exposure_mean: float | None = None


TRADING_DAYS_PER_YEAR = 252


def simulate_pair_backtest(
    *,
    prices_y: pd.Series,
    prices_x: pd.Series,
    spread: SpreadSeries,
    positions: pd.Series,
    allocated_nav: float,
    gross_notional_multiple: float,
    cost: CostScenario,
    pair_label: str,
) -> BacktestResult:
    """Bar-by-bar simulation over the eval-window rows of `positions` (index
    must already be restricted to the window being scored -- callers slice
    before calling). Position sizing is fixed_gross_notional, re-sized on
    every entry/reversal to the current hedge ratio and NAV; held flat
    otherwise (no daily rebalancing drift beyond entry sizing, matching a
    standard pairs-trading convention and keeping turnover attributable to
    actual entries/exits rather than manufactured rebalance noise)."""
    idx = positions.index
    if len(idx) == 0:
        return BacktestResult(
            estimation_mode=spread.estimation_mode, pair=pair_label, n_bars=0, n_trades=0
        )

    cash = 0.0
    qty_y = 0.0
    qty_x = 0.0
    current_state = 0
    fills: List[TradeFill] = []
    equity: List[float] = []
    gross_exposures: List[float] = []
    net_exposures: List[float] = []
    entry_dates: List[pd.Timestamp] = []
    holding_periods: List[float] = []
    total_trade_notional = 0.0

    for t in idx:
        target_state = int(positions.loc[t])
        py = float(prices_y.loc[t])
        px = float(prices_x.loc[t])
        hedge_ratio_t = float(spread.hedge_ratio.loc[t])

        if target_state != current_state:
            # Close any existing position first.
            if current_state != 0:
                exit_notional = abs(qty_y * py) + abs(qty_x * px)
                cash += qty_y * py + qty_x * px
                cash -= cost.fill_cost(qty_y, py) + cost.fill_cost(qty_x, px)
                total_trade_notional += exit_notional
                qty_y = 0.0
                qty_x = 0.0
                fills.append(TradeFill(date=t, reason="exit"))
                if entry_dates:
                    holding_periods.append((t - entry_dates[-1]).days)
            if target_state != 0:
                legs = size_pair_legs(
                    direction=target_state,
                    hedge_ratio=hedge_ratio_t,
                    price_y=py,
                    price_x=px,
                    allocated_nav=allocated_nav,
                    gross_notional_multiple=gross_notional_multiple,
                )
                qty_y = legs.qty_y
                qty_x = legs.qty_x
                cash -= qty_y * py + qty_x * px
                cash -= cost.fill_cost(qty_y, py) + cost.fill_cost(qty_x, px)
                total_trade_notional += legs.gross_notional_actual
                fills.append(TradeFill(date=t, reason="entry"))
                entry_dates.append(t)
            current_state = target_state

        short_market_value = 0.0
        if qty_y < 0:
            short_market_value += abs(qty_y * py)
        if qty_x < 0:
            short_market_value += abs(qty_x * px)
        cash -= cost.daily_borrow_cost(short_market_value)

        mtm = cash + qty_y * py + qty_x * px
        equity.append(mtm)
        gross_exposures.append(abs(qty_y * py) + abs(qty_x * px))
        net_exposures.append(qty_y * py + qty_x * px)

    equity_curve = pd.Series(equity, index=idx, name="equity")
    n_trades = sum(1 for f in fills if f.reason == "entry")

    if allocated_nav <= 0 or equity_curve.empty:
        gross_return = net_return = ann_vol = sharpe = max_dd = None
    else:
        net_return = float(equity_curve.iloc[-1] / allocated_nav)
        daily_returns = equity_curve.diff().fillna(equity_curve.iloc[0]) / allocated_nav
        ann_vol = (
            float(daily_returns.std(ddof=1) * (TRADING_DAYS_PER_YEAR**0.5))
            if len(daily_returns) > 1
            else None
        )
        mean_excess = float(daily_returns.mean())
        std_ret = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
        sharpe = (
            float(mean_excess / std_ret * (TRADING_DAYS_PER_YEAR**0.5)) if std_ret > 0 else None
        )
        running_peak = (allocated_nav + equity_curve).cummax()
        drawdown = ((allocated_nav + equity_curve) - running_peak) / allocated_nav
        max_dd = float(drawdown.min())
        # Gross P&L: net P&L plus total costs paid (costs are the only
        # difference between gross and net here since there is no separate
        # "gross fill price" -- fills execute at the same observed price
        # under every scenario, only the cost add-on differs).
        gross_return = None  # populated by caller when GROSS-scenario equity is available

    cost_drag = None  # populated by caller as BASE_net - GROSS_net (or STRESS - GROSS)
    turnover = (
        float(total_trade_notional / allocated_nav) if allocated_nav > 0 and n_trades > 0 else 0.0
    )

    return BacktestResult(
        estimation_mode=spread.estimation_mode,
        pair=pair_label,
        n_bars=len(idx),
        n_trades=n_trades,
        fills=fills,
        equity_curve=equity_curve,
        gross_return=gross_return,
        net_return=net_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=turnover,
        cost_drag=cost_drag,
        avg_holding_period_days=(
            float(sum(holding_periods) / len(holding_periods)) if holding_periods else None
        ),
        gross_exposure_mean=(
            float(sum(gross_exposures) / len(gross_exposures)) if gross_exposures else None
        ),
        net_exposure_mean=(
            float(sum(net_exposures) / len(net_exposures)) if net_exposures else None
        ),
    )
