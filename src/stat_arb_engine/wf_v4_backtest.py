"""D9-B v4 backtest execution: fixes an execution-timing defect an
independent review found in v3.

v3's simulate_pair_backtest decided a position using information through
bar t (the causal z-score is a function of spread_t and a trailing window
shifted to end at t-1, so it still uses spread_t -- today's own close) and
then filled that trade AT bar t's own price. For the sequential Kalman
hedge ratio specifically, the filtered estimate at bar t also incorporates
bar t's own observation. Deciding and filling at the identical bar-t close
is an unrealistic same-close execution assumption.

v4 enforces a strict decision-vs-execution boundary: a position decided
with information through t is not executable until t+1's observed price,
sized with the hedge ratio estimate as of t (never any later Kalman
update). lag_for_execution() implements this by shifting the decision
series by one bar; simulate_pair_backtest() below applies it internally so
every caller gets it for free just by passing the full (unsliced)
decision series plus the window it wants executed.

Pair selection, sizing, and cost mechanics are unchanged from v3 and
reused directly (not implicated by this defect).

A second, independently-review-flagged defect in the drawdown calculation
(found in this same module, not carried over from a different one) is
also fixed here: `equity_curve` only ever appends bar-by-bar mark-to-
market values starting from the first *executed* bar, with no entry
representing NAV = allocated_nav "before" that first bar. Computing
`running_peak = (allocated_nav + equity_curve).cummax()` directly over
that series therefore never sees the true starting peak whenever the
first executed bar already shows a loss (e.g. immediate entry costs),
understating -- in the worst case erasing entirely -- the reported
drawdown. Concretely, for allocated_nav=100 and per-bar mark-to-market
P&L of [-1, -1] (100 -> 99 -> 99), the old code returned 0% instead of
the correct -1%. The old code also divided by the constant
`allocated_nav` rather than by the running peak at each point, which
understates drawdown further on any window where NAV had risen above
its starting allocation before falling back. Fixed by prepending an
explicit allocated_nav baseline observation to the NAV path before
taking `cummax()`, and dividing by that running peak at each point --
the same pattern `stat_arb_engine.analytics.calculate_max_drawdown`
already uses correctly elsewhere in this codebase. This affects the
reported `max_drawdown` value (and, downstream, the Sharpe-then-
drawdown-then-turnover tie-break that sorts on it) for any window with
at least one trade; windows with zero qualifying trades are unaffected
(a flat, all-zero equity curve has zero drawdown under either formula).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .wf_v3_backtest import SpreadSeries, TradeFill
from .wf_v3_costs import CostScenario
from .wf_v3_sizing import size_pair_legs

TRADING_DAYS_PER_YEAR = 252


def lag_for_execution(positions: pd.Series, hedge_ratio: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Shift decision-time positions/hedge-ratio by one bar: the value used
    to execute at bar t is whatever was decided at bar t-1, not bar t's own
    decision. Both series must be indexed over the full span a caller will
    later execute on (formation+validation+test, not a pre-sliced window),
    so the lagged value at the first bar of that window reflects a real
    prior decision instead of an artificial reset to flat/undefined.

    The leading bar of the full series has no t-1 and is filled with 0
    (flat) for positions; it is never itself used as an execution index in
    this study (formation always precedes any window that gets executed).
    """
    lagged_positions = positions.shift(1).fillna(0).astype(int)
    lagged_hedge_ratio = hedge_ratio.shift(1)
    return lagged_positions, lagged_hedge_ratio


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


def simulate_pair_backtest(
    *,
    prices_y: pd.Series,
    prices_x: pd.Series,
    spread: SpreadSeries,
    decision_positions: pd.Series,
    execution_index: pd.Index,
    allocated_nav: float,
    gross_notional_multiple: float,
    cost: CostScenario,
    pair_label: str,
) -> BacktestResult:
    """Same bar-by-bar mechanics as v3's simulate_pair_backtest, except
    `decision_positions` and `spread.hedge_ratio` are DECISION-time series
    (indexed by the bar the decision used information through, covering
    the full formation+validation+test span) that this function lags by
    one bar via lag_for_execution() before executing: the position and
    hedge ratio applied at each bar in `execution_index` are the values
    decided one bar earlier, filled at that bar's own observed price.
    """
    lagged_positions, lagged_hedge_ratio = lag_for_execution(decision_positions, spread.hedge_ratio)
    idx = execution_index
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
        target_state = int(lagged_positions.loc[t])
        py = float(prices_y.loc[t])
        px = float(prices_x.loc[t])
        hedge_ratio_t = float(lagged_hedge_ratio.loc[t])

        if target_state != current_state:
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
        net_return = ann_vol = sharpe = max_dd = None
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
        nav_path = pd.concat(
            [pd.Series([allocated_nav]), allocated_nav + equity_curve], ignore_index=True
        )
        running_peak = nav_path.cummax()
        drawdown = nav_path / running_peak - 1.0
        max_dd = float(drawdown.min())

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
        gross_return=None,
        net_return=net_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        turnover=turnover,
        cost_drag=None,
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
