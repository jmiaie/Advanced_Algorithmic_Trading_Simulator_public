from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_series(values: Sequence[float] | pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(list(values), dtype=float)


def calculate_conventional_sharpe(
    returns: Sequence[float],
    periods_per_year: int = 252,
    risk_free_rate_per_period: float = 0.0,
) -> float:
    series = pd.Series(list(returns), dtype=float).dropna()
    if series.empty:
        return 0.0
    excess = series - risk_free_rate_per_period
    volatility = float(excess.std(ddof=1))
    if len(excess) < 2 or volatility == 0.0:
        return 0.0
    return float(excess.mean() / volatility * np.sqrt(periods_per_year))



def calculate_max_drawdown(returns: Sequence[float]) -> float:
    series = pd.Series(list(returns), dtype=float).dropna()
    if series.empty:
        return 0.0
    nav = pd.concat([pd.Series([1.0]), (1.0 + series).cumprod()], ignore_index=True)
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())



def compute_strategy_analytics(
    equity_curve: Sequence[float],
    returns: Sequence[float],
    turnover: Sequence[float] | pd.Series | None = None,
    gross_exposure: Sequence[float] | pd.Series | None = None,
    transaction_costs: Sequence[float] | pd.Series | None = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    equity = _as_series(equity_curve)
    ret = _as_series(returns)
    turnover_series = _as_series(turnover)
    exposure_series = _as_series(gross_exposure)
    cost_series = _as_series(transaction_costs)

    # Equity/NAV path is treated as post-cost marked-to-market wealth.
    net_return = 0.0
    if len(equity) > 1 and equity.iloc[0] != 0:
        net_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)

    ann_return = float((1.0 + ret.mean()) ** periods_per_year - 1.0) if not ret.empty else 0.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if len(ret) > 1 else 0.0
    sharpe = calculate_conventional_sharpe(ret, periods_per_year=periods_per_year)
    max_drawdown = calculate_max_drawdown(ret)
    cumulative_turnover = float(turnover_series.sum()) if not turnover_series.empty else 0.0
    avg_daily_turnover = float(turnover_series.mean()) if not turnover_series.empty else 0.0
    average_gross_exposure = float(exposure_series.mean()) if not exposure_series.empty else 0.0
    cost_drag = float(cost_series.sum()) if not cost_series.empty else 0.0

    # Dollar cost add-back is a qualified proxy for gross terminal wealth, not a
    # pathwise pre-cost equity curve. When cost_drag is zero, gross == net.
    if cost_drag > 0.0 and len(equity) > 1 and equity.iloc[0] != 0:
        gross_return = float(((equity.iloc[-1] + cost_drag) / equity.iloc[0]) - 1.0)
    else:
        gross_return = net_return

    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": cumulative_turnover,
        "average_daily_turnover": avg_daily_turnover,
        "average_gross_exposure": average_gross_exposure,
        "cost_drag": cost_drag,
    }


class PerformanceMonitor:
    def __init__(
        self,
        fills_list: Sequence[Dict[str, Any]],
        portfolio_history: Sequence[Dict[str, Any]] | None = None,
    ):
        self.fills = pd.DataFrame(fills_list)
        self.portfolio_history = pd.DataFrame(portfolio_history or [])

    def generate_tearsheet(self) -> None:
        if self.portfolio_history.empty:
            if self.fills.empty:
                print("No trades executed.")
                return
            cash_flow = self.fills["net_cash_flow"].cumsum()
            print(f"Cumulative Net Cash Flow (not equity): ${cash_flow.iloc[-1]:.2f}")
            return

        history = self.portfolio_history.copy()
        history["returns"] = history["equity"].pct_change().fillna(0.0)
        metrics = compute_strategy_analytics(
            equity_curve=history["equity"],
            returns=history["returns"],
            gross_exposure=history.get("gross_exposure", pd.Series(dtype=float)),
        )
        print(f"Final NAV: ${history['equity'].iloc[-1]:.2f}")
        print(f"Realized PnL: ${history['realized_pnl'].iloc[-1]:.2f}")
        print(f"Unrealized PnL: ${history['unrealized_pnl'].iloc[-1]:.2f}")
        print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

        plt.figure(figsize=(10, 6))
        plt.plot(history["timestamp"], history["equity"], label="Portfolio NAV")
        plt.title("Portfolio Equity Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
