import numpy as np
import pandas as pd

from analytics import PerformanceMonitor
from engine import DataStreamer, EventDrivenBacktester
from execution import CostModel, ExecutionHandler, LimitOrderBook
from stat_arb_engine.portfolio import PortfolioLedger
from strategies import PairsTradingStrategy, PositionSizer


def generate_synthetic_data(length: int = 500):
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")
    noise = np.random.normal(0, 1, length)
    price_a = 100 + np.cumsum(noise)
    spread = np.random.normal(0, 2, length)
    price_b = price_a - spread
    return {
        "ASSET_A": pd.DataFrame({"Close": price_a}, index=dates),
        "ASSET_B": pd.DataFrame({"Close": price_b}, index=dates),
    }


if __name__ == "__main__":
    print("Generating Synthetic Data...")
    data_map = generate_synthetic_data()
    streamer = DataStreamer(data_map)
    lob = LimitOrderBook()
    cost_model = CostModel(spread_bps=1.0, slippage_bps=2.0, impact_bps=1.0)
    execution = ExecutionHandler(lob, cost_model=cost_model)
    ledger = PortfolioLedger(starting_cash=100_000.0)
    strategy = PairsTradingStrategy(
        "ASSET_A",
        "ASSET_B",
        hedge_ratio=1.0,
        position_sizer=PositionSizer(
            mode="fixed_gross_notional",
            gross_notional=20_000.0,
        ),
    )
    backtester = EventDrivenBacktester(streamer, strategy, execution, portfolio=ledger)
    backtester.run()
    monitor = PerformanceMonitor(
        execution.fills,
        portfolio_history=backtester.portfolio_history,
    )
    monitor.generate_tearsheet()
