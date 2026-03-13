import pandas as pd
import numpy as np
from execution import LimitOrderBook, ExecutionHandler
from strategies import PairsTradingStrategy
from engine import DataStreamer, EventDrivenBacktester
from analytics import PerformanceMonitor

def generate_synthetic_data(length=500):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    noise = np.random.normal(0, 1, length)
    price_a = 100 + np.cumsum(noise)
    spread = np.random.normal(0, 2, length) 
    price_b = price_a - spread 
    return {'ASSET_A': pd.DataFrame({'Close': price_a}, index=dates),
            'ASSET_B': pd.DataFrame({'Close': price_b}, index=dates)}

if __name__ == "__main__":
    print("Generating Synthetic Data...")
    data_map = generate_synthetic_data()
    streamer = DataStreamer(data_map)
    lob = LimitOrderBook()
    execution = ExecutionHandler(lob)
    strategy = PairsTradingStrategy('ASSET_A', 'ASSET_B', hedge_ratio=1.0)
    backtester = EventDrivenBacktester(streamer, strategy, execution)
    backtester.run()
    monitor = PerformanceMonitor(execution.fills)
    monitor.generate_tearsheet()
