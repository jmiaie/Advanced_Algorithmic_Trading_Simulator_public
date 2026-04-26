import pandas as pd
import matplotlib.pyplot as plt

class PerformanceMonitor:
    def __init__(self, fills_list):
        self.fills = pd.DataFrame(fills_list)
        
    def generate_tearsheet(self):
        if self.fills.empty:
            print("No trades executed.")
            return
        self.fills['cum_pnl'] = self.fills['net_cash_flow'].cumsum()
        print(f"Total PnL: ${self.fills['cum_pnl'].iloc[-1]:.2f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.fills['timestamp'], self.fills['cum_pnl'], label='Strategy Equity')
        plt.title("Cumulative PnL")
        plt.legend()
        plt.grid(True)
        plt.show()
