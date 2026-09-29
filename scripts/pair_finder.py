import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from live_feed import AlpacaDataFeed
from stat_arb_engine.research import (
    compare_hedge_models,
    fit_dynamic_kalman_hedge_ratio,
    fit_static_ols_hedge_ratio,
    select_pairs,
)

logger = logging.getLogger(__name__)


class PairFinder:
    """
    Finds cointegrated pairs from a universe of stocks.
    Uses formation-only Engle-Granger testing with Benjamini-Hochberg control and
    static OLS / sequential Kalman diagnostics.
    """

    SECTOR_PAIRS = {
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
        "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
        "Tech_Hardware": ["AAPL", "MSFT", "DELL", "HPQ", "IBM"],
        "Retail": ["WMT", "TGT", "COST", "DG", "DLTR"],
        "Airlines": ["DAL", "UAL", "LUV", "AAL", "ALK"],
        "Gold_Miners": ["NEM", "GOLD", "AEM", "FNV", "WPM"],
        "Oil_Services": ["SLB", "HAL", "BKR", "FTI"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
        "REITs": ["PLD", "AMT", "CCI", "EQIX", "SPG", "O"],
        "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
    }

    def __init__(self, data_feed: AlpacaDataFeed, min_period_days: int = 252):
        self.data_feed = data_feed
        self.min_period_days = min_period_days

    def _resolve_universe(
        self,
        universe: Optional[List[str]],
        sector: Optional[str],
    ) -> List[str]:
        if universe is not None:
            return universe
        if sector and sector in self.SECTOR_PAIRS:
            return self.SECTOR_PAIRS[sector]
        return (
            self.SECTOR_PAIRS["Energy"][:5]
            + self.SECTOR_PAIRS["Banks"][:5]
            + self.SECTOR_PAIRS["Gold_Miners"][:4]
        )

    def _get_price_frame(self, universe: List[str]) -> pd.DataFrame:
        start_date = (datetime.utcnow() - timedelta(days=self.min_period_days)).strftime(
            "%Y-%m-%d"
        )
        logger.info("Fetching data for %d symbols from %s...", len(universe), start_date)
        data = self.data_feed.get_multi_bars(universe, start=start_date)
        close_prices = {
            symbol: df["Close"]
            for symbol, df in data.items()
            if len(df) >= self.min_period_days * 0.7
        }
        if len(close_prices) < 2:
            return pd.DataFrame()
        return pd.DataFrame(close_prices).dropna()

    def find_pairs(
        self,
        universe: Optional[List[str]] = None,
        sector: Optional[str] = None,
        p_value_threshold: float = 0.05,
        top_n: int = 10,
        fdr_alpha: Optional[float] = None,
    ) -> List[tuple[str, str, float, float]]:
        universe = self._resolve_universe(universe, sector)
        price_df = self._get_price_frame(universe)
        if price_df.empty:
            logger.warning("Not enough data for pair finding")
            return []
        alpha = p_value_threshold if fdr_alpha is None else fdr_alpha
        sector_map = {symbol: sector for symbol in universe} if sector else None
        selection = select_pairs(
            price_df,
            universe=price_df.columns,
            sector_grouping=sector_map,
            fdr_alpha=alpha,
            adf_alpha=p_value_threshold,
        )
        selected = selection.loc[selection["rejected"]].head(top_n)
        return [
            (row.symbol_a, row.symbol_b, row.engle_granger_pvalue, row.hedge_ratio)
            for row in selected.itertuples(index=False)
        ]

    def analyze_pair(self, sym_a: str, sym_b: str) -> Dict:
        prices = self._get_price_frame([sym_a, sym_b])
        if prices.empty or sym_a not in prices or sym_b not in prices:
            return {"error": "Missing data"}

        static_result = fit_static_ols_hedge_ratio(prices[sym_a], prices[sym_b])
        dynamic_result = fit_dynamic_kalman_hedge_ratio(prices[sym_a], prices[sym_b])
        comparison = compare_hedge_models(prices[sym_a], prices[sym_b], test_index=prices.index)
        spread = static_result.residual_spread
        trailing_spread = spread.iloc[:-1]
        spread_mean = float(trailing_spread.mean()) if not trailing_spread.empty else 0.0
        spread_std = (
            float(trailing_spread.std(ddof=1))
            if len(trailing_spread) > 1
            else 0.0
        )
        z_score_current = 0.0
        if spread_std > 0:
            z_score_current = (float(spread.iloc[-1]) - spread_mean) / spread_std

        signal = "NEUTRAL"
        if z_score_current < -2:
            signal = "LONG_SPREAD"
        elif z_score_current > 2:
            signal = "SHORT_SPREAD"

        return {
            "pair": f"{sym_a}/{sym_b}",
            "static_model": "Static OLS Hedge Ratio",
            "coint_pvalue": round(static_result.engle_granger_pvalue, 4),
            "hedge_ratio": round(static_result.slope, 4),
            "intercept": round(static_result.intercept, 4),
            "spread_mean": round(spread_mean, 4),
            "spread_std": round(spread_std, 4),
            "z_score_current": round(z_score_current, 4),
            "half_life_days": round(static_result.half_life, 1),
            "adf_stat": round(static_result.adf_stat, 4),
            "adf_pvalue": round(static_result.adf_pvalue, 4),
            "data_points": static_result.formation_sample_count,
            "signal": signal,
            "dynamic_beta_current": round(float(dynamic_result.beta_path.iloc[-1]), 4),
            "dynamic_alpha_current": round(float(dynamic_result.alpha_path.iloc[-1]), 4),
            "dynamic_beta_path": [
                round(float(value), 6) for value in dynamic_result.beta_path.tail(10)
            ],
            "model_comparison_preview": comparison.tail(5).round(6).to_dict(
                orient="records"
            ),
        }
