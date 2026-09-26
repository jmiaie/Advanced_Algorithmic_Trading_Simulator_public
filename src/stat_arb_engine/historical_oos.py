"""Historical walk-forward / OOS helpers for Stat-Arb.

Holdout (2025) is blocked unless experiment status is frozen-for-holdout.
"""

from .historical_oos_backtest import (  # noqa: F401
    aggregate_pair_analytics,
    backtest_static_pair,
    kalman_diagnostics,
    sha256_text,
    summarize_selection,
    write_json_artifact,
)
from .historical_oos_orch import run_period_study  # noqa: F401
from .historical_oos_panel import (  # noqa: F401
    PeriodSpec,
    load_close_panel,
    select_pairs_within_sectors,
    slice_period,
)
