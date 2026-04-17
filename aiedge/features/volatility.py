"""Volatility and range measurements at the session or daily level.

Extracted from shared/brooks_score.py (Phase 3b). Currently holds the
daily ATR helper. Future additions (regime percentiles, realized vol
buckets) will land here under Phase 6.
"""

import pandas as pd


def _compute_daily_atr(daily_bars: pd.DataFrame, period: int = 20) -> float:
    """Compute Average Daily Range from daily OHLCV bars.

    Note: this is the simple daily-range mean (high - low), NOT true-
    range with gap inclusion. Name "ATR" is legacy; functionally it's
    ADR (Average Daily Range). Callers expect ADR semantics — do not
    silently change to true-range without updating consumers.
    """
    if daily_bars is None or len(daily_bars) < 2:
        return 0.0
    ranges = daily_bars["high"] - daily_bars["low"]
    return float(ranges.tail(period).mean())
