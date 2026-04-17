"""Session-level features — opening range, time-of-day helpers.

Extracted from shared/brooks_score.py (Phase 3b).
"""

import pandas as pd

from aiedge.features.candles import MIN_RANGE


# First N bars that define the opening range. On 5-min bars this is
# the first 30 minutes of the session.
OPENING_RANGE_BARS = 6


def _opening_range(
    df: pd.DataFrame,
    n_bars: int = OPENING_RANGE_BARS,
    avg_daily_range: float = None,
) -> dict:
    """Compute the opening range from the first N bars.

    Returns a dict:
        range_high, range_low, range_size — raw levels
        range_pct — range_size / avg_daily_range, capped implicitly
                    by the scale estimation below

    If avg_daily_range is not provided or is effectively zero, an
    estimate is built from the day's data so far. The estimate ramps
    up aggressiveness for early sessions:
        <2h of data → assume current range ≈ 50% of day
        <4h of data → assume 70% of day
        ≥4h        → assume 85% of day
    These heuristics are legacy — preserve until a better regime-aware
    estimator lands under Phase 6.
    """
    n = min(n_bars, len(df))
    if n < 1:
        return {
            "range_high": 0.0,
            "range_low": 0.0,
            "range_size": 0.0,
            "range_pct": 0.5,
        }

    range_high = float(df.iloc[:n]["high"].max())
    range_low = float(df.iloc[:n]["low"].min())
    range_size = range_high - range_low

    # Estimate avg daily range from the current session if not given.
    if avg_daily_range is None or avg_daily_range <= MIN_RANGE:
        day_range = float(df["high"].max() - df["low"].min())
        hours_of_data = len(df) * 5 / 60  # assume 5-min bars
        if hours_of_data < 2:
            avg_daily_range = day_range / 0.5
        elif hours_of_data < 4:
            avg_daily_range = day_range / 0.7
        else:
            avg_daily_range = day_range / 0.85
        avg_daily_range = max(avg_daily_range, MIN_RANGE)

    range_pct = range_size / avg_daily_range

    return {
        "range_high": round(range_high, 4),
        "range_low": round(range_low, 4),
        "range_size": round(range_size, 4),
        "range_pct": round(range_pct, 3),
    }
