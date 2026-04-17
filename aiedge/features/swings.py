"""Swing high / swing low detection on an OHLC DataFrame.

A "swing" here is the narrowest possible form — a single bar whose
low (or high) is strictly below (above) both neighbours. No pivot-
strength filter, no fractal lookback > 1. Good enough for most of
the scorers that consume it; callers that need stricter pivots
should filter the result themselves.

Extracted from shared/brooks_score.py (Phase 3b).
"""

import pandas as pd


def _find_swing_lows(
    df: pd.DataFrame, min_bars: int = 3
) -> list[tuple[int, float]]:
    """Find swing lows.

    Returns [(bar_index, low_price)] for every bar where the low is
    strictly below both the previous and next bars' lows.

    The `min_bars` argument is historical — currently unused. Kept
    to preserve the legacy signature.
    """
    swings = []
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i]["low"] < df.iloc[i - 1]["low"]
            and df.iloc[i]["low"] < df.iloc[i + 1]["low"]
        ):
            swings.append((i, df.iloc[i]["low"]))
    return swings


def _find_swing_highs(
    df: pd.DataFrame, min_bars: int = 3
) -> list[tuple[int, float]]:
    """Find swing highs.

    Returns [(bar_index, high_price)] for every bar where the high
    is strictly above both the previous and next bars' highs.

    The `min_bars` argument is historical — currently unused.
    """
    swings = []
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i]["high"] > df.iloc[i - 1]["high"]
            and df.iloc[i]["high"] > df.iloc[i + 1]["high"]
        ):
            swings.append((i, df.iloc[i]["high"]))
    return swings
