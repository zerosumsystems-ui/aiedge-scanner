"""Higher-timeframe alignment check.

A 5-min BUY_PULLBACK setup on a daily downtrend is a different trade
than one on a daily uptrend. This module classifies daily + weekly
bias from a longer-period bar series and reports whether the setup
direction aligns with both, one, or neither.

Output is consumed by the signal aggregator as a filter / confidence
modifier (Phase 6 wiring). No I/O here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Bias = Literal["up", "down", "neutral"]
Alignment = Literal["aligned", "mixed", "opposed", "no_data"]


def _trend_bias(closes: pd.Series | list[float], fast: int = 10, slow: int = 50) -> Bias:
    """Binary trend bias from a fast/slow EMA cross.

    `up` if fast > slow, `down` if fast < slow, `neutral` when they're
    within 0.1% of each other OR the series is too short.
    """
    arr = np.asarray(closes, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < slow:
        return "neutral"

    # Exponential smoothing — alpha = 2/(N+1), Wilder-style
    def _ema(x: np.ndarray, period: int) -> float:
        alpha = 2.0 / (period + 1)
        result = x[0]
        for v in x[1:]:
            result = alpha * v + (1 - alpha) * result
        return float(result)

    fast_v = _ema(arr[-fast:], fast)
    slow_v = _ema(arr[-slow:], slow)
    if slow_v == 0:
        return "neutral"
    rel = (fast_v - slow_v) / slow_v
    if abs(rel) < 0.001:
        return "neutral"
    return "up" if rel > 0 else "down"


def classify_htf_alignment(
    daily_closes: pd.Series | list[float],
    weekly_closes: pd.Series | list[float],
    setup_direction: Literal["long", "short"],
) -> dict:
    """Classify the setup against daily + weekly bias.

    Returns:
      {
        "daily_bias":   "up" | "down" | "neutral",
        "weekly_bias":  "up" | "down" | "neutral",
        "setup_direction": "long" | "short",
        "alignment":    "aligned" | "mixed" | "opposed" | "no_data",
      }

    Alignment semantics:
      - aligned:  both daily and weekly match the setup direction
      - opposed:  both daily and weekly oppose
      - mixed:    one matches, one is neutral or opposed
      - no_data:  either timeframe is neutral / not enough history
    """
    daily = _trend_bias(daily_closes, fast=5, slow=20)
    weekly = _trend_bias(weekly_closes, fast=4, slow=12)

    setup_is_up = setup_direction == "long"

    def _agrees(bias: Bias) -> bool | None:
        if bias == "neutral":
            return None
        return (bias == "up") == setup_is_up

    daily_ok = _agrees(daily)
    weekly_ok = _agrees(weekly)

    if daily_ok is None and weekly_ok is None:
        alignment: Alignment = "no_data"
    elif daily_ok and weekly_ok:
        alignment = "aligned"
    elif daily_ok is False and weekly_ok is False:
        alignment = "opposed"
    else:
        alignment = "mixed"

    return {
        "daily_bias": daily,
        "weekly_bias": weekly,
        "setup_direction": setup_direction,
        "alignment": alignment,
    }
