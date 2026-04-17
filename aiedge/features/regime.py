"""Volatility regime classification.

Two complementary regime signals:

  - `atr_percentile(today_atr, atr_history)` — where today sits in the
    distribution of recent daily ATRs. 0.0 = quietest day in window,
    1.0 = loudest. Used by the aggregator to dampen signals in quiet
    regimes (mean-revert bias) and amplify in loud regimes (trend bias).

  - `realized_vol_tercile(closes, lookback_days)` — bucketed
    close-to-close vol relative to the last N days: "low" / "mid" /
    "high" tercile. Used by priors lookup to stratify empirical
    win-rates (Phase 6 risk/priors module).

Both functions are pure pandas/numpy math — no I/O, no state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


VolTercile = Literal["low", "mid", "high"]


def atr_percentile(today_atr: float, atr_history: pd.Series | list[float]) -> float:
    """Return today's ATR percentile in [0.0, 1.0] vs the supplied history.

    `atr_history` should be a sequence of prior-day ATRs (not including
    today). Empty history returns 0.5 ("unknown → centrist default").

    Ties count as "at or below" — if today equals an entry in history
    it counts toward the percentile.
    """
    arr = np.asarray(atr_history, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.5
    at_or_below = int(np.sum(arr <= today_atr))
    return at_or_below / arr.size


def realized_vol_tercile(
    closes: pd.Series | list[float],
    lookback_days: int = 20,
) -> VolTercile:
    """Classify recent close-to-close realized vol into low/mid/high
    terciles relative to the lookback window.

    Uses log returns over the last `lookback_days` closes. Fewer than
    3 closes or zero vol → "mid" (can't classify).
    """
    arr = np.asarray(closes, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 3:
        return "mid"

    window = arr[-lookback_days:] if arr.size > lookback_days else arr
    log_returns = np.diff(np.log(window))
    if log_returns.size < 2:
        return "mid"

    current_vol = float(np.std(log_returns, ddof=1))
    # Rolling history of window-vol — compare today to rolling sigma
    # over the preceding windows (if we have enough closes).
    if arr.size >= 2 * lookback_days:
        rolling = []
        for i in range(arr.size - lookback_days, -1, -lookback_days):
            chunk = arr[i:i + lookback_days]
            if chunk.size >= 3:
                chunk_returns = np.diff(np.log(chunk))
                if chunk_returns.size >= 2:
                    rolling.append(float(np.std(chunk_returns, ddof=1)))
        rolling_arr = np.asarray(rolling[1:], dtype=float)  # exclude current
        if rolling_arr.size >= 3:
            t1 = float(np.quantile(rolling_arr, 1 / 3))
            t2 = float(np.quantile(rolling_arr, 2 / 3))
            if current_vol <= t1:
                return "low"
            if current_vol >= t2:
                return "high"
            return "mid"

    # Fallback: single window → fixed-threshold tercile assignment
    # tuned for SPY-like equities (5-min vol ~0.001-0.003 daily)
    if current_vol < 0.006:
        return "low"
    if current_vol > 0.015:
        return "high"
    return "mid"
