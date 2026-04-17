"""Risk / reward — trader's equation primitives.

The trader's equation is Brooks' core edge check: only take a setup
when (P_win × reward) > (P_lose × risk). This module computes the
mechanical risk and reward legs from a 5-minute bar DataFrame. The
probability side lives in the signal aggregator (Phase 3h).

Extracted from shared/brooks_score.py (Phase 3g).
"""

from typing import Optional

import pandas as pd

from aiedge.features.candles import MIN_RANGE


def _compute_risk_reward(df: pd.DataFrame, gap_direction: str,
                         prior_close: float, spike_bars: int,
                         rr_direction_override: Optional[str] = None,
                         ) -> tuple[float, float, float]:
    """Compute risk, reward, and R:R ratio with a minimum-risk floor.

    Risk: distance from the current close to the recent swing extreme
    (5-bar lookback), floored at half the average bar range. Reward:
    target projected from the measured-move of the initial spike.

    rr_direction_override: if set, use this direction for R/R calc
    instead of gap_direction. Used by bear-flip signals that need
    short-side R/R on a gap-up day.
    """
    direction = rr_direction_override or gap_direction
    current_price = df.iloc[-1]["close"]

    avg_bar_range = df.apply(lambda r: r["high"] - r["low"], axis=1).mean()
    min_risk = max(avg_bar_range * 0.5, MIN_RANGE)

    if direction == "up":
        lookback = min(5, len(df))
        recent_low = df.iloc[-lookback:]["low"].min()
        risk = max(current_price - recent_low, min_risk)

        spike_high = df.iloc[:max(spike_bars, 1)]["high"].max()
        spike_low = df.iloc[0]["low"]
        spike_height = spike_high - spike_low

        if spike_bars > 0 and len(df) > spike_bars:
            pullback_low = df.iloc[spike_bars:]["low"].min()
            target = pullback_low + spike_height
        else:
            target = current_price + spike_height
        reward = max(target - current_price, 0.0)
    else:
        lookback = min(5, len(df))
        recent_high = df.iloc[-lookback:]["high"].max()
        risk = max(recent_high - current_price, min_risk)

        spike_low = df.iloc[:max(spike_bars, 1)]["low"].min()
        spike_high = df.iloc[0]["high"]
        spike_height = spike_high - spike_low

        if spike_bars > 0 and len(df) > spike_bars:
            pullback_high = df.iloc[spike_bars:]["high"].max()
            target = pullback_high - spike_height
        else:
            target = current_price - spike_height
        reward = max(current_price - target, 0.0)

    rr_ratio = reward / risk if risk > min_risk else 0.0
    return round(risk, 2), round(reward, 2), round(rr_ratio, 2)
