"""Layer-2 session-shape classifier.

Post-hoc / late-session labels for the day's overall character. Runs
on the full session, returns a probability distribution over 6 Brooks
session shapes (plus 'undetermined' baseline).

Brooks himself names day types AFTER the session's shape reveals
itself, so confidence is inherently lower in the first hour. A guard
hides the Layer-2 argmax from the live card until ≥10:30 ET
(SESSION_SHAPE_WARMUP_MINUTES).

Extracted from shared/brooks_score.py (Phase 3d).
"""

import numpy as np
import pandas as pd

from aiedge.features.candles import _safe_range
from aiedge.context.phase import _softmax


# ── Tunables ─────────────────────────────────────────────────────────
SESSION_SHAPE_CLASSIFIER_ENABLED = True       # kill-switch
SESSION_SHAPE_SOFTMAX_TEMP = 0.5
SESSION_SHAPE_WARMUP_MINUTES = 60             # suppress Layer 2 on card before this many minutes of session
SESSION_SHAPES = (
    "trend_from_open",
    "spike_and_channel",
    "trend_reversal",
    "trend_resumption",
    "opening_reversal",
    "undetermined",
)


def _shape_trend_from_open_raw(df: pd.DataFrame, direction: str) -> float:
    """Day opened with a clear spike in `direction`, minimal retrace in first hour."""
    if len(df) < 6:
        return 0.0
    first6 = df.iloc[:6]
    session_open = df.iloc[0]["open"]
    current_close = df.iloc[-1]["close"]

    # strong net move in first 30 min
    first6_move = (first6.iloc[-1]["close"] - session_open)
    if direction == "up" and first6_move <= 0:
        return 0.0
    if direction == "down" and first6_move >= 0:
        return 0.0
    session_range = max(df["high"].max() - df["low"].min(), 1e-9)
    first6_strength = abs(first6_move) / session_range
    # >=30% of entire session range in first 30 min = strong trend-from-open
    strength = min(1.0, first6_strength / 0.30)

    # first-hour retrace (bars 6-12): how much did it give back
    if len(df) >= 12:
        first_hour = df.iloc[:12]
        if direction == "up":
            retrace = (first_hour.iloc[:6]["high"].max() - first_hour.iloc[6:]["low"].min())
            leg = first_hour.iloc[:6]["high"].max() - session_open
        else:
            retrace = (first_hour.iloc[6:]["high"].max() - first_hour.iloc[:6]["low"].min())
            leg = session_open - first_hour.iloc[:6]["low"].min()
        retrace_pct = retrace / max(leg, 1e-9)
        retrace_score = max(0.0, 1.0 - retrace_pct * 2.0)  # 50% retrace = 0
    else:
        retrace_score = 0.5  # not enough data

    # current close still away from session open in direction
    held = (current_close - session_open) / session_range
    if direction == "up":
        hold_score = max(0.0, min(1.0, held * 2.0))
    else:
        hold_score = max(0.0, min(1.0, -held * 2.0))

    raw = strength * 0.45 + retrace_score * 0.30 + hold_score * 0.25
    return float(max(0.0, min(1.0, raw)))


def _shape_spike_and_channel_raw(df: pd.DataFrame, direction: str,
                                  spike_bars: int) -> float:
    """Leading spike, then sustained shallower channel in the same direction."""
    if len(df) < 10 or spike_bars < 2 or spike_bars >= len(df) - 3:
        return 0.0
    spike = df.iloc[:spike_bars]
    channel = df.iloc[spike_bars:]

    if direction == "up":
        spike_move = spike.iloc[-1]["high"] - df.iloc[0]["open"]
        channel_move = channel.iloc[-1]["close"] - spike.iloc[-1]["high"]
    else:
        spike_move = df.iloc[0]["open"] - spike.iloc[-1]["low"]
        channel_move = spike.iloc[-1]["low"] - channel.iloc[-1]["close"]

    if spike_move <= 0:
        return 0.0
    # Channel continues in same direction but slope shallower
    same_dir = channel_move > 0
    slope_ratio = (channel_move / max(len(channel), 1)) / max(spike_move / spike_bars, 1e-9)
    shallower = 0.0 < slope_ratio < 0.75  # shallower than spike

    channel_body_avg = np.mean([
        abs(r["close"] - r["open"]) / max(_safe_range(r), 1e-9)
        for _, r in channel.iterrows()
    ])
    spike_body_avg = np.mean([
        abs(r["close"] - r["open"]) / max(_safe_range(r), 1e-9)
        for _, r in spike.iterrows()
    ])
    body_shrink = channel_body_avg < spike_body_avg * 0.8

    raw = 0.0
    if same_dir:
        raw += 0.4
    if shallower:
        raw += 0.3
    if body_shrink:
        raw += 0.2
    # bonus: spike itself was strong
    if spike_body_avg > 0.5:
        raw += 0.1
    return float(max(0.0, min(1.0, raw)))


def _shape_trend_reversal_raw(df: pd.DataFrame, direction: str) -> float:
    """Opening direction reverses: mid-session peak/trough, then lower high / higher low,
    current close ≥50% back from the extreme.
    """
    if len(df) < 15:
        return 0.0
    session_open = df.iloc[0]["open"]
    current_close = df.iloc[-1]["close"]

    if direction == "up":
        peak_idx = df["high"].idxmax()
        try:
            peak_pos = df.index.get_loc(peak_idx)
        except KeyError:
            return 0.0
        if peak_pos < 3 or peak_pos > len(df) - 3:
            return 0.0   # peak must be interior
        peak_val = df.iloc[peak_pos]["high"]
        leg = peak_val - session_open
        if leg <= 0:
            return 0.0
        retrace = peak_val - current_close
        retrace_pct = retrace / max(leg, 1e-9)
        # did we make a lower high after the peak?
        post_peak = df.iloc[peak_pos + 1:]
        if len(post_peak) < 2:
            return 0.0
        second_high = post_peak["high"].max()
        lower_high = second_high < peak_val
    else:
        trough_idx = df["low"].idxmin()
        try:
            trough_pos = df.index.get_loc(trough_idx)
        except KeyError:
            return 0.0
        if trough_pos < 3 or trough_pos > len(df) - 3:
            return 0.0
        trough_val = df.iloc[trough_pos]["low"]
        leg = session_open - trough_val
        if leg <= 0:
            return 0.0
        retrace = current_close - trough_val
        retrace_pct = retrace / max(leg, 1e-9)
        post_trough = df.iloc[trough_pos + 1:]
        if len(post_trough) < 2:
            return 0.0
        second_low = post_trough["low"].min()
        lower_high = second_low > trough_val  # higher low for bear reversal

    # scoring
    raw = 0.0
    if retrace_pct >= 0.50:
        raw += 0.5
    if retrace_pct >= 0.80:
        raw += 0.2
    if lower_high:
        raw += 0.3
    return float(max(0.0, min(1.0, raw)))


def _shape_trend_resumption_raw(df: pd.DataFrame, direction: str) -> float:
    """Opening trend, then mid-session trading range (flag), then continuation."""
    if len(df) < 18:
        return 0.0
    open_leg = df.iloc[:6]
    middle = df.iloc[6:-6] if len(df) > 18 else df.iloc[6:-3]
    late = df.iloc[-6:]

    # open leg moved in direction
    om = open_leg.iloc[-1]["close"] - open_leg.iloc[0]["open"]
    if direction == "up" and om <= 0:
        return 0.0
    if direction == "down" and om >= 0:
        return 0.0

    # middle compressed (narrow range)
    mid_range = middle["high"].max() - middle["low"].min()
    open_range = open_leg["high"].max() - open_leg["low"].min()
    if mid_range <= 0 or open_range <= 0:
        return 0.0
    compression_ratio = mid_range / open_range
    compressed = compression_ratio < 1.2

    # late continuation
    lm = late.iloc[-1]["close"] - middle.iloc[-1]["close"]
    continuation = (direction == "up" and lm > 0) or (direction == "down" and lm < 0)

    raw = 0.0
    if compressed:
        raw += 0.5
    if continuation:
        raw += 0.4
    if abs(lm) > abs(om) * 0.3:
        raw += 0.1    # late move is meaningful size
    return float(max(0.0, min(1.0, raw)))


def _shape_opening_reversal_raw(df: pd.DataFrame, direction: str) -> float:
    """Opening breakout fails within 60-90 min, reverses ≥50% of opening thrust."""
    if len(df) < 18:
        return 0.0
    session_open = df.iloc[0]["open"]
    first_hr = df.iloc[:12]    # 60 min = 12 bars 5min

    # opening thrust in direction
    if direction == "up":
        opening_extreme = first_hr["high"].max()
        thrust = opening_extreme - session_open
    else:
        opening_extreme = first_hr["low"].min()
        thrust = session_open - opening_extreme
    if thrust <= 0:
        return 0.0

    # reversal: did price move BACK through opening and beyond at least 50%
    current = df.iloc[-1]["close"]
    if direction == "up":
        pullback = opening_extreme - current
    else:
        pullback = current - opening_extreme
    reversal_pct = pullback / max(thrust, 1e-9)

    # still reversing (current is below open for up-attempt, etc.)
    if direction == "up":
        net_negative = current < session_open
    else:
        net_negative = current > session_open

    raw = 0.0
    if reversal_pct >= 0.50:
        raw += 0.35
    if reversal_pct >= 1.00:
        raw += 0.25   # beyond the open
    if net_negative:
        raw += 0.30
    # extra credit if reversal happened in first 90 min (opening reversal timing)
    if len(df) > 18:
        mid_idx = 18
        if direction == "up":
            mid_low = df.iloc[:mid_idx]["low"].min()
            if mid_low < session_open:
                raw += 0.10
        else:
            mid_high = df.iloc[:mid_idx]["high"].max()
            if mid_high > session_open:
                raw += 0.10
    return float(max(0.0, min(1.0, raw)))


def classify_session_shape(df: pd.DataFrame, direction: str,
                            spike_bars: int = 0,
                            session_minutes: int = 0) -> dict:
    """Layer-2 session shape classifier.

    Returns probability distribution over 6 session shapes (including
    'undetermined' baseline) + argmax + confidence + a boolean
    `show_on_live_card` that is False during the warmup period.
    """
    if not SESSION_SHAPE_CLASSIFIER_ENABLED:
        return {}
    if df is None or len(df) < 6:
        return {
            "probs": {s: 1.0 / len(SESSION_SHAPES) for s in SESSION_SHAPES},
            "top": "undetermined",
            "confidence": 1.0 / len(SESSION_SHAPES),
            "show_on_live_card": False,
        }

    raw = {
        "trend_from_open":   _shape_trend_from_open_raw(df, direction),
        "spike_and_channel": _shape_spike_and_channel_raw(df, direction, spike_bars),
        "trend_reversal":    _shape_trend_reversal_raw(df, direction),
        "trend_resumption":  _shape_trend_resumption_raw(df, direction),
        "opening_reversal":  _shape_opening_reversal_raw(df, direction),
        "undetermined":      0.15,      # baseline — always present as the null hypothesis
    }
    values = [raw[k] for k in SESSION_SHAPES]
    probs = _softmax(values, SESSION_SHAPE_SOFTMAX_TEMP)
    dist = dict(zip(SESSION_SHAPES, probs))
    top = max(dist, key=dist.get)
    show = session_minutes >= SESSION_SHAPE_WARMUP_MINUTES
    return {
        "probs": {k: round(v, 3) for k, v in dist.items()},
        "top": top,
        "confidence": round(dist[top], 3),
        "raw": {k: round(v, 3) for k, v in raw.items()},
        "show_on_live_card": show,
    }
