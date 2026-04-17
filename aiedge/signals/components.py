"""Signal scoring components — urgency + uncertainty subsystems.

Each function returns a *raw* component score in a bounded range.
Downstream aggregators apply day-type weights (`context/daytype.py`),
combine components, and normalize to the 0-10 urgency / uncertainty
scales.

All functions are pure: they consume a 5-minute OHLCV DataFrame and
(sometimes) a direction / prior-close / spike-bar count. No I/O, no
global state.

Phase 3f-1 carved the 15 urgency scorers + `_find_first_pullback` helper.
Phase 3f-2 adds the uncertainty scorers (`_score_uncertainty`,
`_score_two_sided_ratio`, `_score_liquidity_gaps`) and the
`_check_liquidity` hard-gate helper.
"""

from typing import Optional

import numpy as np
import pandas as pd

from aiedge.context.daytype import (
    SPIKE_MIN_BARS,
    STRONG_BODY_RATIO,
    _compute_two_sided_ratio,
)
from aiedge.features.candles import (
    DOJI_BODY_RATIO,
    MIN_RANGE,
    _body,
    _body_ratio,
    _close_position,
    _is_bear,
    _is_bull,
    _lower_tail_pct,
    _upper_tail_pct,
)
from aiedge.features.ema import EMA_PERIOD, _compute_ema
from aiedge.features.session import OPENING_RANGE_BARS
from aiedge.features.swings import _find_swing_highs, _find_swing_lows


# =============================================================================
# TUNABLE CONSTANTS
# =============================================================================

# ── Spike Quality (component 1, up to 4 pts raw) ──
CLOSE_TOP_PCT = 0.25           # close must be in top 25% of range (bull) or bottom 25% (bear)
TAIL_MAX_PCT = 0.20            # max lower tail (bull) or upper tail (bear) as % of range
MAX_SPIKE_BARS_SCORED = 4      # cap spike quality scoring at this many bars
MICRO_GAP_BONUS = 0.50         # bonus per micro gap (gap between bar bodies)

# ── Gap Integrity (component 2, -2 to +2 pts raw) ──
GAP_PARTIAL_FILL_PCT = 0.50    # gap partially filled if price enters this % of gap
POST_FILL_EVAL_BARS = 6        # bars after gap-fill bar to evaluate recovery structure
GAP_RECOVERY_STRONG_PCT = 0.50 # recovery >= 50% of gap = strong
GAP_RECOVERY_STRONG_BULL = 0.60  # bull-bar ratio threshold for strong recovery
GAP_RECOVERY_PARTIAL_PCT = 0.25  # recovery >= 25% = partial
GAP_RECOVERY_PARTIAL_BULL = 0.50 # bull-bar ratio threshold for partial recovery
# False disables the post-fill-eval branch and restores blanket -2.0 on any filled gap.
GAP_INTEGRITY_POST_FILL_EVAL = True

# ── Pullback Quality (component 3, -1 to +2 pts raw) ──
SHALLOW_PULLBACK_PCT = 0.33    # pullback < 33% of spike = shallow
MODERATE_PULLBACK_PCT = 0.50   # pullback 33-50% = moderate
DEEP_PULLBACK_PCT = 0.66       # pullback > 66% = failing
NO_PULLBACK_BAR_THRESHOLD = 5  # bars in spike with no pullback = urgency signal

# ── Tail Quality (component 5, -0.5 to +1 pt raw) ──
TAIL_CLEAN_PCT = 0.10          # "wrong-side" tail < 10% of range = no tail = urgency
TAIL_CLEAN_RATIO = 0.75        # if this fraction of spike bars have clean tails, +1
TAIL_BAD_PCT = 0.30            # tail > 30% of range on wrong side = penalty trigger

# ── Body Gaps (component 6, up to +1 pt raw) ──
BODY_GAP_BONUS = 0.50          # bonus per body gap between consecutive bars, cap 1.0

# ── MA Separation (component 7, up to +1 pt raw) ──
MA_GAP_BARS_STRONG = 10        # 10+ bars above/below EMA = +1
MA_GAP_BARS_MODERATE = 5       # 5-9 bars = +0.5

# ── Failed Counter-Setups (component 8, up to +1 pt raw) ──
FAILED_SETUP_BONUS = 0.50      # bonus per failed counter-setup, cap 1.0

# ── Volume Confirmation (component 9, up to +1 pt raw) ──
VOLUME_SPIKE_EXTREME = 5.0     # spike vol >= 5x average = +1
VOLUME_SPIKE_STRONG = 3.0      # spike vol >= 3x average = +0.5

# ── Trending Swings (component 10, -1 to +2 pts raw) ──
TRENDING_SWINGS_STRONG = 4     # 4+ consecutive HH/HL or LH/LL = +2
TRENDING_SWINGS_MODERATE = 2   # 2-3 = +1

# ── Spike Duration (component 11, -0.5 to +2 pts raw) ──
SPIKE_DURATION_STRONG = 8      # 8+ bars with no 25% retrace = +2
SPIKE_DURATION_MODERATE = 5    # 5-7 = +1
SPIKE_RETRACE_LIMIT = 0.25     # max retrace per bar relative to move so far

# ── Small Pullback Trend (SPT) — urgency component (up to 3.0 raw) ──
# Detects calm, drifting trends with shallow pullbacks — the kind that ranks
# high on a low-ADR tech day and is exactly what the scanner should surface.
SPT_LOOKBACK_BARS = 15
SPT_TREND_BODY_RATIO = 0.40    # body/range threshold for "trend bar" in SPT
SPT_DEPTH_SHALLOW = 0.25       # all pullbacks below this → perfect
SPT_DEPTH_MODERATE = 0.40      # avg below this → strong

# ── Normalization (urgency) ──
URGENCY_RAW_MAX = 29.0         # 26 old + 3 small_pullback_trend (SPT, new)

# =============================================================================
# UNCERTAINTY CONSTANTS
# =============================================================================

# ── Normalization (uncertainty) ──
UNCERTAINTY_RAW_MAX = 22.0     # raised: 17 old + 3 two-sided + 2 day-type adjustments

# ── Uncertainty thresholds ──
COLOR_ALT_HIGH = 0.60          # alternation rate above this = high uncertainty
DOJI_RATIO_HIGH = 0.40         # doji ratio above this = high uncertainty
BODY_OVERLAP_HIGH = 0.50       # overlap ratio above this = trading range behavior
BEAR_SPIKE_RATIO = 0.75        # bear spike >= 75% of bull spike = two-sided
BARS_STUCK_THRESHOLD = 10      # bars since last new high/low = stuck
MIDPOINT_TOLERANCE = 0.20      # within 20% of center = no one winning
STRONG_TREND_WINDOW = 5        # look at last N bars for always-in detection
UNCERTAINTY_ANALYSIS_WINDOW = 15  # bars to analyze for uncertainty metrics

# ── Reversal Count (uncertainty component 9, up to +2 pts) ──
REVERSAL_HIGH = 4              # 4+ reversals in window = +2
REVERSAL_MODERATE = 3          # 3 reversals = +1

# ── Tight Range (uncertainty component 11, up to +2 pts) ──
TIGHT_RANGE_PCT = 0.005        # highs and lows within 0.5% of each other
TIGHT_RANGE_BARS_STRONG = 10   # 10+ bars in tight range = +2
TIGHT_RANGE_BARS_MODERATE = 6  # 6-9 bars = +1

# ── MA Wrong-Side Closes (uncertainty component 12, +1 pt) ──
# The "last 2 bars on wrong side of EMA" check is implemented inline
# below (uses closes[-1] and closes[-2] directly — the 2 is not a tunable).

# ── Two-Sided Trading Ratio (uncertainty component 13, up to +3 pts) ──
TWO_SIDED_VERY_HIGH = 0.45     # > 45% countertrend bars = +3
TWO_SIDED_HIGH = 0.35          # 35-45% = +2
TWO_SIDED_MODERATE = 0.25      # 25-35% = +1

# ── Liquidity Filter (hard gate in scan_universe) ──
LIQUIDITY_MIN_DOLLAR_VOL = 350_000  # min avg dollar volume per 5-min bar ($350K)
LIQUIDITY_SKIP_BARS = 3             # skip first 3 bars (opening auction noise)

# ── Liquidity Gaps (uncertainty component 14, up to +2 pts) ──
# Penalizes mid-session bar-to-bar price gaps (illiquid jumps)
LIQUIDITY_GAP_PCT = 0.003           # gap > 0.3% of price between consecutive bars
LIQUIDITY_GAPS_HIGH = 5             # 5+ mid-session gaps = +2 uncertainty
LIQUIDITY_GAPS_MODERATE = 3         # 3-4 gaps = +1
LIQUIDITY_GAPS_LOW = 1              # 1-2 gaps = +0.5


# =============================================================================
# URGENCY COMPONENTS
# =============================================================================

def _score_spike_quality(df: pd.DataFrame, gap_direction: str) -> tuple[float, int]:
    """Component 1: Score the initial spike from market open (up to 4 pts raw).

    Returns (spike_quality_score, spike_bar_count).
    """
    is_trend_bar = _is_bull if gap_direction == "up" else _is_bear
    close_check = lambda row: _close_position(row) >= (1 - CLOSE_TOP_PCT)
    tail_check = lambda row: _lower_tail_pct(row) < TAIL_MAX_PCT

    if gap_direction == "down":
        close_check = lambda row: _close_position(row) <= CLOSE_TOP_PCT
        tail_check = lambda row: _upper_tail_pct(row) < TAIL_MAX_PCT

    score = 0.0
    spike_bars = 0

    for i in range(min(len(df), MAX_SPIKE_BARS_SCORED + 3)):
        row = df.iloc[i]
        if not is_trend_bar(row):
            break

        spike_bars += 1
        bar_score = 0.0

        if _body_ratio(row) > STRONG_BODY_RATIO:
            bar_score += 0.5
        if close_check(row):
            bar_score += 0.25
        if tail_check(row):
            bar_score += 0.25

        score += min(bar_score, 1.0)

        if spike_bars >= MAX_SPIKE_BARS_SCORED:
            for j in range(i + 1, len(df)):
                if is_trend_bar(df.iloc[j]):
                    spike_bars += 1
                else:
                    break
            break

    # Micro gap bonus: low[i] >= high[i-2] for bull
    for i in range(2, min(spike_bars, len(df))):
        if gap_direction == "up":
            if df.iloc[i]["low"] >= df.iloc[i - 2]["high"]:
                score += MICRO_GAP_BONUS
        else:
            if df.iloc[i]["high"] <= df.iloc[i - 2]["low"]:
                score += MICRO_GAP_BONUS

    return min(score, 4.0), spike_bars


def _score_gap_integrity(df: pd.DataFrame, prior_close: float, gap_direction: str) -> tuple[float, str]:
    """Component 2: Gap integrity (-2 to +2 pts raw).

    Returns (score, fill_status) where fill_status is one of:
      "held"             — gap never touched
      "partial_fill"     — filled <50% of gap
      "filled_recovered" — filled >50% but price recovered directionally
      "filled_failed"    — filled >50% with no meaningful recovery
    """
    gap_open = df.iloc[0]["open"]

    if gap_direction == "up":
        gap_size = gap_open - prior_close
        if gap_size <= 0:
            return 0.0, "held"
        day_low = df["low"].min()
        if day_low > prior_close:
            return 2.0, "held"
        if day_low > prior_close - (gap_size * GAP_PARTIAL_FILL_PCT):
            return 1.0, "partial_fill"
        # Gap filled >50% — evaluate post-fill structure
        if not GAP_INTEGRITY_POST_FILL_EVAL:
            return -2.0, "filled_failed"
        fill_bar_idx = int(df["low"].idxmin())
        post_fill = df.iloc[fill_bar_idx + 1 : fill_bar_idx + 1 + POST_FILL_EVAL_BARS]
        if len(post_fill) < 2:
            return -2.0, "filled_failed"
        recovery_close = df.iloc[-1]["close"]
        recovery_pct = (recovery_close - prior_close) / gap_size if gap_size > 0 else 0.0
        bull_bars = int((post_fill["close"] > post_fill["open"]).sum())
        bull_ratio = bull_bars / len(post_fill)
        if recovery_pct >= GAP_RECOVERY_STRONG_PCT and bull_ratio >= GAP_RECOVERY_STRONG_BULL:
            return 0.0, "filled_recovered"
        if recovery_pct >= GAP_RECOVERY_PARTIAL_PCT and bull_ratio >= GAP_RECOVERY_PARTIAL_BULL:
            return -0.5, "filled_recovered"
        return -2.0, "filled_failed"
    else:
        gap_size = prior_close - gap_open
        if gap_size <= 0:
            return 0.0, "held"
        day_high = df["high"].max()
        if day_high < prior_close:
            return 2.0, "held"
        if day_high < prior_close + (gap_size * GAP_PARTIAL_FILL_PCT):
            return 1.0, "partial_fill"
        # Gap filled >50% — evaluate post-fill structure
        if not GAP_INTEGRITY_POST_FILL_EVAL:
            return -2.0, "filled_failed"
        fill_bar_idx = int(df["high"].idxmax())
        post_fill = df.iloc[fill_bar_idx + 1 : fill_bar_idx + 1 + POST_FILL_EVAL_BARS]
        if len(post_fill) < 2:
            return -2.0, "filled_failed"
        recovery_close = df.iloc[-1]["close"]
        recovery_pct = (prior_close - recovery_close) / gap_size if gap_size > 0 else 0.0
        bear_bars = int((post_fill["close"] < post_fill["open"]).sum())
        bear_ratio = bear_bars / len(post_fill)
        if recovery_pct >= GAP_RECOVERY_STRONG_PCT and bear_ratio >= GAP_RECOVERY_STRONG_BULL:
            return 0.0, "filled_recovered"
        if recovery_pct >= GAP_RECOVERY_PARTIAL_PCT and bear_ratio >= GAP_RECOVERY_PARTIAL_BULL:
            return -0.5, "filled_recovered"
        return -2.0, "filled_failed"


def _find_first_pullback(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> tuple[float, float, bool]:
    """Component 3: Pullback quality (-1 to +2 pts raw). Returns (depth_pct, score, exists)."""
    if spike_bars < 1 or len(df) <= spike_bars:
        if spike_bars >= NO_PULLBACK_BAR_THRESHOLD:
            return 0.0, 1.0, False
        return 0.0, 0.0, False

    if gap_direction == "up":
        spike_low = df.iloc[0]["low"]
        spike_high = df.iloc[:spike_bars]["high"].max()
        spike_height = spike_high - spike_low
        if spike_height <= MIN_RANGE:
            return 0.0, 0.0, False

        post_spike = df.iloc[spike_bars:]
        if len(post_spike) == 0:
            return (0.0, 1.0, False) if spike_bars >= NO_PULLBACK_BAR_THRESHOLD else (0.0, 0.0, False)

        pullback_low = post_spike["low"].min()
        pullback_pct = (spike_high - pullback_low) / spike_height
        holds_above_open = pullback_low >= df.iloc[0]["open"]
    else:
        spike_high = df.iloc[0]["high"]
        spike_low = df.iloc[:spike_bars]["low"].min()
        spike_height = spike_high - spike_low
        if spike_height <= MIN_RANGE:
            return 0.0, 0.0, False

        post_spike = df.iloc[spike_bars:]
        if len(post_spike) == 0:
            return (0.0, 1.0, False) if spike_bars >= NO_PULLBACK_BAR_THRESHOLD else (0.0, 0.0, False)

        pullback_high = post_spike["high"].max()
        pullback_pct = (pullback_high - spike_low) / spike_height
        holds_above_open = pullback_high <= df.iloc[0]["open"]

    if pullback_pct < SHALLOW_PULLBACK_PCT:
        score = 2.0
    elif pullback_pct < MODERATE_PULLBACK_PCT:
        score = 1.0
    elif pullback_pct < DEEP_PULLBACK_PCT:
        score = 0.0
    else:
        score = -1.0

    if holds_above_open:
        score += 0.5

    return pullback_pct, min(score, 2.0), True


def _score_follow_through(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> float:
    """Component 4: Follow-through after pullback (-1.5 to +2 pts raw)."""
    if spike_bars < 1 or len(df) <= spike_bars + 2:
        return 0.0

    if gap_direction == "up":
        spike_high = df.iloc[:spike_bars]["high"].max()
        post_spike = df.iloc[spike_bars:]
        pullback_low_idx = post_spike["low"].idxmin()
        if pullback_low_idx is None:
            return 0.0
        pullback_pos = df.index.get_loc(pullback_low_idx)
        after_pullback = df.iloc[pullback_pos + 1:] if pullback_pos + 1 < len(df) else pd.DataFrame()
        if len(after_pullback) == 0:
            return 0.0

        score = 0.0
        pullback_low_val = df.loc[pullback_low_idx, "low"]
        if pullback_low_val > df.iloc[0]["low"]:
            score += 1.0
        if after_pullback["high"].max() > spike_high:
            score += 1.0
        elif after_pullback["high"].max() < spike_high:
            score -= 0.5
        if after_pullback["low"].min() < pullback_low_val:
            score -= 1.0
    else:
        spike_low = df.iloc[:spike_bars]["low"].min()
        post_spike = df.iloc[spike_bars:]
        pullback_high_idx = post_spike["high"].idxmax()
        if pullback_high_idx is None:
            return 0.0
        pullback_pos = df.index.get_loc(pullback_high_idx)
        after_pullback = df.iloc[pullback_pos + 1:] if pullback_pos + 1 < len(df) else pd.DataFrame()
        if len(after_pullback) == 0:
            return 0.0

        score = 0.0
        pullback_high_val = df.loc[pullback_high_idx, "high"]
        if pullback_high_val < df.iloc[0]["high"]:
            score += 1.0
        if after_pullback["low"].min() < spike_low:
            score += 1.0
        elif after_pullback["low"].min() > spike_low:
            score -= 0.5
        if after_pullback["high"].max() > pullback_high_val:
            score -= 1.0

    return max(min(score, 2.0), -1.5)


def _score_tail_quality(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> float:
    """Component 5: Tail quality in the spike (-0.5 to +1 pt raw).

    No "wrong-side" tails = extreme urgency. Large wrong-side tails = hesitation.
    """
    if spike_bars < 1:
        return 0.0

    n = min(spike_bars, len(df))
    clean_count = 0
    bad_count = 0

    for i in range(n):
        row = df.iloc[i]
        if gap_direction == "up":
            # Wrong-side tail for bull = upper tail (sellers rejecting highs)
            wrong_tail = _upper_tail_pct(row)
        else:
            # Wrong-side tail for bear = lower tail (buyers rejecting lows)
            wrong_tail = _lower_tail_pct(row)

        if wrong_tail < TAIL_CLEAN_PCT:
            clean_count += 1
        if wrong_tail > TAIL_BAD_PCT:
            bad_count += 1

    score = 0.0
    if n > 0 and (clean_count / n) >= TAIL_CLEAN_RATIO:
        score += 1.0
    if bad_count >= 2:
        score -= 0.5

    return max(min(score, 1.0), -0.5)


def _score_body_gaps(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> float:
    """Component 6: Gaps between consecutive bar bodies (up to +1 pt raw).

    bar[i].open > bar[i-1].close for bull = body gap = sign of strength.
    """
    if spike_bars < 2:
        return 0.0

    n = min(spike_bars, len(df))
    gap_count = 0

    for i in range(1, n):
        if gap_direction == "up":
            if df.iloc[i]["open"] > df.iloc[i - 1]["close"]:
                gap_count += 1
        else:
            if df.iloc[i]["open"] < df.iloc[i - 1]["close"]:
                gap_count += 1

    return min(gap_count * BODY_GAP_BONUS, 1.0)


def _score_ma_separation(df: pd.DataFrame, gap_direction: str) -> float:
    """Component 7: MA separation / gap bars (up to +1 pt raw).

    Count consecutive bars where low (bull) stays above 20 EMA. Brooks calls
    this "20 moving average gap bars" — extreme trend strength.
    """
    if len(df) < EMA_PERIOD:
        return 0.0  # not enough data for meaningful EMA

    closes = df["close"].values.astype(float)
    ema = _compute_ema(closes, EMA_PERIOD)

    consecutive = 0
    max_consecutive = 0

    for i in range(EMA_PERIOD, len(df)):
        if gap_direction == "up":
            if df.iloc[i]["low"] > ema[i]:
                consecutive += 1
            else:
                max_consecutive = max(max_consecutive, consecutive)
                consecutive = 0
        else:
            if df.iloc[i]["high"] < ema[i]:
                consecutive += 1
            else:
                max_consecutive = max(max_consecutive, consecutive)
                consecutive = 0

    max_consecutive = max(max_consecutive, consecutive)

    if max_consecutive >= MA_GAP_BARS_STRONG:
        return 1.0
    elif max_consecutive >= MA_GAP_BARS_MODERATE:
        return 0.5
    return 0.0


def _score_failed_counter_setups(df: pd.DataFrame, gap_direction: str) -> float:
    """Component 8: Failed counter-setups (up to +1 pt raw).

    Detect when a counter-trend signal bar forms but the next bar's breakout
    fails. Each failure = sign of strength for the trend.
    """
    if len(df) < 3:
        return 0.0

    failures = 0

    for i in range(1, len(df) - 1):
        signal_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]

        if gap_direction == "up":
            # Bear signal bar: bar that closes below its open (bear bar)
            if _is_bear(signal_bar) and _body_ratio(signal_bar) > 0.4:
                # Failed: next bar's high exceeds signal bar's high (short entry fails)
                if next_bar["high"] > signal_bar["high"]:
                    failures += 1
        else:
            # Bull signal bar in bear context
            if _is_bull(signal_bar) and _body_ratio(signal_bar) > 0.4:
                if next_bar["low"] < signal_bar["low"]:
                    failures += 1

    return min(failures * FAILED_SETUP_BONUS, 1.0)


def _score_volume_confirmation(df: pd.DataFrame, spike_bars: int) -> float:
    """Component 9: Volume confirmation (up to +1 pt raw).

    Compare spike bar volume to the average volume. "If the volume of the
    large breakout bar is 10 to 20 times the average volume, the chance of
    a measured move increases." — Brooks
    """
    if "volume" not in df.columns or spike_bars < 1:
        return 0.0

    volumes = df["volume"].values.astype(float)
    if np.all(volumes == 0):
        return 0.0

    spike_vol = np.mean(volumes[:min(spike_bars, len(df))])
    # Use bars after spike for baseline, or all bars if not enough
    if len(df) > spike_bars + 5:
        baseline_vol = np.mean(volumes[spike_bars:])
    else:
        baseline_vol = np.mean(volumes)

    if baseline_vol <= 0:
        return 0.0

    ratio = spike_vol / baseline_vol

    if ratio >= VOLUME_SPIKE_EXTREME:
        return 1.0
    elif ratio >= VOLUME_SPIKE_STRONG:
        return 0.5
    return 0.0


def _score_majority_trend_bars(df: pd.DataFrame, gap_direction: str) -> float:
    """H1.2 — Majority trend bars in direction (-1 to +2 raw).

    THE most fundamental measure: are most bars going the same way?
    Brooks: "Most of the bars are trend bars in the direction of the trend."
    """
    n = len(df)
    if n < 3:
        return 0.0

    is_trend = _is_bull if gap_direction == "up" else _is_bear
    # Count bars with body > 50% of range AND in the trend direction
    trend_count = 0
    for i in range(n):
        row = df.iloc[i]
        if is_trend(row) and _body_ratio(row) > 0.50:
            trend_count += 1

    ratio = trend_count / n

    if ratio > 0.70:
        return 2.0
    elif ratio > 0.55:
        return 1.0
    elif ratio > 0.40:
        return 0.0
    return -1.0


def _score_micro_gaps(df: pd.DataFrame, gap_direction: str) -> float:
    """H1.6 — Micro measuring gaps (0 to +2 raw).

    For a strong trend bar at index i, if the bar before and bar after don't
    overlap, that's a micro gap — extreme sign of strength.
    Brooks: "If the low of the bar after a strong bull trend bar is at or above
    the high of the bar before the trend bar, this is a gap."
    """
    if len(df) < 3:
        return 0.0

    is_trend = _is_bull if gap_direction == "up" else _is_bear
    gap_count = 0

    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        # Must be a strong trend bar
        if not (is_trend(bar) and _body_ratio(bar) > STRONG_BODY_RATIO):
            continue

        prev_bar = df.iloc[i - 1]
        next_bar = df.iloc[i + 1]

        if gap_direction == "up":
            # Micro gap: low of bar after >= high of bar before
            if next_bar["low"] >= prev_bar["high"]:
                gap_count += 1
        else:
            # Bear: high of bar after <= low of bar before
            if next_bar["high"] <= prev_bar["low"]:
                gap_count += 1

    if gap_count >= 3:
        return 2.0
    elif gap_count == 2:
        return 1.5
    elif gap_count == 1:
        return 1.0
    return 0.0


def _score_trending_everything(df: pd.DataFrame, gap_direction: str) -> float:
    """H1.16 — Trending 'anything': closes, highs, lows, bodies (-1 to +2 raw).

    Compute linear regression slope on closes, highs, lows, and body midpoints.
    If all 4 slope in the trend direction with decent R², strong trend signal.
    Brooks: "It has trending 'anything': closes, highs, lows, or bodies."
    """
    n = len(df)
    if n < 5:
        return 0.0

    x = np.arange(n, dtype=float)
    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    body_mids = ((df["open"].values + df["close"].values) / 2).astype(float)

    trending_count = 0

    for series in [closes, highs, lows, body_mids]:
        # Simple linear regression: slope and R²
        x_mean = np.mean(x)
        y_mean = np.mean(series)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (series - y_mean))
        ss_yy = np.sum((series - y_mean) ** 2)

        if ss_xx < 1e-10 or ss_yy < 1e-10:
            continue

        slope = ss_xy / ss_xx
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

        # Check direction and fit quality
        if gap_direction == "up" and slope > 0 and r_squared > 0.5:
            trending_count += 1
        elif gap_direction == "down" and slope < 0 and r_squared > 0.5:
            trending_count += 1

    if trending_count == 4:
        return 2.0
    elif trending_count == 3:
        return 1.0
    elif trending_count == 2:
        return 0.0
    return -1.0


def _score_levels_broken(df: pd.DataFrame, gap_direction: str,
                         prior_close: float) -> float:
    """H1.14 — Trend breaks multiple levels (0 to +2 raw).

    Count significant levels the trend has broken through.
    Brooks: "The trend goes very far and breaks several resistance levels,
    like the moving average, prior swing highs, and trend lines."
    """
    n = len(df)
    if n < 5:
        return 0.0

    current_price = df.iloc[-1]["close"]
    levels_broken = 0

    # 1. Prior swing highs (bull) or swing lows (bear) that have been exceeded
    if gap_direction == "up":
        swing_highs = _find_swing_highs(df)
        for _, sh_price in swing_highs:
            if current_price > sh_price:
                levels_broken += 1
    else:
        swing_lows = _find_swing_lows(df)
        for _, sl_price in swing_lows:
            if current_price < sl_price:
                levels_broken += 1

    # 2. 20-bar EMA — has price broken above (bull) or below (bear)?
    if n >= EMA_PERIOD:
        closes = df["close"].values.astype(float)
        ema = _compute_ema(closes, EMA_PERIOD)
        if gap_direction == "up" and current_price > ema[-1]:
            levels_broken += 1
        elif gap_direction == "down" and current_price < ema[-1]:
            levels_broken += 1

    # 3. Prior day's close
    if gap_direction == "up" and current_price > prior_close:
        levels_broken += 1
    elif gap_direction == "down" and current_price < prior_close:
        levels_broken += 1

    # 4. Opening range high/low
    or_high = df.iloc[:min(OPENING_RANGE_BARS, n)]["high"].max()
    or_low = df.iloc[:min(OPENING_RANGE_BARS, n)]["low"].min()
    if gap_direction == "up" and current_price > or_high:
        levels_broken += 1
    elif gap_direction == "down" and current_price < or_low:
        levels_broken += 1

    if levels_broken >= 4:
        return 2.0
    elif levels_broken == 3:
        return 1.5
    elif levels_broken == 2:
        return 1.0
    elif levels_broken == 1:
        return 0.5
    return 0.0


def _score_small_pullback_trend(df: pd.DataFrame, direction: str) -> float:
    """Small Pullback Trend (SPT) — urgency component, 0.0 to 3.0 raw.

    Detects a calm, drifting trend with shallow pullbacks: the kind of tape that
    shows up on a low-ADR tech day where price just grinds one direction with
    two-bar dips. Scores five sub-checks (each 0-1), weighted sum → scaled to 3.0.

    Sub-checks:
      1. Trend-bar density (w 0.8) — fraction of last N bars that are trend bars
         in direction (body ≥ SPT_TREND_BODY_RATIO of range)
      2. Pullback depth     (w 0.8) — pullback extreme vs prior trend-leg height
      3. No broken swings   (w 0.6) — does any bar break the prior higher-low (bull)?
      4. Higher-closes run  (w 0.4) — longest streak of closes > prior close
      5. Pullback tail dens (w 0.4) — of non-trend bars, fraction with a deep
         bottom tail (bull) / top tail (bear)

    Aggregation: weighted_sum (max = 3.0), clamped.
    """
    direction = direction.lower()
    if direction not in ("up", "down"):
        return 0.0

    n = min(len(df), SPT_LOOKBACK_BARS)
    if n < 4:
        return 0.0

    window = df.iloc[-n:].reset_index(drop=True)
    bull = direction == "up"

    # ── Classify each bar in window ──
    # trend_mask[i] = bar i is a trend bar in direction with body >= SPT_TREND_BODY_RATIO
    ranges = (window["high"] - window["low"]).clip(lower=MIN_RANGE).values
    bodies = (window["close"] - window["open"]).values  # signed
    abs_bodies = np.abs(bodies)
    body_ratio = abs_bodies / ranges

    if bull:
        signed_ok = bodies > 0
    else:
        signed_ok = bodies < 0
    trend_mask = signed_ok & (body_ratio >= SPT_TREND_BODY_RATIO)

    # ── Sub-check 1: trend-bar density ──
    density = float(trend_mask.sum()) / float(n)
    if density >= 0.70:
        s1 = 1.0
    elif density >= 0.55:
        s1 = 0.7
    elif density >= 0.40:
        s1 = 0.3
    else:
        s1 = 0.0

    # Hard gate: below the density floor the window is chop — SPT is not meaningful.
    # Zero-out early so shallow-pullback noise in a chop window doesn't leak points.
    if s1 == 0.0:
        return 0.0

    # ── Sub-check 2: pullback depth ──
    # A "pullback" = a contiguous run of non-trend bars between trend-bar runs.
    # Depth = (prior-leg-top  - pullback-low) / prior-leg-height     (bull)
    #        (pullback-high   - prior-leg-bot) / prior-leg-height     (bear)
    pullback_depths: list[float] = []

    # Identify leg starts (trend run boundaries)
    legs: list[tuple[int, int]] = []  # (start, end) inclusive of trend runs
    i = 0
    while i < n:
        if trend_mask[i]:
            j = i
            while j < n and trend_mask[j]:
                j += 1
            legs.append((i, j - 1))
            i = j
        else:
            i += 1

    # Between consecutive legs there is a pullback (the gap between leg_k.end+1 and leg_k+1.start-1)
    for k in range(len(legs) - 1):
        leg_start, leg_end = legs[k]
        next_start = legs[k + 1][0]
        pb_s = leg_end + 1
        pb_e = next_start - 1
        if pb_s > pb_e:
            continue

        if bull:
            leg_top = float(window["high"].iloc[leg_start:leg_end + 1].max())
            leg_bot = float(window["low"].iloc[leg_start:leg_end + 1].min())
            leg_height = max(leg_top - leg_bot, MIN_RANGE)
            pb_low = float(window["low"].iloc[pb_s:pb_e + 1].min())
            depth = (leg_top - pb_low) / leg_height
        else:
            leg_top = float(window["high"].iloc[leg_start:leg_end + 1].max())
            leg_bot = float(window["low"].iloc[leg_start:leg_end + 1].min())
            leg_height = max(leg_top - leg_bot, MIN_RANGE)
            pb_high = float(window["high"].iloc[pb_s:pb_e + 1].max())
            depth = (pb_high - leg_bot) / leg_height
        pullback_depths.append(max(depth, 0.0))

    if not pullback_depths:
        # No pullbacks observed. If density is strong this is a pure trend (reward).
        # If density is weak it means no trend structure at all — cap by s1 so chop
        # doesn't get the "perfect depth" bonus for lacking trend legs.
        s2 = 1.0 if s1 >= 0.7 else s1
    elif all(d < SPT_DEPTH_SHALLOW for d in pullback_depths):
        s2 = 1.0
    elif (sum(pullback_depths) / len(pullback_depths)) < SPT_DEPTH_MODERATE:
        s2 = 0.6
    elif max(pullback_depths) > 0.60:
        s2 = 0.0
    elif (sum(pullback_depths) / len(pullback_depths)) < 0.60:
        s2 = 0.3
    else:
        s2 = 0.0

    # ── Sub-check 3: no broken swing points ──
    # For bull: higher-low = a swing low that is above the prior swing low.
    # Ask: does any subsequent bar break (close below / low-break) that higher-low?
    # Zero breaks → 1.0; one break reclaimed within 2 bars → 0.5; any unreclaimed break → 0.
    breaks_total = 0
    breaks_reclaimed = 0
    breaks_unreclaimed = 0

    higher_lows: list[tuple[int, float]] = []
    lower_highs: list[tuple[int, float]] = []

    if bull:
        # Find swing lows: low[i] < low[i-1] and low[i] < low[i+1]
        swing_lows: list[tuple[int, float]] = []
        for i in range(1, n - 1):
            if window["low"].iloc[i] < window["low"].iloc[i - 1] and window["low"].iloc[i] < window["low"].iloc[i + 1]:
                swing_lows.append((i, float(window["low"].iloc[i])))
        # "Higher lows" = chain where each swing_low > prior swing_low
        prev_lvl = -1e18
        for idx, lvl in swing_lows:
            if lvl > prev_lvl:
                higher_lows.append((idx, lvl))
                prev_lvl = lvl
        # Look for breaks of the most-recent higher-low at each later bar
        for hl_idx, hl_price in higher_lows:
            # Scan bars after hl_idx until the next higher-low (exclusive)
            next_hl_idx = n
            for nxt_idx, _ in higher_lows:
                if nxt_idx > hl_idx:
                    next_hl_idx = nxt_idx
                    break
            broke_at: Optional[int] = None
            for j in range(hl_idx + 1, next_hl_idx):
                if window["low"].iloc[j] < hl_price:
                    broke_at = j
                    break
            if broke_at is not None:
                breaks_total += 1
                # Reclaim = within 2 bars, low rises back above hl_price
                reclaimed = False
                for k in range(broke_at + 1, min(broke_at + 3, n)):
                    if window["low"].iloc[k] >= hl_price:
                        reclaimed = True
                        break
                if reclaimed:
                    breaks_reclaimed += 1
                else:
                    breaks_unreclaimed += 1
    else:
        # Symmetric: break of a lower-high (a swing high lower than the prior swing high)
        swing_highs: list[tuple[int, float]] = []
        for i in range(1, n - 1):
            if window["high"].iloc[i] > window["high"].iloc[i - 1] and window["high"].iloc[i] > window["high"].iloc[i + 1]:
                swing_highs.append((i, float(window["high"].iloc[i])))
        prev_lvl = 1e18
        for idx, lvl in swing_highs:
            if lvl < prev_lvl:
                lower_highs.append((idx, lvl))
                prev_lvl = lvl
        for lh_idx, lh_price in lower_highs:
            next_lh_idx = n
            for nxt_idx, _ in lower_highs:
                if nxt_idx > lh_idx:
                    next_lh_idx = nxt_idx
                    break
            broke_at = None
            for j in range(lh_idx + 1, next_lh_idx):
                if window["high"].iloc[j] > lh_price:
                    broke_at = j
                    break
            if broke_at is not None:
                breaks_total += 1
                reclaimed = False
                for k in range(broke_at + 1, min(broke_at + 3, n)):
                    if window["high"].iloc[k] <= lh_price:
                        reclaimed = True
                        break
                if reclaimed:
                    breaks_reclaimed += 1
                else:
                    breaks_unreclaimed += 1

    # s3 logic:
    #  - If we have a meaningful higher-low / lower-high chain (≥2 swings) and
    #    zero breaks → real "structure intact" signal (1.0).
    #  - If we found ≥2 swings and one break reclaimed within 2 bars → 0.5.
    #  - Any unreclaimed break → 0.
    #  - If we lack structure to judge (no swings found), don't reward — cap by
    #    s1 so chop doesn't collect free points for lacking any swing points.
    swing_chain = higher_lows if bull else lower_highs
    have_structure = len(swing_chain) >= 2
    if not have_structure:
        s3 = min(1.0, s1)
    elif breaks_total == 0:
        s3 = 1.0
    elif breaks_unreclaimed == 0 and breaks_reclaimed == 1:
        s3 = 0.5
    else:
        s3 = 0.0

    # ── Sub-check 4: higher-closes streak (bull) / lower-closes (bear) ──
    closes = window["close"].values
    longest = 0
    cur = 0
    for i in range(1, n):
        if (bull and closes[i] > closes[i - 1]) or ((not bull) and closes[i] < closes[i - 1]):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    if longest >= 6:
        s4 = 1.0
    elif longest >= 4:
        s4 = 0.6
    elif longest >= 2:
        s4 = 0.2
    else:
        s4 = 0.0

    # ── Sub-check 5: pullback-bar tail density ──
    non_trend_idxs = [i for i in range(n) if not trend_mask[i]]
    if not non_trend_idxs:
        # Pure trend — no pullback bars. Reward only if density confirms pure trend;
        # otherwise this branch can't fire in practice (no non-trend bars ⇒ density=1.0)
        s5 = 1.0
    else:
        deep_tail_count = 0
        for i in non_trend_idxs:
            row = window.iloc[i]
            if bull:
                tail = _lower_tail_pct(row)
            else:
                tail = _upper_tail_pct(row)
            if tail >= 0.30:
                deep_tail_count += 1
        frac = deep_tail_count / len(non_trend_idxs)
        if frac >= 0.60:
            s5 = 1.0
        elif frac >= 0.40:
            s5 = 0.6
        elif frac >= 0.20:
            s5 = 0.3
        else:
            s5 = 0.0

    # ── Aggregate: weighted sum, max = 3.0 ──
    weighted_sum = 0.8 * s1 + 0.8 * s2 + 0.6 * s3 + 0.4 * s4 + 0.4 * s5
    spt_score = min(weighted_sum, 3.0)
    return spt_score


def _score_trending_swings(df: pd.DataFrame, gap_direction: str) -> float:
    """Urgency component: Trending highs/lows (-1 to +2 raw).

    Count consecutive HH/HL (bull) or LH/LL (bear).
    """
    swing_lows = _find_swing_lows(df)
    swing_highs = _find_swing_highs(df)

    if len(swing_lows) < 2 or len(swing_highs) < 2:
        return 0.0

    if gap_direction == "up":
        # Count consecutive higher highs
        hh_count = 0
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i - 1][1]:
                hh_count += 1
            else:
                break

        # Count consecutive higher lows
        hl_count = 0
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i - 1][1]:
                hl_count += 1
            else:
                break

        trending = min(hh_count, hl_count)

        # Check for broken sequence (lower low)
        if len(swing_lows) >= 2 and swing_lows[-1][1] < swing_lows[-2][1]:
            return -1.0
    else:
        lh_count = 0
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] < swing_highs[i - 1][1]:
                lh_count += 1
            else:
                break

        ll_count = 0
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] < swing_lows[i - 1][1]:
                ll_count += 1
            else:
                break

        trending = min(lh_count, ll_count)

        if len(swing_highs) >= 2 and swing_highs[-1][1] > swing_highs[-2][1]:
            return -1.0

    if trending >= TRENDING_SWINGS_STRONG:
        return 2.0
    elif trending >= TRENDING_SWINGS_MODERATE:
        return 1.0
    return 0.0


def _score_spike_duration(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> float:
    """Urgency component: Spike duration — how many bars the spike sustains
    without retracing > 25% of the move so far (-0.5 to +2 raw).
    """
    if spike_bars < 1:
        return -0.5

    # Count bars from start where no bar pulls back > 25% from the prior bar's extreme.
    # For bull: does bar[i]'s low dip significantly below bar[i-1]'s close?
    # Measure pullback relative to the total move from bar 0's low.
    if gap_direction == "up":
        sustained = 1  # bar 0 always counts
        for i in range(1, len(df)):
            move = df.iloc[i - 1]["close"] - df.iloc[0]["low"]
            if move <= MIN_RANGE:
                sustained += 1
                continue
            pullback = df.iloc[i - 1]["close"] - df.iloc[i]["low"]
            if pullback / move > SPIKE_RETRACE_LIMIT:
                break
            sustained += 1
    else:
        sustained = 1
        for i in range(1, len(df)):
            move = df.iloc[0]["high"] - df.iloc[i - 1]["close"]
            if move <= MIN_RANGE:
                sustained += 1
                continue
            pullback = df.iloc[i]["high"] - df.iloc[i - 1]["close"]
            if pullback / move > SPIKE_RETRACE_LIMIT:
                break
            sustained += 1

    if sustained >= SPIKE_DURATION_STRONG:
        return 2.0
    elif sustained >= SPIKE_DURATION_MODERATE:
        return 1.0
    elif sustained >= 3:
        return 0.0
    return -0.5


# =============================================================================
# UNCERTAINTY COMPONENTS + LIQUIDITY GATE
# =============================================================================

def _check_liquidity(df: pd.DataFrame) -> dict:
    """Hard liquidity gate for `scan_universe()`.

    Computes average dollar volume per 5-min bar (skipping the first 3 bars
    to avoid opening-auction noise). Returns a dict with pass/fail and the
    metrics. This is a gate, not a score, but it shares liquidity tunables
    with `_score_liquidity_gaps` so it lives here.
    """
    if "volume" not in df.columns or len(df) <= LIQUIDITY_SKIP_BARS:
        return {"passed": False, "avg_dollar_vol": 0.0, "bars_measured": 0}

    measured = df.iloc[LIQUIDITY_SKIP_BARS:]
    if len(measured) == 0:
        return {"passed": False, "avg_dollar_vol": 0.0, "bars_measured": 0}

    # Dollar volume per bar = bar's VWAP-ish price × volume
    # Use (high + low + close) / 3 as a typical price proxy
    typical_price = (measured["high"] + measured["low"] + measured["close"]) / 3
    dollar_vol_per_bar = (typical_price * measured["volume"]).values
    avg_dollar_vol = float(np.mean(dollar_vol_per_bar))

    return {
        "passed": avg_dollar_vol >= LIQUIDITY_MIN_DOLLAR_VOL,
        "avg_dollar_vol": round(avg_dollar_vol, 0),
        "bars_measured": len(measured),
    }


def _score_liquidity_gaps(df: pd.DataFrame) -> float:
    """Uncertainty component: penalizes mid-session bar-to-bar price gaps (0 to +2 raw).

    Illiquid stocks jump between bars — the close-to-open gap between consecutive
    bars is large relative to price. This makes price action unreliable.
    Skips first 3 bars (opening auction) since those gaps are normal.
    """
    if len(df) <= LIQUIDITY_SKIP_BARS + 1:
        return 0.0

    gap_count = 0
    measured = df.iloc[LIQUIDITY_SKIP_BARS:]

    for i in range(1, len(measured)):
        prev_close = measured.iloc[i - 1]["close"]
        curr_open = measured.iloc[i]["open"]
        if prev_close <= MIN_RANGE:
            continue
        gap_pct = abs(curr_open - prev_close) / prev_close
        if gap_pct > LIQUIDITY_GAP_PCT:
            gap_count += 1

    if gap_count >= LIQUIDITY_GAPS_HIGH:
        return 2.0
    elif gap_count >= LIQUIDITY_GAPS_MODERATE:
        return 1.0
    elif gap_count >= LIQUIDITY_GAPS_LOW:
        return 0.5
    return 0.0


def _score_two_sided_ratio(df: pd.DataFrame, gap_direction: str) -> float:
    """Uncertainty component: Two-sided trading ratio (up to +3 raw).

    High ratio of countertrend bars = more uncertainty. The ratio itself is
    computed in `context.daytype` because day-type classification needs the
    same metric.
    """
    ratio = _compute_two_sided_ratio(df, gap_direction)

    if ratio > TWO_SIDED_VERY_HIGH:
        return 3.0
    elif ratio > TWO_SIDED_HIGH:
        return 2.0
    elif ratio > TWO_SIDED_MODERATE:
        return 1.0
    return 0.0


def _score_uncertainty(df: pd.DataFrame, gap_direction: str) -> tuple[float, str]:
    """Score how confused/two-sided the chart is (12 components).

    Returns (raw_uncertainty_score, always_in_direction).
    """
    n = min(len(df), UNCERTAINTY_ANALYSIS_WINDOW)
    if n < 3:
        return UNCERTAINTY_RAW_MAX * 0.5, "unclear"

    recent = df.iloc[-n:].reset_index(drop=True)
    uncertainty = 0.0

    # ── 1. Color alternation rate (up to +3) ──
    color_changes = 0
    for i in range(1, len(recent)):
        if _is_bull(recent.iloc[i - 1]) != _is_bull(recent.iloc[i]):
            color_changes += 1
    alt_rate = color_changes / max(len(recent) - 1, 1)
    if alt_rate > COLOR_ALT_HIGH:
        uncertainty += 3.0

    # ── 2. Doji ratio (up to +2) ──
    doji_count = sum(1 for i in range(len(recent)) if _body_ratio(recent.iloc[i]) < DOJI_BODY_RATIO)
    if len(recent) > 0 and (doji_count / len(recent)) > DOJI_RATIO_HIGH:
        uncertainty += 2.0

    # ── 3. Body overlap ratio (up to +2) ──
    overlaps = []
    for i in range(1, len(recent)):
        prev_top = max(recent.iloc[i - 1]["open"], recent.iloc[i - 1]["close"])
        prev_bot = min(recent.iloc[i - 1]["open"], recent.iloc[i - 1]["close"])
        curr_top = max(recent.iloc[i]["open"], recent.iloc[i]["close"])
        curr_bot = min(recent.iloc[i]["open"], recent.iloc[i]["close"])
        overlap = max(0, min(prev_top, curr_top) - max(prev_bot, curr_bot))
        avg_body = (_body(recent.iloc[i - 1]) + _body(recent.iloc[i])) / 2
        if avg_body > MIN_RANGE:
            overlaps.append(overlap / avg_body)
    if overlaps and np.mean(overlaps) > BODY_OVERLAP_HIGH:
        uncertainty += 2.0

    # ── 4. Counter-spike present (up to +2) ──
    if gap_direction == "up":
        bull_move = df["high"].max() - df.iloc[0]["low"]
        bear_move = 0
        for i in range(len(df) - 2):
            if _is_bear(df.iloc[i]) or _is_bear(df.iloc[i + 1]):
                w_range = df.iloc[i:i + 3]["high"].max() - df.iloc[i:i + 3]["low"].min()
                bear_move = max(bear_move, w_range)
        if bull_move > MIN_RANGE and bear_move >= BEAR_SPIKE_RATIO * bull_move:
            uncertainty += 2.0
    else:
        bear_move = df.iloc[0]["high"] - df["low"].min()
        bull_move = 0
        for i in range(len(df) - 2):
            if _is_bull(df.iloc[i]) or _is_bull(df.iloc[i + 1]):
                w_range = df.iloc[i:i + 3]["high"].max() - df.iloc[i:i + 3]["low"].min()
                bull_move = max(bull_move, w_range)
        if bear_move > MIN_RANGE and bull_move >= BEAR_SPIKE_RATIO * bear_move:
            uncertainty += 2.0

    # ── 5. Bars since last new high or low (up to +1) ──
    running_high = df.iloc[0]["high"]
    running_low = df.iloc[0]["low"]
    bars_since_extreme = 0
    for i in range(1, len(df)):
        if df.iloc[i]["high"] > running_high or df.iloc[i]["low"] < running_low:
            running_high = max(running_high, df.iloc[i]["high"])
            running_low = min(running_low, df.iloc[i]["low"])
            bars_since_extreme = 0
        else:
            bars_since_extreme += 1
    if bars_since_extreme > BARS_STUCK_THRESHOLD:
        uncertainty += 1.0

    # ── 6. Price near midrange (up to +1) ──
    day_high = df["high"].max()
    day_low = df["low"].min()
    day_range = day_high - day_low
    if day_range > MIN_RANGE:
        midpoint = (day_high + day_low) / 2
        dist_from_mid = abs(df.iloc[-1]["close"] - midpoint) / day_range
        if dist_from_mid < MIDPOINT_TOLERANCE:
            uncertainty += 1.0

    # ── 7. Always-in detection (-2 offset) ──
    always_in = "unclear"
    window = df.iloc[-min(len(df), STRONG_TREND_WINDOW):]

    max_consec_bull = 0
    consec = 0
    for i in range(len(window)):
        row = window.iloc[i]
        if _is_bull(row) and _body_ratio(row) > STRONG_BODY_RATIO:
            consec += 1
            max_consec_bull = max(max_consec_bull, consec)
        else:
            consec = 0

    max_consec_bear = 0
    consec = 0
    for i in range(len(window)):
        row = window.iloc[i]
        if _is_bear(row) and _body_ratio(row) > STRONG_BODY_RATIO:
            consec += 1
            max_consec_bear = max(max_consec_bear, consec)
        else:
            consec = 0

    if max_consec_bull >= 2 and max_consec_bear < 2:
        always_in = "long"
        uncertainty -= 2.0
    elif max_consec_bear >= 2 and max_consec_bull < 2:
        always_in = "short"
        uncertainty -= 2.0
    else:
        uncertainty += 2.0

    # ── 8. Trend line broken (up to +2) ──
    if gap_direction == "up":
        swing_lows = _find_swing_lows(df)
        if len(swing_lows) >= 2:
            # Connect two most recent swing lows
            (x1, y1), (x2, y2) = swing_lows[-2], swing_lows[-1]
            if x2 > x1:
                slope = (y2 - y1) / (x2 - x1)
                projected = y2 + slope * (len(df) - 1 - x2)
                if df.iloc[-1]["close"] < projected:
                    uncertainty += 2.0
    else:
        swing_highs = _find_swing_highs(df)
        if len(swing_highs) >= 2:
            (x1, y1), (x2, y2) = swing_highs[-2], swing_highs[-1]
            if x2 > x1:
                slope = (y2 - y1) / (x2 - x1)
                projected = y2 + slope * (len(df) - 1 - x2)
                if df.iloc[-1]["close"] > projected:
                    uncertainty += 2.0

    # ── 9. Reversal count (up to +2) ──
    # Count alternating swing highs and swing lows
    swing_lows = _find_swing_lows(recent)
    swing_highs = _find_swing_highs(recent)
    # Merge and sort all swings by index
    all_swings = [(idx, "L") for idx, _ in swing_lows] + [(idx, "H") for idx, _ in swing_highs]
    all_swings.sort(key=lambda x: x[0])
    reversals = 0
    for i in range(1, len(all_swings)):
        if all_swings[i][1] != all_swings[i - 1][1]:
            reversals += 1
    if reversals >= REVERSAL_HIGH:
        uncertainty += 2.0
    elif reversals >= REVERSAL_MODERATE:
        uncertainty += 1.0

    # ── 10. Largest bar is counter-trend (up to +1) ──
    if len(recent) > 0:
        ranges = [(recent.iloc[i]["high"] - recent.iloc[i]["low"], i) for i in range(len(recent))]
        largest_idx = max(ranges, key=lambda x: x[0])[1]
        largest_bar = recent.iloc[largest_idx]
        if gap_direction == "up" and _is_bear(largest_bar):
            uncertainty += 1.0
        elif gap_direction == "down" and _is_bull(largest_bar):
            uncertainty += 1.0

    # ── 11. Tight trading range (up to +2) ──
    max_tight_run = 0
    current_run = 1
    for i in range(1, len(df)):
        prev_h, curr_h = df.iloc[i - 1]["high"], df.iloc[i]["high"]
        prev_l, curr_l = df.iloc[i - 1]["low"], df.iloc[i]["low"]
        avg_price = (prev_h + curr_h + prev_l + curr_l) / 4
        if avg_price > MIN_RANGE:
            h_diff = abs(curr_h - prev_h) / avg_price
            l_diff = abs(curr_l - prev_l) / avg_price
            if h_diff <= TIGHT_RANGE_PCT and l_diff <= TIGHT_RANGE_PCT:
                current_run += 1
            else:
                max_tight_run = max(max_tight_run, current_run)
                current_run = 1
    max_tight_run = max(max_tight_run, current_run)

    if max_tight_run >= TIGHT_RANGE_BARS_STRONG:
        uncertainty += 2.0
    elif max_tight_run >= TIGHT_RANGE_BARS_MODERATE:
        uncertainty += 1.0

    # ── 12. Two closes on wrong side of MA (+1) ──
    if len(df) >= EMA_PERIOD + 2:
        closes = df["close"].values.astype(float)
        ema = _compute_ema(closes, EMA_PERIOD)
        last_close = closes[-1]
        prev_close = closes[-2]
        last_ema = ema[-1]
        prev_ema = ema[-2]

        if gap_direction == "up":
            if last_close < last_ema and prev_close < prev_ema:
                uncertainty += 1.0
        else:
            if last_close > last_ema and prev_close > prev_ema:
                uncertainty += 1.0

    return max(uncertainty, 0.0), always_in
