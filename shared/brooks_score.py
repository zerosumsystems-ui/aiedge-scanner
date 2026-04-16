"""
Brooks Price Action Gap Scorer
==============================
Scores gap-up (and gap-down) stocks using Al Brooks' price action methodology.

Produces two independent scores:
  - Urgency (0-10): How strongly is the chart pulling in one direction?
  - Uncertainty (0-10): How confused/two-sided is the chart?

These combine into a signal decision (BUY_PULLBACK, BUY_SPIKE, WAIT, FOG, AVOID, PASS).

Pure math on OHLCV bars — no ML, no external APIs.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── BPA detector integration (guarded — graceful fallback if module missing) ──
try:
    from shared.bpa_detector import detect_all as _bpa_detect_all
    _BPA_AVAILABLE = True
except ImportError:
    _bpa_detect_all = None
    _BPA_AVAILABLE = False

# =============================================================================
# TUNABLE CONSTANTS — adjust these to calibrate scoring sensitivity
# =============================================================================

# ── Spike Quality (urgency component 1, up to 4 pts raw) ──
STRONG_BODY_RATIO = 0.60       # body/range ratio for a "strong trend bar"
CLOSE_TOP_PCT = 0.25           # close must be in top 25% of range (bull) or bottom 25% (bear)
TAIL_MAX_PCT = 0.20            # max lower tail (bull) or upper tail (bear) as % of range
MAX_SPIKE_BARS_SCORED = 4      # cap spike quality scoring at this many bars
MICRO_GAP_BONUS = 0.50         # bonus per micro gap (gap between bar bodies)

# ── Gap Integrity (urgency component 2, -2 to +2 pts raw) ──
GAP_PARTIAL_FILL_PCT = 0.50    # gap partially filled if price enters this % of gap
POST_FILL_EVAL_BARS = 6        # bars after gap-fill bar to evaluate recovery structure
GAP_RECOVERY_STRONG_PCT = 0.50 # recovery >= 50% of gap = strong
GAP_RECOVERY_STRONG_BULL = 0.60  # bull-bar ratio threshold for strong recovery
GAP_RECOVERY_PARTIAL_PCT = 0.25  # recovery >= 25% = partial
GAP_RECOVERY_PARTIAL_BULL = 0.50 # bull-bar ratio threshold for partial recovery

# ── Pullback Quality (urgency component 3, -1 to +2 pts raw) ──
SHALLOW_PULLBACK_PCT = 0.33    # pullback < 33% of spike = shallow
MODERATE_PULLBACK_PCT = 0.50   # pullback 33-50% = moderate
DEEP_PULLBACK_PCT = 0.66       # pullback > 66% = failing
NO_PULLBACK_BAR_THRESHOLD = 5  # bars in spike with no pullback = urgency signal

# ── Follow-Through (urgency component 4, -1.5 to +2 pts raw) ──
# (no additional constants needed)

# ── Tail Quality (urgency component 5, -0.5 to +1 pt raw) ──
TAIL_CLEAN_PCT = 0.10          # "wrong-side" tail < 10% of range = no tail = urgency
TAIL_CLEAN_RATIO = 0.75        # if this fraction of spike bars have clean tails, +1
TAIL_BAD_PCT = 0.30            # tail > 30% of range on wrong side = penalty trigger

# ── Body Gaps (urgency component 6, up to +1 pt raw) ──
BODY_GAP_BONUS = 0.50          # bonus per body gap between consecutive bars, cap 1.0

# ── MA Separation (urgency component 7, up to +1 pt raw) ──
EMA_PERIOD = 20                # exponential moving average period
MA_GAP_BARS_STRONG = 10        # 10+ bars above/below EMA = +1
MA_GAP_BARS_MODERATE = 5       # 5-9 bars = +0.5

# ── Failed Counter-Setups (urgency component 8, up to +1 pt raw) ──
FAILED_SETUP_BONUS = 0.50      # bonus per failed counter-setup, cap 1.0

# ── Volume Confirmation (urgency component 9, up to +1 pt raw) ──
VOLUME_SPIKE_EXTREME = 5.0     # spike vol >= 5x average = +1
VOLUME_SPIKE_STRONG = 3.0      # spike vol >= 3x average = +0.5

# ── Trending Swings (urgency component 10, -1 to +2 pts raw) ──
TRENDING_SWINGS_STRONG = 4     # 4+ consecutive HH/HL or LH/LL = +2
TRENDING_SWINGS_MODERATE = 2   # 2-3 = +1

# ── Spike Duration (urgency component 11, -0.5 to +2 pts raw) ──
SPIKE_DURATION_STRONG = 8      # 8+ bars with no 25% retrace = +2
SPIKE_DURATION_MODERATE = 5    # 5-7 = +1
SPIKE_RETRACE_LIMIT = 0.25    # max retrace per bar relative to move so far

# ── Small Pullback Trend (SPT) — urgency component (up to 3.0 raw) ──
# Detects calm, drifting trends with shallow pullbacks — the kind that ranks
# high on a low-ADR tech day and is exactly what the scanner should surface.
SPT_LOOKBACK_BARS = 15
SPT_TREND_BODY_RATIO = 0.40      # body/range threshold for "trend bar" in SPT
SPT_DEPTH_SHALLOW = 0.25         # all pullbacks below this → perfect
SPT_DEPTH_MODERATE = 0.40        # avg below this → strong

# ── Normalization ──
URGENCY_RAW_MAX = 29.0         # 26 old + 3 small_pullback_trend (SPT, new)
UNCERTAINTY_RAW_MAX = 22.0     # raised: 17 old + 3 two-sided + 2 day-type adjustments

# ── Uncertainty thresholds ──
COLOR_ALT_HIGH = 0.60          # alternation rate above this = high uncertainty
DOJI_BODY_RATIO = 0.30         # body < 30% of range = doji
DOJI_RATIO_HIGH = 0.40         # doji ratio above this = high uncertainty
BODY_OVERLAP_HIGH = 0.50       # overlap ratio above this = trading range behavior
BEAR_SPIKE_RATIO = 0.75        # bear spike >= 75% of bull spike = two-sided
BARS_STUCK_THRESHOLD = 10      # bars since last new high/low = stuck
MIDPOINT_TOLERANCE = 0.20      # within 20% of center = no one winning
STRONG_TREND_WINDOW = 5        # look at last N bars for always-in detection
UNCERTAINTY_ANALYSIS_WINDOW = 15  # bars to analyze for uncertainty metrics (expanded from 10)

# ── Trend Line Broken (uncertainty component 8, up to +2 pts) ──
# (uses swing low detection — no extra constants)

# ── Reversal Count (uncertainty component 9, up to +2 pts) ──
REVERSAL_HIGH = 4              # 4+ reversals in window = +2
REVERSAL_MODERATE = 3          # 3 reversals = +1

# ── Largest Counter-Trend Bar (uncertainty component 10, up to +1 pt) ──
# (no extra constants)

# ── Tight Range (uncertainty component 11, up to +2 pts) ──
TIGHT_RANGE_PCT = 0.005        # highs and lows within 0.5% of each other
TIGHT_RANGE_BARS_STRONG = 10   # 10+ bars in tight range = +2
TIGHT_RANGE_BARS_MODERATE = 6  # 6-9 bars = +1

# ── MA Wrong-Side Closes (uncertainty component 12, +1 pt) ──
MA_WRONG_SIDE_BARS = 2         # last 2 bars both on wrong side of EMA = +1

# ── Two-Sided Trading Ratio (uncertainty component 13, up to +3 pts) ──
TWO_SIDED_VERY_HIGH = 0.45     # > 45% countertrend bars = +3
TWO_SIDED_HIGH = 0.35          # 35-45% = +2
TWO_SIDED_MODERATE = 0.25      # 25-35% = +1

# ── Opening Range Analysis ──
OPENING_RANGE_BARS = 6         # first 30 min on 5-min bars
OR_TREND_FROM_OPEN = 0.25      # OR < 25% of avg range = trend from open
OR_TRENDING_TR_LOW = 0.25      # OR 25-50% = trending TR setup
OR_TRENDING_TR_HIGH = 0.50
OR_TRADING_RANGE = 0.50        # OR > 50% = probably trading range day

# ── Day Type Classifier ──
WARMUP_BARS = 7                # need this many bars before classifying
TREND_BAR_PCT = 0.70           # > 70% trend bars in one direction = trend from open
TREND_MAX_PULLBACK = 0.25      # no pullback > 25% of move = trend day

# ── Liquidity Filter (hard gate in scan_universe) ──
LIQUIDITY_MIN_DOLLAR_VOL = 350_000  # min avg dollar volume per 5-min bar ($350K)
LIQUIDITY_SKIP_BARS = 3             # skip first 3 bars (opening auction noise)

# ── Liquidity Gaps (uncertainty component 14, up to +2 pts) ──
# Penalizes mid-session bar-to-bar price gaps (illiquid jumps)
LIQUIDITY_GAP_PCT = 0.003           # gap > 0.3% of price between consecutive bars
LIQUIDITY_GAPS_HIGH = 5             # 5+ mid-session gaps = +2 uncertainty
LIQUIDITY_GAPS_MODERATE = 3         # 3-4 gaps = +1
LIQUIDITY_GAPS_LOW = 1              # 1-2 gaps = +0.5

# ── Magnitude filter ──
MAGNITUDE_FLOOR = 0.5         # Min move/ATR to appear on leaderboard
MAGNITUDE_CAP_9 = 0.7         # Min move/ATR for urgency > 9.0
MAGNITUDE_CAP_10 = 1.0        # Min move/ATR for urgency > 9.5 (stacks with EMA rule)
CHOP_RATIO_THRESHOLD = 0.25   # Late-session range / day range — below this it's a pullback, not a real trading range

# ── Signal decision thresholds ──
URGENCY_HIGH = 7
UNCERTAINTY_LOW = 3
UNCERTAINTY_MED = 5
UNCERTAINTY_HIGH = 5
UNCERTAINTY_TRAP = 7

# Phase detection
SPIKE_MIN_BARS = 3
TRADING_RANGE_OVERLAP_BARS = 10

# Minimum bar range to avoid division by zero
MIN_RANGE = 0.001

# ── Feature flags (backward compat) ──
GAP_INTEGRITY_POST_FILL_EVAL = True   # False → restores blanket -2.0 on any filled gap
BPA_INTEGRATION_ENABLED = True        # False → disables BPA overlay + bear-flip detection

# ── BPA pattern integration ──
BPA_LONG_SETUP_TYPES = frozenset({"H1", "H2", "FL1", "FL2"})
BPA_SHORT_SETUP_TYPES = frozenset({"L1", "L2"})
BPA_COUNTER_TYPES = frozenset({"spike_channel", "failed_bo"})
BPA_MIN_CONFIDENCE = 0.60
BPA_RECENCY_BARS = 8        # only consider setups detected in last N bars of the DataFrame
BPA_MIN_DF_LEN = 12          # minimum bars before running BPA detectors

# ── Intraday flip signal labels ──
SIGNAL_SELL_INTRADAY = "SELL_PULLBACK_INTRADAY"
SIGNAL_BUY_INTRADAY = "BUY_PULLBACK_INTRADAY"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_range(row) -> float:
    """Bar range, floored to avoid division by zero."""
    return max(row["high"] - row["low"], MIN_RANGE)


def _body(row) -> float:
    """Absolute body size."""
    return abs(row["close"] - row["open"])


def _body_ratio(row) -> float:
    """Body as fraction of range."""
    return _body(row) / _safe_range(row)


def _is_bull(row) -> bool:
    return row["close"] > row["open"]


def _is_bear(row) -> bool:
    return row["close"] < row["open"]


def _lower_tail_pct(row) -> float:
    """Lower tail as fraction of range (for bull bars: open-low, for bear: close-low)."""
    rng = _safe_range(row)
    body_bottom = min(row["open"], row["close"])
    return (body_bottom - row["low"]) / rng


def _upper_tail_pct(row) -> float:
    """Upper tail as fraction of range."""
    rng = _safe_range(row)
    body_top = max(row["open"], row["close"])
    return (row["high"] - body_top) / rng


def _close_position(row) -> float:
    """Where the close sits in the bar's range (0 = low, 1 = high)."""
    rng = _safe_range(row)
    return (row["close"] - row["low"]) / rng


# =============================================================================
# SCORING COMPONENTS — URGENCY (9 components, ~14 raw pts → normalized 0-10)
# =============================================================================

def _compute_ema(closes: np.ndarray, period: int = EMA_PERIOD) -> np.ndarray:
    """Simple EMA calculation. Returns array same length as input."""
    if len(closes) == 0:
        return np.array([])
    ema = np.zeros_like(closes, dtype=float)
    multiplier = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = closes[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


def _find_swing_lows(df: pd.DataFrame, min_bars: int = 3) -> list[tuple[int, float]]:
    """Find swing lows: bar where low < low[i-1] AND low < low[i+1]. Returns [(index, price)]."""
    swings = []
    for i in range(1, len(df) - 1):
        if df.iloc[i]["low"] < df.iloc[i - 1]["low"] and df.iloc[i]["low"] < df.iloc[i + 1]["low"]:
            swings.append((i, df.iloc[i]["low"]))
    return swings


def _find_swing_highs(df: pd.DataFrame, min_bars: int = 3) -> list[tuple[int, float]]:
    """Find swing highs: bar where high > high[i-1] AND high > high[i+1]. Returns [(index, price)]."""
    swings = []
    for i in range(1, len(df) - 1):
        if df.iloc[i]["high"] > df.iloc[i - 1]["high"] and df.iloc[i]["high"] > df.iloc[i + 1]["high"]:
            swings.append((i, df.iloc[i]["high"]))
    return swings


def _score_spike_quality(df: pd.DataFrame, gap_direction: str) -> tuple[float, int]:
    """
    Component 1: Score the initial spike from market open (up to 4 pts raw).
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
    """
    Component 5: Tail quality in the spike (-0.5 to +1 pt raw).
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
    """
    Component 6: Gaps between consecutive bar bodies (up to +1 pt raw).
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
    """
    Component 7: MA separation / gap bars (up to +1 pt raw).
    Count consecutive bars where low (bull) stays above 20 EMA.
    Brooks calls this "20 moving average gap bars" — extreme trend strength.
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
    """
    Component 8: Failed counter-setups (up to +1 pt raw).
    Detect when a counter-trend signal bar forms but the next bar's
    breakout fails. Each failure = sign of strength for the trend.
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
    """
    Component 9: Volume confirmation (up to +1 pt raw).
    Compare spike bar volume to the average volume.
    "If the volume of the large breakout bar is 10 to 20 times the average
    volume, the chance of a measured move increases." — Brooks
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
    """
    H1.2 — Majority trend bars in direction (-1 to +2 raw).
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
    """
    H1.6 — Micro measuring gaps (0 to +2 raw).
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
    """
    H1.16 — Trending 'anything': closes, highs, lows, bodies (-1 to +2 raw).
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
    """
    H1.14 — Trend breaks multiple levels (0 to +2 raw).
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


def _check_liquidity(df: pd.DataFrame) -> dict:
    """
    Hard liquidity gate for scan_universe(). Computes average dollar volume
    per 5-min bar (skipping the first 3 bars to avoid opening-auction noise).
    Returns a dict with pass/fail and the metrics.
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


def _score_small_pullback_trend(df: pd.DataFrame, direction: str) -> float:
    """
    Small Pullback Trend (SPT) — urgency component, 0.0 to 3.0 raw.

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
    in_pb = False
    pb_start = 0
    last_leg_start = None

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

    if bull:
        # Find swing lows: low[i] < low[i-1] and low[i] < low[i+1]
        swing_lows: list[tuple[int, float]] = []
        for i in range(1, n - 1):
            if window["low"].iloc[i] < window["low"].iloc[i - 1] and window["low"].iloc[i] < window["low"].iloc[i + 1]:
                swing_lows.append((i, float(window["low"].iloc[i])))
        # "Higher lows" = chain where each swing_low > prior swing_low
        higher_lows: list[tuple[int, float]] = []
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
        lower_highs: list[tuple[int, float]] = []
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


# ═══════════════════════════════════════════════════════════════════════════
# Cycle-phase classifier (Layer 1) — real-time probability distribution over
# Brooks cycle phases. Evaluated on a rolling window. Five detectors each
# return a raw score in [0, 1]; softmax-normalized to a probability dist.
#
# Kill-switch: set CYCLE_PHASE_CLASSIFIER_ENABLED = False to disable this
# entire subsystem and fall back to no cycle-phase output (downstream consumers
# should treat absence as "undetermined"). This is a recoverable bit flip.
# ═══════════════════════════════════════════════════════════════════════════

# =============================================================================
# BPA PATTERN ADAPTER
# =============================================================================

def _score_bpa_patterns(
    df: pd.DataFrame,
    gap_direction: str,
    gap_fill_status: str,
) -> tuple[float, list[dict]]:
    """Run bpa_detector on df and score how well detected patterns align with direction.

    Returns (bpa_alignment_score, bpa_active_setups):
      +2.0  H2/L2 in-direction (highest-confidence pullback)
      +1.5  H1/L1 in-direction
      +1.0  FL1/FL2 after gap fill recovery (reversal confirmation)
      +0.5  failed_bo or spike_channel in-direction
       0.0  no recent in-direction setup
      -1.0  strong opposing setup (L2 on gap-up, H2 on gap-down)

    The bpa_active_setups list carries serialized setup dicts for details/dashboard.
    """
    if not BPA_INTEGRATION_ENABLED or not _BPA_AVAILABLE or _bpa_detect_all is None:
        return 0.0, []
    if len(df) < BPA_MIN_DF_LEN:
        return 0.0, []

    try:
        raw_setups = _bpa_detect_all(df)
    except Exception:
        logger.warning("bpa_detector.detect_all raised — skipping BPA scoring")
        return 0.0, []

    if not raw_setups:
        return 0.0, []

    # Filter by recency and confidence
    last_bar = len(df) - 1
    filtered = [
        s for s in raw_setups
        if s.confidence >= BPA_MIN_CONFIDENCE
        and s.bar_index >= last_bar - BPA_RECENCY_BARS
    ]
    if not filtered:
        return 0.0, []

    # Determine which types are "in-direction" vs "opposing"
    if gap_direction == "up":
        in_dir_types = BPA_LONG_SETUP_TYPES
        opp_types = BPA_SHORT_SETUP_TYPES
    else:
        in_dir_types = BPA_SHORT_SETUP_TYPES
        opp_types = BPA_LONG_SETUP_TYPES

    best_score = 0.0
    for s in filtered:
        st = s.setup_type
        if st in in_dir_types:
            if st in ("H2", "L2"):
                best_score = max(best_score, 2.0)
            elif st in ("H1", "L1"):
                best_score = max(best_score, 1.5)
            elif st in ("FL1", "FL2") and gap_fill_status == "filled_recovered":
                best_score = max(best_score, 1.0)
            elif st in ("FL1", "FL2"):
                best_score = max(best_score, 0.5)
        elif st in BPA_COUNTER_TYPES:
            best_score = max(best_score, 0.5)
        elif st in opp_types:
            if st in ("H2", "L2"):
                best_score = min(best_score, -1.0)
            else:
                best_score = min(best_score, -0.5)

    # Serialize for details dict
    active = [
        {
            "type": s.setup_type,
            "entry": s.entry,
            "stop": s.stop,
            "target": s.target,
            "confidence": round(s.confidence, 2),
            "bar_index": s.bar_index,
        }
        for s in filtered[:3]
    ]

    return round(best_score, 2), active


CYCLE_PHASE_CLASSIFIER_ENABLED = True    # kill-switch
CYCLE_PHASE_LOOKBACK_BARS = 15
CYCLE_PHASE_SOFTMAX_TEMP = 0.6           # tuned by eye for now; refine with labeled data later
CYCLE_PHASES = ("bull_spike", "bear_spike", "bull_channel", "bear_channel", "trading_range")


def _cycle_bull_spike_raw(df: pd.DataFrame) -> float:
    """Raw [0,1] score for bull spike: recent bars are big bull trend bars with strong closes."""
    if len(df) < 3:
        return 0.0
    recent = df.tail(min(5, len(df)))
    bull_count = 0
    body_strength = 0.0
    close_pos_sum = 0.0
    for _, bar in recent.iterrows():
        if _is_bull(bar):
            bull_count += 1
            rng = _safe_range(bar)
            if rng > 0:
                body_strength += _body(bar) / rng
                close_pos_sum += _close_position(bar)
    n = len(recent)
    if n == 0:
        return 0.0
    bull_density = bull_count / n
    avg_body = body_strength / max(bull_count, 1)
    avg_close_pos = close_pos_sum / max(bull_count, 1)
    # all three must be high: many bull bars, big bodies, closes near highs
    raw = bull_density * avg_body * avg_close_pos
    # bonus if last bar broke recent swing high
    if len(df) >= CYCLE_PHASE_LOOKBACK_BARS:
        recent_high = df.iloc[-CYCLE_PHASE_LOOKBACK_BARS:-1]["high"].max()
        if df.iloc[-1]["close"] > recent_high:
            raw = min(1.0, raw * 1.3)
    return float(max(0.0, min(1.0, raw)))


def _cycle_bear_spike_raw(df: pd.DataFrame) -> float:
    """Mirror of bull spike for down direction."""
    if len(df) < 3:
        return 0.0
    recent = df.tail(min(5, len(df)))
    bear_count = 0
    body_strength = 0.0
    close_pos_sum = 0.0
    for _, bar in recent.iterrows():
        if _is_bear(bar):
            bear_count += 1
            rng = _safe_range(bar)
            if rng > 0:
                body_strength += _body(bar) / rng
                close_pos_sum += (1.0 - _close_position(bar))  # bear close is near LOW
    n = len(recent)
    if n == 0:
        return 0.0
    bear_density = bear_count / n
    avg_body = body_strength / max(bear_count, 1)
    avg_close_pos = close_pos_sum / max(bear_count, 1)
    raw = bear_density * avg_body * avg_close_pos
    if len(df) >= CYCLE_PHASE_LOOKBACK_BARS:
        recent_low = df.iloc[-CYCLE_PHASE_LOOKBACK_BARS:-1]["low"].min()
        if df.iloc[-1]["close"] < recent_low:
            raw = min(1.0, raw * 1.3)
    return float(max(0.0, min(1.0, raw)))


def _cycle_bull_channel_raw(df: pd.DataFrame) -> float:
    """
    Raw [0,1] for bull channel: sustained bull bias with smaller bodies, shallow
    pullbacks, higher-closes drift. Not a spike — quieter and more two-sided.
    """
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    # net drift up
    net_up = (closes[-1] - closes[0]) / max(highs.max() - lows.min(), 1e-9)
    drift = max(0.0, min(1.0, net_up * 2.0))   # scale: 50% of range = full credit

    # higher-closes streak fraction
    higher = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
    higher_frac = higher / (len(closes) - 1)

    # reject if bodies are TOO big (that's a spike, not a channel)
    avg_body_ratio = np.mean([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    channel_body_fit = 1.0 - abs(avg_body_ratio - 0.45) / 0.45   # peak at 0.45
    channel_body_fit = max(0.0, min(1.0, channel_body_fit))

    # shallow pullback: max drawdown from running peak stays small
    running_peak = np.maximum.accumulate(highs)
    drawdowns = (running_peak - lows) / np.maximum(running_peak, 1e-9)
    max_dd = drawdowns.max()
    shallow = max(0.0, 1.0 - max_dd * 5.0)   # 20% DD → zero credit

    raw = drift * 0.35 + higher_frac * 0.25 + channel_body_fit * 0.20 + shallow * 0.20
    return float(max(0.0, min(1.0, raw)))


def _cycle_bear_channel_raw(df: pd.DataFrame) -> float:
    """Mirror of bull channel for down direction."""
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    net_down = (closes[0] - closes[-1]) / max(highs.max() - lows.min(), 1e-9)
    drift = max(0.0, min(1.0, net_down * 2.0))

    lower = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
    lower_frac = lower / (len(closes) - 1)

    avg_body_ratio = np.mean([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    channel_body_fit = 1.0 - abs(avg_body_ratio - 0.45) / 0.45
    channel_body_fit = max(0.0, min(1.0, channel_body_fit))

    running_trough = np.minimum.accumulate(lows)
    rallies = (highs - running_trough) / np.maximum(running_trough, 1e-9)
    max_rally = rallies.max()
    shallow = max(0.0, 1.0 - max_rally * 5.0)

    raw = drift * 0.35 + lower_frac * 0.25 + channel_body_fit * 0.20 + shallow * 0.20
    return float(max(0.0, min(1.0, raw)))


def _cycle_trading_range_raw(df: pd.DataFrame) -> float:
    """
    Raw [0,1] for trading range: doji-heavy, balanced bull/bear, horizontal
    closes, no clear net direction.
    """
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    # doji density
    body_ratios = np.array([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    doji_frac = float((body_ratios < DOJI_BODY_RATIO).mean())

    # bull / bear balance: closer to 0.5 = more rangebound
    bull_count = sum(1 for c, o in zip(closes, opens) if c > o)
    bull_frac = bull_count / len(closes)
    balance = 1.0 - abs(bull_frac - 0.5) * 2.0

    # low net drift relative to range
    total_range = highs.max() - lows.min()
    net_move = abs(closes[-1] - closes[0])
    drift_ratio = net_move / max(total_range, 1e-9)
    horizontal = max(0.0, 1.0 - drift_ratio * 2.5)   # drift > 40% of range → zero credit

    # close clustering: low stdev of closes relative to range
    if total_range > 0:
        close_stdev = closes.std()
        cluster = max(0.0, 1.0 - (close_stdev / total_range) * 4.0)
    else:
        cluster = 1.0

    raw = doji_frac * 0.3 + balance * 0.3 + horizontal * 0.25 + cluster * 0.15
    return float(max(0.0, min(1.0, raw)))


def _softmax(values: list[float], temperature: float) -> list[float]:
    """Numerically stable softmax."""
    arr = np.array(values) / max(temperature, 1e-9)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def classify_cycle_phase(df: pd.DataFrame) -> dict:
    """
    Layer-1 cycle-phase classifier. Returns a probability distribution over the
    five Brooks cycle phases (bull_spike, bear_spike, bull_channel, bear_channel,
    trading_range), plus the argmax label and its confidence.

    Returns {} if the classifier is disabled via kill-switch.
    """
    if not CYCLE_PHASE_CLASSIFIER_ENABLED:
        return {}
    if df is None or len(df) < 3:
        return {
            "probs": {p: 0.2 for p in CYCLE_PHASES},
            "top": "trading_range",
            "confidence": 0.2,
        }
    raw = [
        _cycle_bull_spike_raw(df),
        _cycle_bear_spike_raw(df),
        _cycle_bull_channel_raw(df),
        _cycle_bear_channel_raw(df),
        _cycle_trading_range_raw(df),
    ]
    probs = _softmax(raw, CYCLE_PHASE_SOFTMAX_TEMP)
    dist = dict(zip(CYCLE_PHASES, probs))
    top = max(dist, key=dist.get)
    return {
        "probs": {k: round(v, 3) for k, v in dist.items()},
        "top": top,
        "confidence": round(dist[top], 3),
        "raw": {k: round(v, 3) for k, v in zip(CYCLE_PHASES, raw)},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Session-shape classifier (Layer 2) — post-hoc / late-session labels for the
# day's overall character. Runs on the full session, returns a probability
# distribution over 6 Brooks session shapes plus "undetermined" baseline.
#
# Brooks himself names day types AFTER the session's shape reveals itself,
# so confidence is inherently lower in the first hour. A guard hides the
# Layer-2 argmax from the live card until ≥10:30 ET (WARMUP_MINUTES).
# ═══════════════════════════════════════════════════════════════════════════

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
    """
    Day opened with a clear spike in `direction`, minimal retrace in first hour,
    and still holds the open-direction bias at current bar.
    """
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
    """
    Leading spike of `spike_bars` bars then a sustained shallower channel in
    the same direction. Net post-spike move should be positive-but-slower.
    """
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

    channel_body_avg = np.mean([abs(r["close"] - r["open"]) / max(_safe_range(r), 1e-9) for _, r in channel.iterrows()])
    spike_body_avg = np.mean([abs(r["close"] - r["open"]) / max(_safe_range(r), 1e-9) for _, r in spike.iterrows()])
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
    """
    Opening direction reverses: mid-session peak (or trough), then lower high
    (higher low for bear reversal), current close ≥50% back from the extreme.
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
    """
    Opening trend, then a trading range mid-session (the "flag"), then a
    continuation in the same direction as the opening trend.
    """
    if len(df) < 18:
        return 0.0
    open_leg = df.iloc[:6]
    middle = df.iloc[6:-6] if len(df) > 18 else df.iloc[6:-3]
    late = df.iloc[-6:]

    # open leg moved in direction
    om = open_leg.iloc[-1]["close"] - open_leg.iloc[0]["open"]
    if direction == "up" and om <= 0: return 0.0
    if direction == "down" and om >= 0: return 0.0

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
    if compressed: raw += 0.5
    if continuation: raw += 0.4
    if abs(lm) > abs(om) * 0.3:
        raw += 0.1    # late move is meaningful size
    return float(max(0.0, min(1.0, raw)))


def _shape_opening_reversal_raw(df: pd.DataFrame, direction: str) -> float:
    """
    Opening breakout attempt in direction, fails within 60-90 min, reverses
    ≥50% of the opening thrust and holds into session.
    """
    if len(df) < 18:
        return 0.0
    session_open = df.iloc[0]["open"]
    first_hr = df.iloc[:12]    # 60 min = 12 bars 5min
    rest = df.iloc[12:]

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
    if reversal_pct >= 0.50: raw += 0.35
    if reversal_pct >= 1.00: raw += 0.25   # beyond the open
    if net_negative:         raw += 0.30
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
    """
    Layer-2 session shape classifier. Returns probability distribution over 6
    session shapes (including 'undetermined' baseline) + argmax + confidence +
    a boolean `show_on_live_card` that is False during the warmup period.
    """
    if not SESSION_SHAPE_CLASSIFIER_ENABLED:
        return {}
    if df is None or len(df) < 6:
        return {
            "probs": {s: 1.0/len(SESSION_SHAPES) for s in SESSION_SHAPES},
            "top": "undetermined", "confidence": 1.0/len(SESSION_SHAPES),
            "show_on_live_card": False,
        }

    raw = {
        "trend_from_open":  _shape_trend_from_open_raw(df, direction),
        "spike_and_channel":_shape_spike_and_channel_raw(df, direction, spike_bars),
        "trend_reversal":   _shape_trend_reversal_raw(df, direction),
        "trend_resumption": _shape_trend_resumption_raw(df, direction),
        "opening_reversal": _shape_opening_reversal_raw(df, direction),
        "undetermined":     0.15,      # baseline — always present as the null hypothesis
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


def _score_liquidity_gaps(df: pd.DataFrame) -> float:
    """
    Uncertainty component: penalizes mid-session bar-to-bar price gaps (0 to +2 raw).
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


# =============================================================================
# SCORING COMPONENTS — UNCERTAINTY (12 components, ~19 raw pts → normalized 0-10)
# =============================================================================

def _score_uncertainty(df: pd.DataFrame, gap_direction: str) -> tuple[float, str]:
    """
    Score how confused/two-sided the chart is (12 components).
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


# =============================================================================
# PHASE 1 — OPENING RANGE, DAY TYPE, NEW COMPONENTS, WEIGHT MATRIX
# =============================================================================

def _opening_range(df: pd.DataFrame, n_bars: int = OPENING_RANGE_BARS,
                   avg_daily_range: float = None) -> dict:
    """
    Compute the opening range from the first N bars (default 6 = first 30 min).
    If avg_daily_range is not provided, estimate from the data we have.
    """
    n = min(n_bars, len(df))
    if n < 1:
        return {"range_high": 0.0, "range_low": 0.0, "range_size": 0.0, "range_pct": 0.5}

    range_high = float(df.iloc[:n]["high"].max())
    range_low = float(df.iloc[:n]["low"].min())
    range_size = range_high - range_low

    # Estimate avg daily range from the full session data if not given
    if avg_daily_range is None or avg_daily_range <= MIN_RANGE:
        # Use the current day's full range as a rough proxy (underestimates early in day)
        day_range = float(df["high"].max() - df["low"].min())
        # Scale up if we don't have a full day: assume first hour ≈ 60-70% of day range
        hours_of_data = len(df) * 5 / 60  # assume 5-min bars
        if hours_of_data < 2:
            avg_daily_range = day_range / 0.5  # early = aggressive estimate
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


def _compute_two_sided_ratio(df: pd.DataFrame, gap_direction: str) -> float:
    """
    Ratio of countertrend bars to total bars over the ENTIRE bar set.
    A stock that trended for 80% of bars and chopped for 20% should have a LOW
    ratio, not a high one. Using all bars prevents late-session chop from
    dominating the reading.
    """
    n = len(df)
    if n < 3:
        return 0.5

    counter = 0
    for i in range(n):
        row = df.iloc[i]
        if gap_direction == "up" and _is_bear(row):
            counter += 1
        elif gap_direction == "down" and _is_bull(row):
            counter += 1
    return counter / n


def _classify_day_type(df: pd.DataFrame, opening_range: dict,
                       spike_bars: int, two_sided_ratio: float,
                       gap_direction: str, gap_held: bool) -> dict:
    """
    Classify the day type based on Brooks' taxonomy using STRUCTURAL analysis.

    Classification is based on WHERE THE OPEN SITS in the day's range and how
    bars are distributed — NOT on net move from prior close. A stock can gap 10%
    and still be a trading range day if it opens and then goes sideways.

    Returns dict with day_type, confidence, and a plain-English warning.
    """
    n = len(df)
    or_pct = opening_range["range_pct"]

    # ── Warmup: not enough bars to classify ──
    if n < WARMUP_BARS:
        is_trend = _is_bull if gap_direction == "up" else _is_bear
        strong_start = 0
        for i in range(min(3, n)):
            row = df.iloc[i]
            if is_trend(row) and _body_ratio(row) > STRONG_BODY_RATIO:
                strong_start += 1
        if strong_start >= 2 and gap_held:
            return {
                "day_type": "trend_from_open",
                "confidence": 0.4,
                "warning": "Too early to confirm — preliminary trend from open signal. Wide stops.",
            }
        return {
            "day_type": "undetermined",
            "confidence": 0.0,
            "warning": "Too early to classify — only opening bar strength is scored.",
        }

    # ══════════════════════════════════════════════════════════════════════
    # Compute whole-session STRUCTURAL metrics
    # ══════════════════════════════════════════════════════════════════════

    first_open = df.iloc[0]["open"]
    day_high = df["high"].max()
    day_low = df["low"].min()
    total_range = day_high - day_low

    # ── Where is the open relative to the day's range? ──
    # open_position near 0 = open at the low → bull trend from open
    # open_position near 1 = open at the high → bear trend from open
    # open_position near 0.5 = open in the middle → range day / gap-and-chop
    if total_range > MIN_RANGE:
        open_position = (first_open - day_low) / total_range
    else:
        open_position = 0.5

    # ── Max pullback as fraction of day range (structural, not % of move) ──
    if gap_direction == "up":
        running_high = df.iloc[0]["high"]
        max_pullback_abs = 0.0
        for i in range(1, n):
            running_high = max(running_high, df.iloc[i]["high"])
            retrace = running_high - df.iloc[i]["low"]
            max_pullback_abs = max(max_pullback_abs, retrace)
    else:
        running_low = df.iloc[0]["low"]
        max_pullback_abs = 0.0
        for i in range(1, n):
            running_low = min(running_low, df.iloc[i]["low"])
            retrace = df.iloc[i]["high"] - running_low
            max_pullback_abs = max(max_pullback_abs, retrace)
    max_pullback_of_range = max_pullback_abs / max(total_range, MIN_RANGE)

    # ── Trend bar count over ALL bars ──
    is_trend = _is_bull if gap_direction == "up" else _is_bear
    trend_bar_count = sum(1 for i in range(n) if is_trend(df.iloc[i]))
    trend_pct = trend_bar_count / n

    # ── Trend bar % for just the first 70% of bars (late-session protection) ──
    early_cutoff = max(int(n * 0.7), WARMUP_BARS)
    early_trend_count = sum(1 for i in range(min(early_cutoff, n)) if is_trend(df.iloc[i]))
    early_trend_pct = early_trend_count / max(min(early_cutoff, n), 1)

    # ── TTR / range metrics ──
    bar_ranges = df.apply(lambda r: r["high"] - r["low"], axis=1).values
    avg_bar_range = float(np.mean(bar_ranges)) if len(bar_ranges) > 0 else MIN_RANGE
    bodies = df.apply(lambda r: _body(r), axis=1).values
    avg_body_size = float(np.mean(bodies)) if len(bodies) > 0 else MIN_RANGE
    avg_body_ratio = avg_body_size / max(avg_bar_range, MIN_RANGE)

    color_changes = 0
    for i in range(1, n):
        if _is_bull(df.iloc[i - 1]) != _is_bull(df.iloc[i]):
            color_changes += 1
    color_change_rate = color_changes / max(n - 1, 1)

    consec_overlaps = []
    for i in range(1, n):
        overlap_range = max(0, min(df.iloc[i]["high"], df.iloc[i - 1]["high"])
                           - max(df.iloc[i]["low"], df.iloc[i - 1]["low"]))
        consec_overlaps.append(overlap_range / max(avg_bar_range, MIN_RANGE))
    avg_consec_overlap = float(np.mean(consec_overlaps)) if consec_overlaps else 0.0

    # Impulsive legs detection
    max_bull_run = 0
    max_bear_run = 0
    bull_run = 0
    bear_run = 0
    for i in range(n):
        row = df.iloc[i]
        if _is_bull(row):
            bull_run += 1
            max_bull_run = max(max_bull_run, bull_run)
            bear_run = 0
        elif _is_bear(row):
            bear_run += 1
            max_bear_run = max(max_bear_run, bear_run)
            bull_run = 0
        else:
            bull_run = 0
            bear_run = 0
    has_impulsive_legs = (max_bull_run >= 3 and max_bear_run >= 3)

    # Slope ratio (early vs late) for spike & channel
    if n >= 10:
        early_move = abs(df.iloc[min(4, n - 1)]["close"] - df.iloc[0]["open"])
        late_bars_count = n - 5
        late_move = abs(df.iloc[-1]["close"] - df.iloc[5]["open"]) if late_bars_count > 0 else 0
        early_per_bar = early_move / max(min(5, n), 1)
        late_per_bar = late_move / max(late_bars_count, 1)
        slope_ratio = late_per_bar / max(early_per_bar, MIN_RANGE)
    else:
        slope_ratio = 1.0

    # ══════════════════════════════════════════════════════════════════════
    # STRUCTURAL CLASSIFICATION
    # Priority: trend_from_open > spike_and_channel > tight_tr > trading_range
    #           > trending_tr > undetermined
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. Trend from open (structural: open set the extreme) ──
    # ALL criteria must be met:
    #   a) open_position < 0.15 (bull) or > 0.85 (bear) — open IS the extreme
    #   b) trend_pct > 0.50 — majority of ALL bars trend in direction
    #   c) max_pullback < 30% of day range — no deep pullbacks
    #   d) at least WARMUP_BARS of data
    open_at_extreme = (
        (gap_direction == "up" and open_position < 0.15) or
        (gap_direction == "down" and open_position > 0.85)
    )

    if open_at_extreme and trend_pct > 0.50 and max_pullback_of_range < 0.30:
        # Confidence scales with how extreme the open is and how clean the bars are
        extremity = (0.15 - open_position) / 0.15 if gap_direction == "up" else (open_position - 0.85) / 0.15
        extremity = max(min(extremity, 1.0), 0.0)
        conf = min(0.5 + extremity * 0.2 + trend_pct * 0.2 + (1 - max_pullback_of_range) * 0.1, 0.95)
        return {
            "day_type": "trend_from_open",
            "confidence": round(conf, 2),
            "warning": "Strong trend from open — stops must be wide. Consider reduced size.",
        }

    # ── Late-session protection ──
    # If the first 70% was clearly trending (open at extreme + strong early bars),
    # don't let late chop reclassify. Late chop is normal on trend days.
    if (open_at_extreme
        and early_trend_pct > 0.55
        and trend_pct > 0.45
        and spike_bars >= SPIKE_MIN_BARS):
        late_note = ""
        if n > early_cutoff:
            late_bars_n = n - early_cutoff
            late_counter = sum(1 for i in range(early_cutoff, n) if not is_trend(df.iloc[i]))
            if late_counter / max(late_bars_n, 1) > 0.5:
                late_note = " (late channel deterioration — normal, stay with-trend)"
        conf = min(0.5 + early_trend_pct * 0.3, 0.85)
        return {
            "day_type": "spike_and_channel",
            "confidence": round(conf, 2),
            "warning": f"Spike & channel{late_note} — with-trend pullback entries preferred.",
        }

    # ── 2. Spike and channel (structural) ──
    # Open near extreme (< 0.25 or > 0.75), initial spike, then shallower slope
    open_near_extreme = (
        (gap_direction == "up" and open_position < 0.25) or
        (gap_direction == "down" and open_position > 0.75)
    )

    if (spike_bars >= SPIKE_MIN_BARS
        and n >= WARMUP_BARS
        and open_near_extreme
        and slope_ratio < 0.6
        and two_sided_ratio < 0.45):
        conf = min(0.4 + (spike_bars / 8) * 0.3 + (1 - slope_ratio) * 0.2, 0.9)
        return {
            "day_type": "spike_and_channel",
            "confidence": round(conf, 2),
            "warning": ("Spike done, now in channel — countertrend setups look good "
                        "but almost all fail. With-trend entries force buying near highs."),
        }

    # Also catch spike-and-channel when open isn't as extreme but structure is clear
    if (spike_bars >= SPIKE_MIN_BARS
        and n >= WARMUP_BARS
        and trend_pct > 0.50
        and two_sided_ratio < 0.40
        and max_pullback_of_range < 0.50):
        conf = min(0.35 + trend_pct * 0.3 + (spike_bars / 10) * 0.2, 0.8)
        return {
            "day_type": "spike_and_channel",
            "confidence": round(conf, 2),
            "warning": "Spike & channel — with-trend pullback entries preferred.",
        }

    # ── 3. Tight trading range (barbwire — all 5 criteria) ──
    if n >= 12:
        est_daily_range = total_range / 0.7 if total_range > MIN_RANGE else total_range
        ttr_criteria = [
            avg_bar_range / max(total_range, MIN_RANGE) < 0.15,
            avg_body_ratio < 0.4,
            color_change_rate > 0.5,
            avg_consec_overlap > 0.4,
            total_range / max(est_daily_range, MIN_RANGE) < 0.3,
        ]
        ttr_count = sum(ttr_criteria)
        if has_impulsive_legs:
            ttr_count = 0  # real legs → not tight
        if ttr_count >= 5:
            return {
                "day_type": "tight_tr",
                "confidence": round(min(0.6 + ttr_count * 0.05, 0.9), 2),
                "warning": "Tight range — stop entries are a LOSING strategy here. Wait for breakout.",
            }

    # ── 4. Trading range (no net-move gate — purely structural) ──
    if n >= 12:
        is_range = False
        range_conf = 0.4

        # Range with impulsive legs in both directions
        if has_impulsive_legs and two_sided_ratio > 0.30:
            is_range = True
            range_conf = min(0.5 + two_sided_ratio * 0.3, 0.85)

        # Classic range: open near middle of range (gap-and-chop pattern)
        # OR wide opening range OR high two-sided ratio with weak trend
        elif 0.30 < open_position < 0.70 and two_sided_ratio > 0.35:
            is_range = True
            mid_score = 1.0 - 2.0 * abs(open_position - 0.5)  # 0=extreme, 1=center
            range_conf = min(0.4 + mid_score * 0.2 + two_sided_ratio * 0.2, 0.85)

        elif (or_pct > OR_TRADING_RANGE or two_sided_ratio > 0.45) and trend_pct < 0.55:
            is_range = True
            range_conf = min(0.4 + two_sided_ratio * 0.4 + or_pct * 0.2, 0.85)

        if is_range:
            # Guard: if the late-session chop is tiny relative to the full day's range,
            # the "range" is just a pullback/consolidation in a trend — not a real range day.
            day_range = df["high"].max() - df["low"].min()
            if day_range > 0:
                late_n = max(int(len(df) * 0.3), 3)
                late_bars = df.tail(late_n)
                late_range = late_bars["high"].max() - late_bars["low"].min()
                chop_ratio = late_range / day_range
                if chop_ratio < CHOP_RATIO_THRESHOLD:
                    # The "range" is trivial — trend paused, not a true range day
                    if open_position < 0.20:
                        return {
                            "day_type": "trend_from_open",
                            "confidence": 0.80,
                            "warning": "Strong trend from open — late consolidation is normal",
                            "chop_ratio": round(chop_ratio, 3),
                        }
                    elif open_position > 0.80:
                        return {
                            "day_type": "trend_from_open",
                            "confidence": 0.80,
                            "warning": "Strong bear trend from open — late consolidation is normal",
                            "chop_ratio": round(chop_ratio, 3),
                        }
                    else:
                        return {
                            "day_type": "spike_and_channel",
                            "confidence": 0.65,
                            "warning": "Spike followed by narrow channel — trend intact",
                            "chop_ratio": round(chop_ratio, 3),
                        }
            return {
                "day_type": "trading_range",
                "confidence": round(range_conf, 2),
                "warning": ("Trading range — strong spikes to edges are probably traps. "
                            "Probability ~50%. Buy low, sell high."),
            }

    # ── 5. Trending trading range ──
    if (n >= 12
        and OR_TRENDING_TR_LOW <= or_pct <= OR_TRENDING_TR_HIGH
        and 0.25 < two_sided_ratio < 0.45):
        return {
            "day_type": "trending_tr",
            "confidence": round(min(0.3 + (1 - two_sided_ratio) * 0.3, 0.7), 2),
            "warning": ("Trending TR — wait for breakout from current range, "
                        "don't chase the breakout bar."),
        }

    # ── 6. Default: best guess ──
    if spike_bars >= SPIKE_MIN_BARS and trend_pct > 0.5:
        return {
            "day_type": "spike_and_channel",
            "confidence": 0.35,
            "warning": "Probable spike & channel — with-trend pullback entries preferred.",
        }

    return {
        "day_type": "undetermined",
        "confidence": 0.2,
        "warning": "Mixed signals — no clear day type yet. Wait for more bars.",
    }


def _score_trending_swings(df: pd.DataFrame, gap_direction: str) -> float:
    """
    Urgency component: Trending highs/lows (-1 to +2 raw).
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


def _score_two_sided_ratio(df: pd.DataFrame, gap_direction: str) -> float:
    """
    Uncertainty component: Two-sided trading ratio (up to +3 raw).
    High ratio of countertrend bars = more uncertainty.
    """
    ratio = _compute_two_sided_ratio(df, gap_direction)

    if ratio > TWO_SIDED_VERY_HIGH:
        return 3.0
    elif ratio > TWO_SIDED_HIGH:
        return 2.0
    elif ratio > TWO_SIDED_MODERATE:
        return 1.0
    return 0.0


def _score_spike_duration(df: pd.DataFrame, spike_bars: int, gap_direction: str) -> float:
    """
    Urgency component: Spike duration — how many bars the spike sustains
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
# DAY-TYPE WEIGHT MATRIX
# =============================================================================

# Each weight multiplies the corresponding raw component score before normalization.
# Trend days amplify trend-strength components, dampen two-sided metrics.
# Range days do the opposite — spikes at range edges are traps, not setups.

DAY_TYPE_WEIGHTS = {
    "trend_from_open": {
        # Urgency components — amplified on strong trend days
        "spike_quality": 1.5,
        "gap_integrity": 1.3,
        "pullback_quality": 1.2,
        "follow_through": 1.3,
        "tail_quality": 1.2,
        "body_gaps": 1.3,
        "ma_separation": 1.3,
        "failed_setups": 1.2,
        "volume_conf": 1.2,
        "trending_swings": 1.5,
        "spike_duration": 1.3,
        "majority_trend_bars": 1.5,
        "micro_gaps": 1.3,
        "trending_everything": 1.5,
        "levels_broken": 1.3,
        "small_pullback_trend": 1.5,   # SPT — same treatment as spike_quality
        # Uncertainty components — dampened
        "two_sided_ratio": 0.3,
        "color_alternation": 0.3,
        "doji_ratio": 0.3,
        "counter_spike": 0.5,
        "liquidity_gaps": 1.0,
    },
    "spike_and_channel": {
        "spike_quality": 1.2,
        "gap_integrity": 1.0,
        "pullback_quality": 1.3,
        "follow_through": 1.0,
        "tail_quality": 1.0,
        "body_gaps": 1.0,
        "ma_separation": 1.0,
        "failed_setups": 1.2,
        "volume_conf": 1.0,
        "trending_swings": 1.0,
        "spike_duration": 1.0,
        "majority_trend_bars": 1.2,
        "micro_gaps": 1.2,
        "trending_everything": 1.0,
        "levels_broken": 1.0,
        "small_pullback_trend": 1.2,   # SPT — spike & channel often has small pullbacks
        "two_sided_ratio": 0.8,
        "color_alternation": 0.8,
        "doji_ratio": 0.8,
        "counter_spike": 1.0,
        "liquidity_gaps": 1.0,
    },
    "trending_tr": {
        "spike_quality": 0.8,
        "gap_integrity": 0.8,
        "pullback_quality": 0.8,
        "follow_through": 1.0,
        "tail_quality": 0.8,
        "body_gaps": 0.8,
        "ma_separation": 0.8,
        "failed_setups": 1.0,
        "volume_conf": 1.0,
        "trending_swings": 0.8,
        "spike_duration": 0.8,
        "majority_trend_bars": 0.8,
        "micro_gaps": 0.8,
        "trending_everything": 0.8,
        "levels_broken": 0.8,
        "small_pullback_trend": 1.0,   # SPT — trending TR often exhibits SPT behavior
        "two_sided_ratio": 1.2,
        "color_alternation": 1.0,
        "doji_ratio": 1.0,
        "counter_spike": 1.2,
        "liquidity_gaps": 1.0,
    },
    "trading_range": {
        "spike_quality": 0.3,   # spikes in ranges are traps
        "gap_integrity": 0.5,
        "pullback_quality": 0.3,
        "follow_through": 0.5,
        "tail_quality": 0.3,
        "body_gaps": 0.3,
        "ma_separation": 0.3,
        "failed_setups": 0.5,
        "volume_conf": 0.5,
        "trending_swings": 0.3,
        "spike_duration": 0.3,
        "majority_trend_bars": 0.5,
        "micro_gaps": 0.5,
        "trending_everything": 0.3,
        "levels_broken": 0.3,
        "small_pullback_trend": 0.3,   # SPT — spikes and SPT in ranges are traps
        "two_sided_ratio": 1.5,
        "color_alternation": 1.3,
        "doji_ratio": 1.3,
        "counter_spike": 1.5,
        "liquidity_gaps": 1.0,
    },
    "tight_tr": {
        "spike_quality": 0.1,
        "gap_integrity": 0.3,
        "pullback_quality": 0.1,
        "follow_through": 0.1,
        "tail_quality": 0.1,
        "body_gaps": 0.1,
        "ma_separation": 0.1,
        "failed_setups": 0.1,
        "volume_conf": 0.3,
        "trending_swings": 0.1,
        "spike_duration": 0.1,
        "majority_trend_bars": 0.3,
        "micro_gaps": 0.1,
        "trending_everything": 0.1,
        "levels_broken": 0.1,
        "small_pullback_trend": 0.1,   # SPT — suppressed hard in tight TR
        "two_sided_ratio": 1.5,
        "color_alternation": 1.5,
        "doji_ratio": 1.5,
        "counter_spike": 1.5,
        "liquidity_gaps": 1.0,
    },
    "undetermined": {
        # Neutral weights — everything at 1.0 during warmup
        "spike_quality": 1.0, "gap_integrity": 1.0, "pullback_quality": 1.0,
        "follow_through": 1.0, "tail_quality": 1.0, "body_gaps": 1.0,
        "ma_separation": 1.0, "failed_setups": 1.0, "volume_conf": 1.0,
        "trending_swings": 1.0, "spike_duration": 1.0,
        "majority_trend_bars": 1.0, "micro_gaps": 1.0,
        "trending_everything": 1.0, "levels_broken": 1.0,
        "small_pullback_trend": 1.0,
        "two_sided_ratio": 1.0, "color_alternation": 1.0, "doji_ratio": 1.0,
        "counter_spike": 1.0, "liquidity_gaps": 1.0,
    },
}


def _apply_day_type_weight(raw: float, component: str, day_type: str) -> float:
    """Multiply a raw component score by its day-type weight."""
    weights = DAY_TYPE_WEIGHTS.get(day_type, DAY_TYPE_WEIGHTS["undetermined"])
    return raw * weights.get(component, 1.0)


# =============================================================================
# PHASE, RISK/REWARD, SIGNAL, SUMMARY
# =============================================================================

def _detect_phase(df: pd.DataFrame, spike_bars: int, uncertainty: float,
                  gap_direction: str, gap_held: bool) -> str:
    """Determine the current market phase."""
    if spike_bars >= SPIKE_MIN_BARS and len(df) <= spike_bars + 2:
        return "spike"

    if not gap_held and uncertainty > UNCERTAINTY_HIGH:
        return "failed_gap"

    if uncertainty > UNCERTAINTY_HIGH:
        return "trading_range"

    if spike_bars >= SPIKE_MIN_BARS and len(df) > spike_bars + 3:
        post_spike = df.iloc[spike_bars:]
        if gap_direction == "up":
            lows = post_spike["low"].values
            if len(lows) >= 3:
                mid = len(lows) // 2
                if np.mean(lows[mid:]) > np.mean(lows[:mid]):
                    return "channel"
        else:
            highs = post_spike["high"].values
            if len(highs) >= 3:
                mid = len(highs) // 2
                if np.mean(highs[mid:]) < np.mean(highs[:mid]):
                    return "channel"

    if spike_bars >= SPIKE_MIN_BARS:
        return "channel"

    return "trading_range"


def _compute_risk_reward(df: pd.DataFrame, gap_direction: str,
                         prior_close: float, spike_bars: int,
                         rr_direction_override: str = None) -> tuple[float, float, float]:
    """Compute risk, reward, and R:R ratio with minimum risk floor.

    rr_direction_override: if set, use this direction for R/R calc instead of
    gap_direction. Used by bear-flip signals that need short-side R/R on a
    gap-up day.
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


def _determine_signal(urgency: float, uncertainty: float, gap_held: bool,
                      gap_direction: str, rr_ratio: float,
                      spike_bars: int, pullback_exists: bool,
                      bpa_alignment: float = 0.0,
                      always_in: str = "unclear",
                      gap_fill_status: str = "held") -> str:
    """
    Decision matrix for the final signal.
    Urgency and uncertainty are ALREADY normalized to 0-10 at this point.

    BPA overlay (Fix 1) and bear-flip detection (Fix 3) operate after the
    core matrix, modifying the signal based on pattern confirmation.
    """
    # Failed gap override — but BPA can reconsider if gap recovered
    if not gap_held:
        if (BPA_INTEGRATION_ENABLED
                and gap_fill_status == "filled_recovered"
                and bpa_alignment >= 1.0):
            # Gap filled but price recovered with BPA confirmation — not AVOID
            signal = "WAIT"
        else:
            return "AVOID"
    # Trap: high urgency AND high uncertainty — most dangerous state
    elif urgency >= URGENCY_HIGH and uncertainty >= UNCERTAINTY_TRAP:
        return "AVOID"
    # Best setups: high urgency, low uncertainty
    elif urgency >= URGENCY_HIGH and uncertainty <= UNCERTAINTY_LOW:
        if not pullback_exists and spike_bars >= SPIKE_MIN_BARS:
            signal = "BUY_SPIKE"
        else:
            signal = "BUY_PULLBACK"
    # Promising: moderate-to-high urgency, moderate uncertainty
    elif urgency >= URGENCY_HIGH and uncertainty <= UNCERTAINTY_MED:
        signal = "BUY_PULLBACK"
    # Wait: decent urgency but needs more bars
    elif urgency >= 5 and uncertainty <= UNCERTAINTY_MED:
        signal = "WAIT"
    # Foggy: uncertainty too high regardless of urgency
    elif uncertainty >= UNCERTAINTY_HIGH:
        signal = "FOG"
    # Readable but weak
    elif urgency <= 3 and uncertainty <= UNCERTAINTY_LOW:
        signal = "PASS"
    else:
        signal = "WAIT"

    # Trader's equation adjustment
    if rr_ratio < 1.0 and signal in ("BUY_PULLBACK", "BUY_SPIKE"):
        signal = "WAIT"
    elif rr_ratio < 1.0 and signal == "WAIT":
        signal = "PASS"
    elif rr_ratio > 2.0 and signal == "WAIT":
        signal = "BUY_PULLBACK"
    elif rr_ratio > 2.0 and signal == "PASS":
        signal = "WAIT"

    # Mirror labels for gap-down
    if gap_direction == "down":
        if signal == "BUY_PULLBACK":
            signal = "SELL_PULLBACK"
        elif signal == "BUY_SPIKE":
            signal = "SELL_SPIKE"

    # ── BPA overlay (Fix 1): upgrade/downgrade based on pattern confirmation ──
    if BPA_INTEGRATION_ENABLED and bpa_alignment != 0.0:
        # Strong in-direction pattern upgrades WAIT/PASS to actionable signal
        if signal in ("WAIT", "PASS") and bpa_alignment >= 2.0:
            signal = "BUY_PULLBACK" if gap_direction == "up" else "SELL_PULLBACK"
        # Opposing pattern downgrades actionable BUY to WAIT
        elif bpa_alignment <= -1.0 and signal in ("BUY_PULLBACK", "BUY_SPIKE"):
            signal = "WAIT"

    # ── Bear-flip detection (Fix 3): intraday direction reversal ──
    if BPA_INTEGRATION_ENABLED:
        # Gap-up day where always_in flipped short + L1/L2 BPA confirmation
        if (gap_direction == "up"
                and always_in == "short"
                and uncertainty < UNCERTAINTY_TRAP
                and bpa_alignment <= -1.0):
            signal = SIGNAL_SELL_INTRADAY
        # Symmetric: gap-down day where always_in flipped long + H1/H2 BPA
        elif (gap_direction == "down"
                and always_in == "long"
                and uncertainty < UNCERTAINTY_TRAP
                and bpa_alignment >= 1.5):
            signal = SIGNAL_BUY_INTRADAY

    return signal


def _generate_summary(signal: str, urgency: float, uncertainty: float,
                      phase: str, always_in: str, gap_direction: str,
                      spike_bars: int, pullback_depth_pct: float,
                      gap_held: bool = True,
                      bpa_active_setups: list = None) -> str:
    """One-sentence Brooks-style summary."""
    direction = "bull" if gap_direction == "up" else "bear"
    bpa_count = len(bpa_active_setups) if bpa_active_setups else 0

    if signal == SIGNAL_SELL_INTRADAY:
        return (f"Intraday bear flip — gap-up day reversed. Always-in: short. "
                f"BPA {bpa_count} confirming setup(s). Urgency {urgency:.1f}.")

    if signal == SIGNAL_BUY_INTRADAY:
        return (f"Intraday bull flip on gap-down day. Always-in: long. "
                f"BPA {bpa_count} confirming setup(s). Urgency {urgency:.1f}.")

    if signal in ("BUY_PULLBACK", "SELL_PULLBACK"):
        return (f"Strong {direction} gap with {spike_bars}-bar spike, "
                f"shallow {pullback_depth_pct:.0%} pullback — "
                f"always-in {always_in}, good trader's equation for {phase} entry.")

    if signal in ("BUY_SPIKE", "SELL_SPIKE"):
        return (f"Powerful {direction} spike ({spike_bars} consecutive trend bars), "
                f"no pullback yet — market is leaving, consider market order with wide stop.")

    if signal == "AVOID":
        if urgency >= URGENCY_HIGH and uncertainty >= UNCERTAINTY_TRAP:
            return (f"Trap — {direction} gap looks urgent (U={urgency:.1f}) "
                    f"but chart is two-sided (uncertainty={uncertainty:.1f}), "
                    f"both sides showing strength. Most dangerous state.")
        if not gap_held:
            return (f"Gap filled — {direction} gap failed to hold prior close. "
                    f"Bears took control, no edge for longs.")
        return f"Chart unreadable — uncertainty={uncertainty:.1f}, avoid."

    if signal == "FOG":
        return (f"Can't read the chart — overlapping bars, dojis, "
                f"alternating colors (uncertainty={uncertainty:.1f}). Sit on hands.")

    if signal == "WAIT":
        return (f"Promising {direction} gap but need more bars — "
                f"urgency only {urgency:.1f}, wait for clearer pullback or breakout.")

    if signal == "PASS":
        return f"Readable but weak — no urgency, no edge. Pass."

    return f"Phase: {phase}, always-in: {always_in}. No clear setup."


# =============================================================================
# ATR HELPER
# =============================================================================

def _compute_daily_atr(daily_bars: pd.DataFrame, period: int = 20) -> float:
    """Compute Average Daily Range from daily OHLCV bars (high - low, no true-range gap)."""
    if daily_bars is None or len(daily_bars) < 2:
        return 0.0
    ranges = daily_bars["high"] - daily_bars["low"]
    return float(ranges.tail(period).mean())


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_gap(
    df: pd.DataFrame,
    prior_close: float,
    gap_direction: str,
    ticker: str = "UNKNOWN",
    avg_daily_range: float = None,
    daily_atr: float = None,
) -> dict:
    """
    Score a gap-up or gap-down stock using Al Brooks price action methodology.

    Parameters
    ----------
    df : pd.DataFrame
        5-min OHLCV bars from market open.
        Required columns: open, high, low, close, volume (volume optional).
    prior_close : float
        Friday/prior day's close price.
    gap_direction : str
        "up" or "down".
    ticker : str
        Ticker symbol for labeling.
    avg_daily_range : float, optional
        Average daily range in $ for opening range % calculation.
        If not provided, estimated from available data.
    daily_atr : float, optional
        20-day Average Daily Range (high-low) in dollars. Used for move_ratio
        and magnitude caps. Falls back to an intraday estimate if not provided.

    Returns
    -------
    dict with urgency, uncertainty, phase, signal, day_type, and supporting detail.
    """
    if len(df) < 3:
        return {
            "ticker": ticker,
            "urgency": 0.0,
            "uncertainty": 10.0,
            "phase": "insufficient_data",
            "always_in": "unclear",
            "signal": "FOG",
            "spike_bars": 0,
            "pullback_depth_pct": 0.0,
            "gap_held": False,
            "risk": 0.0,
            "reward": 0.0,
            "rr_ratio": 0.0,
            "summary": "Not enough bars to score.",
            "day_type": "undetermined",
            "day_type_confidence": 0.0,
            "day_type_warning": "Not enough bars to classify.",
            "opening_range_pct": 0.0,
            "details": {
                "spike_quality": 0.0, "gap_integrity": 0.0, "pullback_quality": 0.0,
                "follow_through": 0.0, "equation_check": 0.0,
            },
        }

    gap_direction = gap_direction.lower()
    assert gap_direction in ("up", "down"), f"gap_direction must be 'up' or 'down', got '{gap_direction}'"

    # ── Gap held check ──
    if gap_direction == "up":
        gap_held = df["low"].min() > prior_close
    else:
        gap_held = df["high"].max() < prior_close

    # ── Magnitude: daily ATR and move from open ───────────────────────────
    if daily_atr and daily_atr > 0:
        atr = daily_atr
    else:
        # Fallback: estimate daily range from intraday bars
        # avg 5-min bar range × 78 bars/day ≈ rough daily range
        atr = float((df["high"] - df["low"]).mean()) * 78
        atr = max(atr, 0.01)
    move_from_open = abs(float(df.iloc[-1]["close"]) - float(df.iloc[0]["open"]))
    move_ratio = move_from_open / atr if atr > 0 else 0.0

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Compute all raw component scores (11 urgency, 13 uncertainty)
    # ══════════════════════════════════════════════════════════════════════

    # Urgency components (raw, unweighted)
    spike_quality, spike_bars = _score_spike_quality(df, gap_direction)
    gap_integrity, gap_fill_status = _score_gap_integrity(df, prior_close, gap_direction)
    pullback_depth_pct, pullback_quality, pullback_exists = _find_first_pullback(df, spike_bars, gap_direction)
    follow_through = _score_follow_through(df, spike_bars, gap_direction)
    tail_quality = _score_tail_quality(df, spike_bars, gap_direction)
    body_gaps = _score_body_gaps(df, spike_bars, gap_direction)
    ma_separation = _score_ma_separation(df, gap_direction)
    failed_setups = _score_failed_counter_setups(df, gap_direction)
    volume_conf = _score_volume_confirmation(df, spike_bars)
    trending_swings = _score_trending_swings(df, gap_direction)
    spike_duration = _score_spike_duration(df, spike_bars, gap_direction)
    majority_trend_bars = _score_majority_trend_bars(df, gap_direction)
    micro_gaps = _score_micro_gaps(df, gap_direction)
    trending_everything = _score_trending_everything(df, gap_direction)
    levels_broken = _score_levels_broken(df, gap_direction, prior_close)
    spt_score = _score_small_pullback_trend(df, gap_direction)

    # Uncertainty (12 core components inside _score_uncertainty + 2 separate)
    uncertainty_base_raw, always_in = _score_uncertainty(df, gap_direction)
    two_sided_raw = _score_two_sided_ratio(df, gap_direction)
    liquidity_gaps_raw = _score_liquidity_gaps(df)

    # BPA pattern overlay (runs detectors, scores alignment with gap direction)
    bpa_alignment_score, bpa_active_setups = _score_bpa_patterns(df, gap_direction, gap_fill_status)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Opening range + day type classification
    # ══════════════════════════════════════════════════════════════════════

    or_info = _opening_range(df, avg_daily_range=avg_daily_range)
    # Use first 70% of bars for day-type classification so late-session chop
    # doesn't reclassify an early morning trend as trading_range.
    early_n = max(int(len(df) * 0.7), min(12, len(df)))
    two_sided_ratio = _compute_two_sided_ratio(df.iloc[:early_n], gap_direction)

    day_info = _classify_day_type(
        df, or_info, spike_bars, two_sided_ratio, gap_direction, gap_held,
    )
    day_type = day_info["day_type"]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Apply day-type weights to raw scores, then normalize
    # ══════════════════════════════════════════════════════════════════════

    w = lambda raw, name: _apply_day_type_weight(raw, name, day_type)

    # Weighted urgency (15 components)
    urgency_weighted = (
        w(spike_quality, "spike_quality")
        + w(gap_integrity, "gap_integrity")
        + w(pullback_quality, "pullback_quality")
        + w(follow_through, "follow_through")
        + w(tail_quality, "tail_quality")
        + w(body_gaps, "body_gaps")
        + w(ma_separation, "ma_separation")
        + w(failed_setups, "failed_setups")
        + w(volume_conf, "volume_conf")
        + w(trending_swings, "trending_swings")
        + w(spike_duration, "spike_duration")
        + w(majority_trend_bars, "majority_trend_bars")
        + w(micro_gaps, "micro_gaps")
        + w(trending_everything, "trending_everything")
        + w(levels_broken, "levels_broken")
        + w(spt_score, "small_pullback_trend")
    )

    # Weighted uncertainty additions (base has 12 internal components, these are separate)
    uncertainty_weighted = (uncertainty_base_raw
                            + w(two_sided_raw, "two_sided_ratio")
                            + w(liquidity_gaps_raw, "liquidity_gaps"))

    # Gap integrity failure feeds into uncertainty only when fill truly failed
    # (filled_recovered means post-fill structure was directional — not uncertain)
    if gap_integrity < 0 and gap_fill_status == "filled_failed":
        uncertainty_weighted += abs(gap_integrity)

    # Normalize → 0-10
    urgency = max(0.0, min(10.0, (max(urgency_weighted, 0.0) / URGENCY_RAW_MAX) * 10.0))
    uncertainty = max(0.0, min(10.0, (max(uncertainty_weighted, 0.0) / UNCERTAINTY_RAW_MAX) * 10.0))

    # ── Perfect 10 rule: zero closes on wrong side of EMA ─────────────────
    # Urgency can only reach 10.0 if no bar closes on the wrong side of the EMA.
    ema_20 = df["close"].ewm(span=20, adjust=False).mean()
    if gap_direction == "up":
        closes_wrong_side_ema = int((df["close"] < ema_20).sum())
    else:
        closes_wrong_side_ema = int((df["close"] > ema_20).sum())
    if closes_wrong_side_ema > 0 and urgency > 9.5:
        urgency = 9.5

    # ── Magnitude-based urgency caps (stacks with EMA rule) ───────────────
    if move_ratio < MAGNITUDE_CAP_9 and urgency > 9.0:
        urgency = 9.0
    if move_ratio < MAGNITUDE_CAP_10 and urgency > 9.5:
        urgency = 9.5

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Phase, risk/reward, signal, summary
    # ══════════════════════════════════════════════════════════════════════

    phase = _detect_phase(df, spike_bars, uncertainty, gap_direction, gap_held)

    # Two-pass R/R: detect probable bear-flip BEFORE computing R/R so the
    # risk/reward direction is correct for intraday shorts on gap-up days.
    tentative_flip_short = (
        BPA_INTEGRATION_ENABLED
        and gap_direction == "up"
        and always_in == "short"
        and bpa_alignment_score <= -1.0
    )
    tentative_flip_long = (
        BPA_INTEGRATION_ENABLED
        and gap_direction == "down"
        and always_in == "long"
        and bpa_alignment_score >= 1.5
    )
    if tentative_flip_short:
        rr_direction = "down"
    elif tentative_flip_long:
        rr_direction = "up"
    else:
        rr_direction = None  # use gap_direction default

    risk, reward, rr_ratio = _compute_risk_reward(
        df, gap_direction, prior_close, spike_bars,
        rr_direction_override=rr_direction,
    )

    if rr_ratio >= 2.0:
        equation_check = 2.0
    elif rr_ratio >= 1.5:
        equation_check = 1.0
    elif rr_ratio >= 1.0:
        equation_check = 0.0
    else:
        equation_check = -1.0

    signal = _determine_signal(
        urgency, uncertainty, gap_held, gap_direction,
        rr_ratio, spike_bars, pullback_exists,
        bpa_alignment=bpa_alignment_score,
        always_in=always_in,
        gap_fill_status=gap_fill_status,
    )

    summary = _generate_summary(
        signal, urgency, uncertainty, phase,
        always_in, gap_direction, spike_bars, pullback_depth_pct,
        gap_held=gap_held, bpa_active_setups=bpa_active_setups,
    )

    return {
        "ticker": ticker,
        "urgency": round(urgency, 1),
        "uncertainty": round(uncertainty, 1),
        "phase": phase,
        "always_in": always_in,
        "signal": signal,
        "spike_bars": spike_bars,
        "pullback_depth_pct": round(pullback_depth_pct, 3),
        "gap_held": gap_held,
        "risk": risk,
        "reward": reward,
        "rr_ratio": rr_ratio,
        "summary": summary,
        # Phase 1 additions
        "day_type": day_type,
        "day_type_confidence": day_info["confidence"],
        "day_type_warning": day_info["warning"],
        "opening_range_pct": or_info["range_pct"],
        "daily_atr": round(atr, 4),
        "move_from_open": round(move_from_open, 4),
        "move_ratio": round(move_ratio, 3),
        "details": {
            # Urgency components (15)
            "spike_quality": round(spike_quality, 2),
            "gap_integrity": round(gap_integrity, 2),
            "gap_fill_status": gap_fill_status,
            "pullback_quality": round(pullback_quality, 2),
            "follow_through": round(follow_through, 2),
            "tail_quality": round(tail_quality, 2),
            "body_gaps": round(body_gaps, 2),
            "ma_separation": round(ma_separation, 2),
            "failed_setups": round(failed_setups, 2),
            "volume_conf": round(volume_conf, 2),
            "trending_swings": round(trending_swings, 2),
            "spike_duration": round(spike_duration, 2),
            "majority_trend_bars": round(majority_trend_bars, 2),
            "micro_gaps": round(micro_gaps, 2),
            "trending_everything": round(trending_everything, 2),
            "levels_broken": round(levels_broken, 2),
            "small_pullback_trend": round(spt_score, 2),
            # Uncertainty additions
            "two_sided_ratio": round(two_sided_raw, 2),
            "liquidity_gaps": round(liquidity_gaps_raw, 2),
            # Trader's equation
            "equation_check": round(equation_check, 2),
            # Weighted totals for debugging
            "urgency_weighted": round(urgency_weighted, 2),
            "uncertainty_weighted": round(uncertainty_weighted, 2),
            # Perfect 10 rule
            "closes_wrong_side_ema": closes_wrong_side_ema,
            # Day type weight multipliers used
            "day_type_applied": day_type,
            # BPA pattern overlay
            "bpa_alignment": round(bpa_alignment_score, 2),
            "bpa_setups": bpa_active_setups[:3],
            # Cycle-phase classifier (Layer 1, rolling)
            "cycle_phase": classify_cycle_phase(df),
            # Session-shape classifier (Layer 2, full session)
            "session_shape": classify_session_shape(
                df,
                gap_direction,
                spike_bars=spike_bars,
                session_minutes=int((df.iloc[-1].get("datetime") - df.iloc[0].get("datetime")).total_seconds() / 60)
                    if "datetime" in df.columns and len(df) > 1 else len(df) * 5,
            ),
        },
    }


# =============================================================================
# CONVENIENCE: Score multiple gaps and rank them
# =============================================================================

def score_multiple(
    gaps: list[dict],
    df_dict: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Score and rank a list of gap stocks.

    Parameters
    ----------
    gaps : list[dict]
        Each dict must have: ticker, prior_close, gap_direction.
    df_dict : dict[str, pd.DataFrame]
        Mapping of ticker -> DataFrame of 5-min bars.

    Returns
    -------
    list[dict] sorted by urgency desc, uncertainty asc. Best setups first.
    """
    results = []
    for gap in gaps:
        ticker = gap["ticker"]
        if ticker not in df_dict:
            logger.warning(f"No bar data for {ticker}, skipping.")
            continue
        result = score_gap(
            df=df_dict[ticker],
            prior_close=gap["prior_close"],
            gap_direction=gap["gap_direction"],
            ticker=ticker,
        )
        results.append(result)

    # Sort: highest urgency first, then lowest uncertainty
    results.sort(key=lambda r: (-r["urgency"], r["uncertainty"]))
    return results


# =============================================================================
# FULL UNIVERSE SCANNER
# =============================================================================

# Default 5-min resample from 1-min bars for scoring
SCAN_BAR_SCHEMA = "ohlcv-1m"
SCAN_RESAMPLE = "5min"

def _get_default_universe() -> list[str]:
    """
    Import the 498-symbol universe from the screener.
    Falls back to a small default if screener isn't available.
    """
    try:
        from stages.screener import _default_gap_universe
        return _default_gap_universe()
    except ImportError:
        logger.warning("Could not import screener universe, using fallback S&P leaders")
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
            "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "AVGO", "CVX",
            "LLY", "MRK", "ABBV", "PEP", "KO", "COST", "ADBE", "CRM", "WMT",
            "TMO", "ACN", "MCD", "CSCO", "ABT", "DHR", "LIN", "NEE", "TXN",
            "AMGN", "PM", "RTX", "LOW", "UNP", "HON", "IBM", "QCOM", "SPGI",
            "AMAT", "DE", "GE", "CAT", "BKNG", "NOW", "ISRG", "ADP", "MRVL",
            "SPY", "QQQ", "IWM", "SMH", "XLE", "XLF",
        ]


def _normalize_databento_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Databento DataFrame column names to lowercase.
    Handles the ts_event index and symbol column that Databento returns.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    return df


def _resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min bars to 5-min bars for scoring.
    Input must have columns: open, high, low, close, volume.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df_1m.columns:
        agg["volume"] = "sum"

    df_5m = df_1m.resample(SCAN_RESAMPLE).agg(agg).dropna(subset=["open", "close"])
    return df_5m


def scan_universe(
    tickers: list[str] = None,
    min_urgency: float = 3.0,
    max_uncertainty: float = 7.0,
    min_dollar_vol: float = LIQUIDITY_MIN_DOLLAR_VOL,
    databento_key: str = None,
    verbose: bool = False,
) -> list[dict]:
    """
    Score every stock in the universe and return a ranked leaderboard.

    Two Databento API calls total (NOT per-symbol):
      1. ohlcv-1d for all symbols → prior close
      2. ohlcv-1m for all symbols → today's 5-min bars (resampled internally)

    Parameters
    ----------
    tickers : list[str], optional
        Symbols to scan. Defaults to the full 498-symbol universe.
    min_urgency : float
        Filter out anything below this urgency score (default 3.0).
    max_uncertainty : float
        Filter out anything above this uncertainty score (default 7.0).
    databento_key : str, optional
        Databento API key. Falls back to DATABENTO_API_KEY env var.
    verbose : bool
        If True, log progress per symbol.

    Returns
    -------
    list[dict]
        Ranked list of score_gap() outputs, sorted by urgency desc.
        Only includes stocks passing the urgency/uncertainty filters.
    """
    import os
    import time as _time
    from datetime import datetime, timedelta, timezone

    # Lazy import — DatabentClient lives in the same shared/ package
    from shared.databento_client import DatabentClient, _safe_end

    if tickers is None:
        tickers = _get_default_universe()

    api_key = databento_key or os.environ.get("DATABENTO_API_KEY")
    client = DatabentClient(api_key=api_key)

    now_utc = datetime.now(timezone.utc)
    today_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # Detect ET offset for market open time
    is_dst = bool(_time.localtime().tm_isdst)
    utc_offset_hours = 4 if is_dst else 5
    market_open_utc = today_midnight_utc.replace(
        hour=13 + (0 if utc_offset_hours == 4 else 1), minute=30
    )

    safe_end = _safe_end()

    # ── QUERY 1: Prior day's close (ohlcv-1d, all symbols, last 5 days) ──
    logger.info(f"Scanning {len(tickers)} symbols — fetching daily bars (30 days for ADR)...")
    try:
        daily_df = client.query_ohlcv(
            dataset="EQUS.MINI",
            symbols=tickers,
            schema="ohlcv-1d",
            start=today_midnight_utc - timedelta(days=30),  # ~20 trading days
            end=today_midnight_utc,
        )
    except Exception as e:
        logger.error(f"Daily data query failed: {e}")
        return []

    if daily_df.empty:
        logger.error("No daily data returned")
        return []

    daily_df = _normalize_databento_df(daily_df)

    # Compute 20-day Average Daily Range per symbol for magnitude scoring
    daily_atrs: dict[str, float] = {}
    if "symbol" in daily_df.columns:
        for sym in tickers:
            sym_d = daily_df[daily_df["symbol"] == sym]
            daily_atrs[sym] = _compute_daily_atr(sym_d)
    else:
        # Single-symbol query
        if len(tickers) == 1:
            daily_atrs[tickers[0]] = _compute_daily_atr(daily_df)

    # ── QUERY 2: Today's 1-min bars from market open (all symbols) ──
    if safe_end <= market_open_utc:
        logger.warning("Market hasn't opened yet or data not available")
        return []

    logger.info(f"Fetching intraday 1-min bars ({market_open_utc.strftime('%H:%M')} → {safe_end.strftime('%H:%M')} UTC)...")
    try:
        intra_df = client.query_ohlcv(
            dataset="EQUS.MINI",
            symbols=tickers,
            schema=SCAN_BAR_SCHEMA,
            start=market_open_utc,
            end=safe_end,
        )
    except Exception as e:
        logger.error(f"Intraday data query failed: {e}")
        return []

    if intra_df.empty:
        logger.error("No intraday data returned")
        return []

    intra_df = _normalize_databento_df(intra_df)

    # ── PARSE & SCORE each symbol ──
    results = []
    skipped = 0

    for ticker in tickers:
        try:
            # Extract this symbol's daily data → prior close
            if "symbol" in daily_df.columns:
                sym_daily = daily_df[daily_df["symbol"] == ticker]
            else:
                # Single-symbol query won't have a symbol column
                sym_daily = daily_df if len(tickers) == 1 else pd.DataFrame()

            if sym_daily.empty:
                skipped += 1
                continue

            prior_close = float(sym_daily["close"].iloc[-1])
            if prior_close <= 0:
                skipped += 1
                continue

            # Extract this symbol's intraday data
            if "symbol" in intra_df.columns:
                sym_intra = intra_df[intra_df["symbol"] == ticker].copy()
            else:
                sym_intra = intra_df.copy() if len(tickers) == 1 else pd.DataFrame()

            if sym_intra.empty or len(sym_intra) < 5:
                skipped += 1
                continue

            # Drop the symbol column before resampling (keep only OHLCV)
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"]
                          if c in sym_intra.columns]
            sym_intra = sym_intra[ohlcv_cols]

            # Resample 1-min → 5-min
            df_5m = _resample_to_5min(sym_intra)
            if len(df_5m) < 3:
                skipped += 1
                continue

            # Hard liquidity filter: skip illiquid stocks
            liq = _check_liquidity(df_5m)
            if not liq["passed"] and min_dollar_vol > 0:
                if verbose:
                    logger.info(f"  {ticker:6s}  SKIP — avg $vol/bar ${liq['avg_dollar_vol']:,.0f} < ${min_dollar_vol:,.0f}")
                skipped += 1
                continue

            # Infer gap direction: today's open vs prior close
            today_open = float(df_5m.iloc[0]["open"])
            gap_direction = "up" if today_open > prior_close else "down"
            gap_pct = (today_open - prior_close) / prior_close * 100

            # Score it
            result = score_gap(
                df=df_5m.reset_index(drop=True),  # score_gap expects integer index
                prior_close=prior_close,
                gap_direction=gap_direction,
                ticker=ticker,
                daily_atr=daily_atrs.get(ticker),
            )

            # Attach gap % and liquidity info for the leaderboard display
            result["gap_pct"] = round(gap_pct, 2)
            result["today_open"] = round(today_open, 2)
            result["prior_close_price"] = round(prior_close, 2)
            result["bars_scored"] = len(df_5m)
            result["liquidity"] = liq

            # Apply filters
            if result["urgency"] < min_urgency:
                continue
            if result["uncertainty"] > max_uncertainty:
                continue
            if result.get("move_ratio", 0) < MAGNITUDE_FLOOR:
                continue

            results.append(result)

            if verbose:
                logger.info(
                    f"  {ticker:6s}  U={result['urgency']:4.1f}  "
                    f"Unc={result['uncertainty']:4.1f}  {result['signal']:16s}  "
                    f"Gap={gap_pct:+.1f}%"
                )

        except Exception as e:
            logger.debug(f"Scoring failed for {ticker}: {e}")
            skipped += 1

    # Sort: urgency desc, uncertainty asc
    results.sort(key=lambda r: (-r["urgency"], r["uncertainty"]))

    logger.info(
        f"Scan complete: {len(results)} stocks passed filters, "
        f"{skipped} skipped (no data or filtered out)"
    )

    return results


# =============================================================================
# DEMO with synthetic data
# =============================================================================

def _make_bars(specs: list[tuple], start_price: float) -> pd.DataFrame:
    """
    Build a synthetic DataFrame from bar specs.
    Each spec: (direction, body_pct, range_size)
      direction: "bull" or "bear" or "doji_bull" or "doji_bear"
      body_pct: body as fraction of range (0-1)
      range_size: absolute range of the bar
    Doji variants let us control whether close > open (doji_bull) or close < open (doji_bear)
    so color alternation scoring works correctly on synthetic data.
    """
    rows = []
    price = start_price
    for i, (direction, body_pct, range_size) in enumerate(specs):
        body = range_size * body_pct
        tail_total = range_size - body

        if direction == "bull":
            low = price - tail_total * 0.3
            high = low + range_size
            o = low + tail_total * 0.2
            c = high - tail_total * 0.1
            price = c
        elif direction == "bear":
            high = price + tail_total * 0.3
            low = high - range_size
            o = high - tail_total * 0.2
            c = low + tail_total * 0.1
            price = c
        elif direction == "doji_bull":
            # Doji with close slightly above open (registers as bull for color)
            mid = price
            high = mid + range_size / 2
            low = mid - range_size / 2
            o = mid - body / 2
            c = mid + body / 2
            price = mid
        else:  # doji_bear
            mid = price
            high = mid + range_size / 2
            low = mid - range_size / 2
            o = mid + body / 2
            c = mid - body / 2
            price = mid

        rows.append({
            "datetime": pd.Timestamp("2026-04-13 09:30") + pd.Timedelta(minutes=5 * i),
            "open": round(o, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(c, 2),
            "volume": 50000 + i * 1000,
        })

    return pd.DataFrame(rows)


def _demo_now_like():
    """
    NOW-like: Strong bull gap-up with clean spike, shallow pullback, early follow-through.
    Should score urgency ~7+, uncertainty ~2-3, signal = BUY_PULLBACK.
    """
    prior_close = 870.00
    gap_open = 885.00  # ~1.7% gap

    specs = [
        # Spike: 4 strong bull bars — consecutive, big bodies, small tails
        ("bull", 0.80, 4.00),  # bar 1: strong bull, big body
        ("bull", 0.75, 3.50),  # bar 2: strong bull
        ("bull", 0.70, 3.00),  # bar 3: strong bull
        ("bull", 0.65, 2.80),  # bar 4: strong bull
        # Shallow pullback: 2 small bear bars retracing ~25% of spike
        ("bear", 0.40, 1.80),  # bar 5: small bear, weak selling
        ("bear", 0.35, 1.20),  # bar 6: even smaller bear — buyers absorbing
        # Early follow-through: higher low forming, just starting to push back up
        ("bull", 0.65, 2.50),  # bar 7: bull resumes — higher low confirmed
        ("bull", 0.70, 2.80),  # bar 8: strong bull — approaching spike high
    ]

    df = _make_bars(specs, gap_open)
    return df, prior_close


def _demo_xle_like():
    """
    XLE-like: Weak gap-up that goes nowhere. Rapid color alternation, dojis, two-sided.
    Should score urgency ~2, uncertainty ~5+, signal = FOG.
    """
    prior_close = 60.00
    gap_open = 60.80  # small gap

    specs = [
        # No real spike — weak opening, immediately two-sided
        ("bull", 0.40, 0.50),       # bar 1: weak bull
        ("bear", 0.45, 0.55),       # bar 2: bear — selling right away
        ("bull", 0.30, 0.40),       # bar 3: weak bull response
        ("bear", 0.50, 0.70),       # bar 4: bear larger than bulls
        ("doji_bull", 0.12, 0.60),  # bar 5: doji — indecision
        ("bear", 0.55, 0.65),       # bar 6: bear — sellers persist
        ("bull", 0.35, 0.50),       # bar 7: weak bull
        ("bear", 0.45, 0.60),       # bar 8: bear again
        ("doji_bear", 0.10, 0.45),  # bar 9: doji
        ("bull", 0.30, 0.40),       # bar 10: weak bull — no conviction either way
        ("bear", 0.40, 0.55),       # bar 11: bear
        ("doji_bull", 0.15, 0.50),  # bar 12: doji — completely stuck
    ]

    df = _make_bars(specs, gap_open)
    return df, prior_close


def _demo_mrvl_like():
    """
    MRVL-like: Decent initial 2-bar move, then stalls and becomes two-sided.
    Should score urgency ~3-4, uncertainty ~5+, signal = FOG.
    """
    prior_close = 65.00
    gap_open = 67.00  # decent gap

    specs = [
        # Decent start — 2 bull bars but not a strong spike
        ("bull", 0.65, 1.20),       # bar 1: decent bull bar
        ("bull", 0.55, 0.90),       # bar 2: weaker bull — steam fading
        # Sellers show up — strong counter-move
        ("bear", 0.70, 1.30),       # bar 3: strong bear bar — sellers showed up hard
        ("bear", 0.65, 1.10),       # bar 4: bear continuation — this is a bear spike
        # Now it's a fight — alternating, overlapping, foggy
        ("bull", 0.40, 0.70),       # bar 5: weak bull bounce
        ("bear", 0.50, 0.80),       # bar 6: bear pushes back
        ("bull", 0.35, 0.60),       # bar 7: weak bull
        ("doji_bear", 0.10, 0.55),  # bar 8: doji — stuck
        ("bear", 0.45, 0.75),       # bar 9: bear nudge
        ("bull", 0.40, 0.65),       # bar 10: bull nudge
        ("doji_bull", 0.12, 0.50),  # bar 11: doji
        ("bear", 0.50, 0.70),       # bar 12: bear — range continues
    ]

    df = _make_bars(specs, gap_open)
    return df, prior_close


def _demo_gap_chop():
    """
    Gap-and-chop: Opens up 8.7% from prior close, then trades sideways.
    The open is in the MIDDLE of the day's range — NOT at the extreme.
    Should classify as trading_range despite the large gap from prior close.
    """
    prior_close = 92.00
    gap_open = 100.00  # 8.7% gap

    specs = [
        # Opens at 100, immediately sells to 99 then bounces to 101 — range is 99-101
        ("bear", 0.45, 0.60),       # bar 1: sells off from open
        ("bear", 0.40, 0.50),       # bar 2: continues down
        ("bull", 0.35, 0.55),       # bar 3: bounce
        ("bull", 0.40, 0.50),       # bar 4: bounce continues
        ("bear", 0.45, 0.55),       # bar 5: back down — range forming
        ("doji_bull", 0.12, 0.40),  # bar 6: doji at midrange
        ("bull", 0.35, 0.45),       # bar 7: slight push up
        ("bear", 0.40, 0.50),       # bar 8: back down
        ("bull", 0.30, 0.40),       # bar 9: weak bull
        ("bear", 0.35, 0.45),       # bar 10: weak bear — alternating
        ("doji_bear", 0.10, 0.35),  # bar 11: doji
        ("bull", 0.40, 0.50),       # bar 12: push up
        ("bear", 0.45, 0.55),       # bar 13: back down — stuck
        ("bull", 0.30, 0.40),       # bar 14: weak
        ("doji_bull", 0.15, 0.35),  # bar 15: doji
        ("bear", 0.35, 0.45),       # bar 16: weak bear
        ("bull", 0.40, 0.50),       # bar 17: alternating
        ("bear", 0.30, 0.40),       # bar 18: more chop
        ("doji_bear", 0.10, 0.35),  # bar 19: doji
        ("bull", 0.35, 0.45),       # bar 20: more chop
    ]

    df = _make_bars(specs, gap_open)
    return df, prior_close


def _demo_orcl_like():
    """
    ORCL-like: Strong trend from open. Open sets the low, 80% trend bars,
    small pullbacks, then some late-session chop (normal on trend days).
    Should classify as trend_from_open or spike_and_channel.
    """
    prior_close = 150.00
    gap_open = 167.72  # +11.81% gap

    specs = [
        # Strong bull spike from open — 8 bars
        ("bull", 0.75, 1.80),
        ("bull", 0.70, 1.60),
        ("bull", 0.75, 1.70),
        ("bull", 0.70, 1.50),
        ("bull", 0.65, 1.40),
        ("bull", 0.70, 1.60),
        ("bull", 0.65, 1.30),
        ("bull", 0.70, 1.50),
        # Tiny pullback — 2 bars
        ("bear", 0.35, 0.80),
        ("bear", 0.30, 0.60),
        # Resumes — 6 bars of channel continuation
        ("bull", 0.65, 1.30),
        ("bull", 0.60, 1.20),
        ("bull", 0.65, 1.10),
        ("bull", 0.55, 1.00),
        ("bull", 0.60, 1.10),
        ("bull", 0.55, 0.90),
        # Another small pullback
        ("bear", 0.40, 0.90),
        ("bull", 0.60, 1.00),
        # More continuation
        ("bull", 0.55, 1.00),
        ("bull", 0.60, 1.10),
        ("bull", 0.50, 0.90),
        # Late-session chop (last 30%) — normal on trend days
        ("bear", 0.50, 1.00),
        ("bull", 0.40, 0.80),
        ("bear", 0.45, 0.90),
        ("bull", 0.35, 0.70),
        ("bear", 0.40, 0.80),
        ("doji_bull", 0.15, 0.60),
    ]

    df = _make_bars(specs, gap_open)
    return df, prior_close


def _run_demo():
    """Run the synthetic data demo (original __main__ behavior)."""
    import json

    print("=" * 80)
    print("BROOKS GAP SCORER — DEMO (synthetic data)")
    print("=" * 80)

    demos = [
        ("NOW-like (strong bull gap, clean spike)", _demo_now_like, "up"),
        ("XLE-like (weak gap, overlapping, foggy)", _demo_xle_like, "up"),
        ("MRVL-like (decent gap, stalls, two-sided)", _demo_mrvl_like, "up"),
        ("GAP-CHOP (big gap, then sideways)", _demo_gap_chop, "up"),
        ("ORCL-like (trend from open, late chop)", _demo_orcl_like, "up"),
    ]

    all_gaps = []
    all_dfs = {}

    for label, demo_fn, direction in demos:
        df, prior_close = demo_fn()
        ticker = label.split("(")[0].strip().replace("-like", "").strip()

        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"  Prior close: ${prior_close:.2f}  |  Gap open: ${df.iloc[0]['open']:.2f}  |  Bars: {len(df)}")
        print(f"{'─' * 80}")

        result = score_gap(df, prior_close, direction, ticker=ticker)

        print(f"  SIGNAL:      {result['signal']}")
        print(f"  Urgency:     {result['urgency']}/10")
        print(f"  Uncertainty: {result['uncertainty']}/10")
        print(f"  Day type:    {result['day_type']} (conf={result['day_type_confidence']:.0%})")
        print(f"  Warning:     {result['day_type_warning']}")
        print(f"  OR %:        {result['opening_range_pct']:.0%}")
        print(f"  Phase:       {result['phase']}")
        print(f"  Always-in:   {result['always_in']}")
        print(f"  Spike bars:  {result['spike_bars']}")
        print(f"  Pullback:    {result['pullback_depth_pct']:.1%} of spike")
        print(f"  Gap held:    {result['gap_held']}")
        print(f"  Risk:        ${result['risk']:.2f}")
        print(f"  Reward:      ${result['reward']:.2f}")
        print(f"  R:R:         {result['rr_ratio']:.1f}")
        print(f"  Summary:     {result['summary']}")
        print(f"  Details:     {json.dumps(result['details'], indent=2)}")

        all_gaps.append({"ticker": ticker, "prior_close": prior_close, "gap_direction": direction})
        all_dfs[ticker] = df

    print(f"\n{'=' * 80}")
    print("RANKED (best setup first):")
    print(f"{'=' * 80}")
    ranked = score_multiple(all_gaps, all_dfs)
    for i, r in enumerate(ranked, 1):
        dt = r.get("day_type", "?")
        print(f"  {i}. {r['ticker']:8s}  {r['signal']:16s}  "
              f"U={r['urgency']:4.1f}  Unc={r['uncertainty']:4.1f}  "
              f"R:R={r['rr_ratio']:4.1f}  {dt:<18}  {r['phase']}")


def _run_scan(args):
    """Run the live universe scan with Databento data."""
    from datetime import datetime
    import pytz

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)

    print(f"\n{'=' * 80}")
    print(f"  STRONGEST STOCKS — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"  Filters: urgency >= {args.min_urgency}  |  uncertainty <= {args.max_uncertainty}  |  min $vol/bar >= ${args.min_dollar_vol:,.0f}")
    print(f"{'=' * 80}\n")

    # Parse custom ticker list if provided
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        print(f"  Custom universe: {len(tickers)} symbols\n")

    results = scan_universe(
        tickers=tickers,
        min_urgency=args.min_urgency,
        max_uncertainty=args.max_uncertainty,
        min_dollar_vol=args.min_dollar_vol,
        verbose=args.verbose,
    )

    if not results:
        print("  No stocks passed the filters. Try lowering --min-urgency or raising --max-uncertainty.\n")
        return

    # Truncate to --top N
    top_n = results[:args.top]

    # Print leaderboard table
    header = (f"  {'#':>3}  {'Ticker':<7}  {'Urgency':>7}  {'Uncert':>6}  "
              f"{'Signal':<16}  {'DayType':<18}  {'Gap%':>6}  {'R:R':>5}  Warning")
    print(header)
    print(f"  {'─' * len(header)}")

    for i, r in enumerate(top_n, 1):
        gap_str = f"{r.get('gap_pct', 0):+.1f}%"
        dt = r.get("day_type", "?")
        warning_short = r.get("day_type_warning", "")[:55]
        if len(r.get("day_type_warning", "")) > 55:
            warning_short += "…"
        print(
            f"  {i:>3}  {r['ticker']:<7}  {r['urgency']:>7.1f}  {r['uncertainty']:>6.1f}  "
            f"{r['signal']:<16}  {dt:<18}  {gap_str:>6}  {r['rr_ratio']:>5.1f}  "
            f"{warning_short}"
        )

    print(f"\n  {len(results)} total stocks passed filters | showing top {len(top_n)}")
    print(f"  Data: EQUS.MINI ohlcv-1d + ohlcv-1m via Databento\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Brooks Price Action Gap Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shared/brooks_score.py                          # Run synthetic demo
  python shared/brooks_score.py --mode scan              # Scan full 498-symbol universe
  python shared/brooks_score.py --mode scan --top 10     # Show top 10 only
  python shared/brooks_score.py --mode scan --min-urgency 5 --max-uncertainty 4
  python shared/brooks_score.py --mode scan --tickers "AAPL,NVDA,NOW,MRVL,TSLA"
        """,
    )
    parser.add_argument(
        "--mode", choices=["demo", "scan"], default="demo",
        help="'demo' runs synthetic examples, 'scan' runs live Databento scan (default: demo)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Show top N results in scan mode (default: 20)",
    )
    parser.add_argument(
        "--min-urgency", type=float, default=3.0,
        help="Minimum urgency score to include (default: 3.0)",
    )
    parser.add_argument(
        "--max-uncertainty", type=float, default=7.0,
        help="Maximum uncertainty score to include (default: 7.0)",
    )
    parser.add_argument(
        "--min-dollar-vol", type=float, default=LIQUIDITY_MIN_DOLLAR_VOL,
        help=f"Min avg dollar volume per 5-min bar (default: ${LIQUIDITY_MIN_DOLLAR_VOL:,.0f})",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated list of tickers to scan (default: full 498 universe)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log per-symbol scoring progress",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        _run_demo()
    elif args.mode == "scan":
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
        _run_scan(args)
