"""Day-type classifier (Brooks taxonomy).

Structural analysis of the full session — classifies into one of six
Brooks day types:
    trend_from_open, spike_and_channel, trending_tr, trading_range,
    tight_tr, undetermined

Classification hinges on WHERE THE OPEN SITS in the day's range and
how bars are distributed — NOT on net move from prior close. A stock
can gap 10% and still be a trading range day if it opens and then
goes sideways.

Also exposes `DAY_TYPE_WEIGHTS` — a per-day-type multiplier matrix
applied to raw component scores before normalization — and
`_apply_day_type_weight`, the helper that applies it.

Extracted from shared/brooks_score.py (Phase 3e).
"""

import numpy as np
import pandas as pd

from aiedge.features.candles import (
    MIN_RANGE,
    _body,
    _body_ratio,
    _is_bear,
    _is_bull,
)


# ── Tunables ─────────────────────────────────────────────────────────
# Body-ratio threshold for a "strong trend bar". Shared with scoring
# components (STRONG_BODY_RATIO); lives here for now and will migrate
# to signals/components.py in Phase 3f.
STRONG_BODY_RATIO = 0.60

# Minimum bars before the classifier will return anything but
# `undetermined`. Paired warm-up behavior for the early session.
WARMUP_BARS = 7

# Minimum spike-bar count for spike-and-channel recognition. Shared
# with phase detection; will migrate to signals/components.py in
# Phase 3f alongside STRONG_BODY_RATIO.
SPIKE_MIN_BARS = 3

# Opening-range width thresholds (fraction of average bar range):
#   < OR_TRENDING_TR_LOW  → narrow OR, open set the extreme
#   [OR_TRENDING_TR_LOW, OR_TRENDING_TR_HIGH] → trending trading range
#   > OR_TRADING_RANGE    → wide OR, probable trading range day
OR_TRADING_RANGE = 0.50
OR_TRENDING_TR_LOW = 0.25
OR_TRENDING_TR_HIGH = 0.50

# Late-session range / full-day range cutoff. Below this, a late
# "range" is really just a pullback or consolidation inside a trend
# — not a true range day. Used to flip trading_range → trend_from_open
# or spike_and_channel when the late chop is trivially small.
CHOP_RATIO_THRESHOLD = 0.25


def _compute_two_sided_ratio(df: pd.DataFrame, gap_direction: str) -> float:
    """Ratio of countertrend bars to total bars over the ENTIRE bar set.

    A stock that trended for 80% of bars and chopped for 20% should
    have a LOW ratio, not a high one. Using all bars prevents late-
    session chop from dominating the reading.
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
    """Classify day type using Brooks' taxonomy via structural analysis.

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
