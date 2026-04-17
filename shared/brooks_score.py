"""
Brooks Price Action Gap Scorer — library entry points + compat shim
===================================================================

Three top-level functions:

  - score_gap(df, prior_close, gap_direction, ...)     — score one symbol
  - score_multiple(gaps, dfs)                          — score N symbols
  - scan_universe(tickers, ...)                        — full Databento scan

Under the hood the work is done by modules under `aiedge/`:

  features/  — bar-math primitives (candles, EMA, swings, volatility, session)
  context/   — day-type, cycle-phase, session-shape classifiers
  signals/   — 18 scoring components + aggregator + BPA overlay + summary
  risk/      — trader's-equation risk / reward / R:R
  data/      — Databento normalization, resampling, universe

This file preserves the `from shared.brooks_score import X` API so
existing callers (live_scanner, pattern_lab_api, tools/*, tests) don't
need to change. The long list of re-imports below is the compat layer;
everything the scanner actually does now lives in `aiedge/`.

CLI demo + live-scan entry point moved to `bin/brooks_score_cli.py`
in Phase 3j.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# BPA detector integration (_bpa_detect_all, _BPA_AVAILABLE) and the
# _score_bpa_patterns function now live in aiedge.signals.bpa.

# =============================================================================
# TUNABLE CONSTANTS — adjust these to calibrate scoring sensitivity
# =============================================================================

# Urgency scoring component constants (CLOSE_TOP_PCT, TAIL_*, GAP_*, pullback
# thresholds, BODY_GAP_BONUS, MA_GAP_BARS_*, FAILED_SETUP_BONUS, VOLUME_SPIKE_*,
# TRENDING_SWINGS_*, SPIKE_DURATION_*, SPIKE_RETRACE_LIMIT, SPT_*, URGENCY_RAW_MAX)
# now live in aiedge.signals.components (re-imported below in Phase 3f-1).

# Uncertainty + liquidity scoring constants (UNCERTAINTY_RAW_MAX, COLOR_ALT_HIGH,
# DOJI_RATIO_HIGH, BODY_OVERLAP_HIGH, BEAR_SPIKE_RATIO, BARS_STUCK_THRESHOLD,
# MIDPOINT_TOLERANCE, STRONG_TREND_WINDOW, UNCERTAINTY_ANALYSIS_WINDOW, REVERSAL_*,
# TIGHT_RANGE_*, TWO_SIDED_*, LIQUIDITY_*) now live in
# aiedge.signals.components (re-imported below in Phase 3f-2).

# ── Opening Range Analysis ──
# OPENING_RANGE_BARS now lives in aiedge.features.session (re-imported below).
# OR_TRENDING_TR_LOW, OR_TRENDING_TR_HIGH, OR_TRADING_RANGE now live in
# aiedge.context.daytype (re-imported below).
OR_TREND_FROM_OPEN = 0.25      # OR < 25% of avg range = trend from open

# ── Day Type Classifier ──
# WARMUP_BARS now lives in aiedge.context.daytype (re-imported below).

# ── Magnitude filter ──
MAGNITUDE_FLOOR = 0.5         # Min move/ATR to appear on leaderboard
MAGNITUDE_CAP_9 = 0.7         # Min move/ATR for urgency > 9.0
MAGNITUDE_CAP_10 = 1.0        # Min move/ATR for urgency > 9.5 (stacks with EMA rule)
# CHOP_RATIO_THRESHOLD now lives in aiedge.context.daytype (re-imported below).

# Signal decision thresholds (URGENCY_HIGH, UNCERTAINTY_LOW/MED/HIGH/TRAP),
# phase constants (FAILED_GAP_MIN_FRAC_ADR), and intraday flip labels
# (SIGNAL_SELL_INTRADAY, SIGNAL_BUY_INTRADAY) now live in
# aiedge.signals.aggregator (re-imported below).

# BPA pattern integration constants (BPA_INTEGRATION_ENABLED, BPA_LONG_SETUP_TYPES,
# BPA_SHORT_SETUP_TYPES, BPA_COUNTER_TYPES, BPA_MIN_CONFIDENCE, BPA_RECENCY_BARS,
# BPA_MIN_DF_LEN) now live in aiedge.signals.bpa (re-imported below).

TRADING_RANGE_OVERLAP_BARS = 10

# MIN_RANGE and the candle helpers (_safe_range, _body, _body_ratio,
# _is_bull, _is_bear, _lower_tail_pct, _upper_tail_pct, _close_position)
# now live in aiedge.features.candles. Re-imported below for internal
# callers in this file and for any external consumer that still does
# `from shared.brooks_score import _body`.

# GAP_INTEGRITY_POST_FILL_EVAL now lives in aiedge.signals.components (re-imported below).


# =============================================================================
# HELPER FUNCTIONS — candle/bar math extracted to aiedge.features.candles
# =============================================================================
from aiedge.features.candles import (  # noqa: E402  (import after constants)
    MIN_RANGE,
    DOJI_BODY_RATIO,
    _safe_range,
    _body,
    _body_ratio,
    _is_bull,
    _is_bear,
    _lower_tail_pct,
    _upper_tail_pct,
    _close_position,
)
from aiedge.features.ema import EMA_PERIOD, _compute_ema  # noqa: E402
from aiedge.features.swings import _find_swing_lows, _find_swing_highs  # noqa: E402
from aiedge.features.volatility import _compute_daily_atr  # noqa: E402
from aiedge.features.session import OPENING_RANGE_BARS, _opening_range  # noqa: E402
from aiedge.context.phase import (  # noqa: E402
    CYCLE_PHASE_CLASSIFIER_ENABLED,
    CYCLE_PHASE_LOOKBACK_BARS,
    CYCLE_PHASE_SOFTMAX_TEMP,
    CYCLE_PHASES,
    _cycle_bull_spike_raw,
    _cycle_bear_spike_raw,
    _cycle_bull_channel_raw,
    _cycle_bear_channel_raw,
    _cycle_trading_range_raw,
    _softmax,
    classify_cycle_phase,
)
from aiedge.context.shape import (  # noqa: E402
    SESSION_SHAPE_CLASSIFIER_ENABLED,
    SESSION_SHAPE_SOFTMAX_TEMP,
    SESSION_SHAPE_WARMUP_MINUTES,
    SESSION_SHAPES,
    _shape_trend_from_open_raw,
    _shape_spike_and_channel_raw,
    _shape_trend_reversal_raw,
    _shape_trend_resumption_raw,
    _shape_opening_reversal_raw,
    classify_session_shape,
)
from aiedge.context.daytype import (  # noqa: E402
    CHOP_RATIO_THRESHOLD,
    DAY_TYPE_WEIGHTS,
    OR_TRADING_RANGE,
    OR_TRENDING_TR_HIGH,
    OR_TRENDING_TR_LOW,
    SPIKE_MIN_BARS,
    STRONG_BODY_RATIO,
    WARMUP_BARS,
    _apply_day_type_weight,
    _classify_day_type,
    _compute_two_sided_ratio,
)
from aiedge.data.normalize import _normalize_databento_df  # noqa: E402
from aiedge.data.resample import (  # noqa: E402
    SCAN_BAR_SCHEMA,
    SCAN_RESAMPLE,
    _resample_to_5min,
)
from aiedge.data.universe import _get_default_universe  # noqa: E402
from aiedge.risk.trader_eq import _compute_risk_reward  # noqa: E402
from aiedge.signals.aggregator import (  # noqa: E402
    FAILED_GAP_MIN_FRAC_ADR,
    SIGNAL_BUY_INTRADAY,
    SIGNAL_SELL_INTRADAY,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_LOW,
    UNCERTAINTY_MED,
    UNCERTAINTY_TRAP,
    URGENCY_HIGH,
    _detect_phase,
    _determine_signal,
)
from aiedge.signals.bpa import (  # noqa: E402
    BPA_COUNTER_TYPES,
    BPA_INTEGRATION_ENABLED,
    BPA_LONG_SETUP_TYPES,
    BPA_MIN_CONFIDENCE,
    BPA_MIN_DF_LEN,
    BPA_RECENCY_BARS,
    BPA_SHORT_SETUP_TYPES,
    _score_bpa_patterns,
)
from aiedge.signals.summary import _generate_summary  # noqa: E402
from aiedge.signals.components import (  # noqa: E402
    BARS_STUCK_THRESHOLD,
    BEAR_SPIKE_RATIO,
    BODY_GAP_BONUS,
    BODY_OVERLAP_HIGH,
    CLOSE_TOP_PCT,
    COLOR_ALT_HIGH,
    DEEP_PULLBACK_PCT,
    DOJI_RATIO_HIGH,
    FAILED_SETUP_BONUS,
    GAP_INTEGRITY_POST_FILL_EVAL,
    GAP_PARTIAL_FILL_PCT,
    GAP_RECOVERY_PARTIAL_BULL,
    GAP_RECOVERY_PARTIAL_PCT,
    GAP_RECOVERY_STRONG_BULL,
    GAP_RECOVERY_STRONG_PCT,
    LIQUIDITY_GAP_PCT,
    LIQUIDITY_GAPS_HIGH,
    LIQUIDITY_GAPS_LOW,
    LIQUIDITY_GAPS_MODERATE,
    LIQUIDITY_MIN_DOLLAR_VOL,
    LIQUIDITY_SKIP_BARS,
    MA_GAP_BARS_MODERATE,
    MA_GAP_BARS_STRONG,
    MAX_SPIKE_BARS_SCORED,
    MICRO_GAP_BONUS,
    MIDPOINT_TOLERANCE,
    MODERATE_PULLBACK_PCT,
    NO_PULLBACK_BAR_THRESHOLD,
    POST_FILL_EVAL_BARS,
    REVERSAL_HIGH,
    REVERSAL_MODERATE,
    SHALLOW_PULLBACK_PCT,
    SPIKE_DURATION_MODERATE,
    SPIKE_DURATION_STRONG,
    SPIKE_RETRACE_LIMIT,
    SPT_DEPTH_MODERATE,
    SPT_DEPTH_SHALLOW,
    SPT_LOOKBACK_BARS,
    SPT_TREND_BODY_RATIO,
    STRONG_TREND_WINDOW,
    TAIL_BAD_PCT,
    TAIL_CLEAN_PCT,
    TAIL_CLEAN_RATIO,
    TAIL_MAX_PCT,
    TIGHT_RANGE_BARS_MODERATE,
    TIGHT_RANGE_BARS_STRONG,
    TIGHT_RANGE_PCT,
    TRENDING_SWINGS_MODERATE,
    TRENDING_SWINGS_STRONG,
    TWO_SIDED_HIGH,
    TWO_SIDED_MODERATE,
    TWO_SIDED_VERY_HIGH,
    UNCERTAINTY_ANALYSIS_WINDOW,
    UNCERTAINTY_RAW_MAX,
    URGENCY_RAW_MAX,
    VOLUME_SPIKE_EXTREME,
    VOLUME_SPIKE_STRONG,
    _check_liquidity,
    _find_first_pullback,
    _score_body_gaps,
    _score_failed_counter_setups,
    _score_follow_through,
    _score_gap_integrity,
    _score_levels_broken,
    _score_liquidity_gaps,
    _score_ma_separation,
    _score_majority_trend_bars,
    _score_micro_gaps,
    _score_small_pullback_trend,
    _score_spike_duration,
    _score_spike_quality,
    _score_tail_quality,
    _score_trending_everything,
    _score_trending_swings,
    _score_two_sided_ratio,
    _score_uncertainty,
    _score_volume_confirmation,
)


# =============================================================================
# SCORING COMPONENTS — URGENCY + UNCERTAINTY (all moved to aiedge.signals.components)
# =============================================================================






























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



# =============================================================================
# Cycle-phase classifier functions now live in aiedge.context.phase.
# Re-imported near top of this file.
# =============================================================================



# =============================================================================
# Session-shape classifier functions now live in aiedge.context.shape.
# Re-imported near top of this file.
# =============================================================================



# =============================================================================
# SCORING COMPONENTS — UNCERTAINTY (12 components, ~19 raw pts → normalized 0-10)
# =============================================================================



# =============================================================================
# PHASE 1 — OPENING RANGE, DAY TYPE, NEW COMPONENTS, WEIGHT MATRIX
# =============================================================================

# _opening_range now lives in aiedge.features.session
# _compute_two_sided_ratio, _classify_day_type, _apply_day_type_weight, and
# DAY_TYPE_WEIGHTS now live in aiedge.context.daytype
# (all re-imported near top of this file).








# =============================================================================
# PHASE, RISK/REWARD, SIGNAL, SUMMARY
# =============================================================================









# =============================================================================
# ATR HELPER
# =============================================================================

# _compute_daily_atr now lives in aiedge.features.volatility
# (re-imported near top of this file).


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

    # BPA pattern overlay (runs detectors, scores alignment with gap direction).
    # Pass ADR so failed_bo can use a volatility-normalized stop buffer.
    bpa_alignment_score, bpa_active_setups = _score_bpa_patterns(
        df, gap_direction, gap_fill_status, adr=atr,
    )

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

    gap_open = float(df.iloc[0]["open"])
    abs_gap_frac_adr = abs(gap_open - prior_close) / atr if atr > 0 else 0.0
    phase = _detect_phase(df, spike_bars, uncertainty, gap_direction, gap_held,
                          abs_gap_frac_adr=abs_gap_frac_adr)

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

# SCAN_BAR_SCHEMA and SCAN_RESAMPLE now live in aiedge.data.resample
# (re-imported near the top of this file).







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
