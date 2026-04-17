"""Signal pipeline — top-level `score_gap` entry point.

Extracted from shared/brooks_score.py (ROADMAP C1). `score_gap` is the
single-symbol entry point that orchestrates every component under
`aiedge/`: features (candles, EMA, swings, volatility, session), context
(day-type, cycle-phase, session-shape), signals (18 components +
aggregator + BPA overlay + summary), and risk (trader's equation).

The private helpers it calls (`_score_*`, `_find_first_pullback`,
`_detect_phase`, etc.) still live in `shared/brooks_score.py` and
`aiedge/signals/components.py`; this module is a thin orchestrator.

TODO C2: migrate remaining private helpers out of `shared/brooks_score.py`
into `aiedge/signals/components.py` so this module no longer needs to
round-trip back through the compat shim.
"""

import pandas as pd

from aiedge.context.daytype import (
    _apply_day_type_weight,
    _classify_day_type,
    _compute_two_sided_ratio,
)
from aiedge.context.phase import classify_cycle_phase
from aiedge.context.shape import classify_session_shape
from aiedge.features.session import _opening_range
from aiedge.risk.trader_eq import _compute_risk_reward
from aiedge.signals.aggregator import _detect_phase, _determine_signal
from aiedge.signals.bpa import BPA_INTEGRATION_ENABLED, _score_bpa_patterns
from aiedge.signals.components import (
    URGENCY_RAW_MAX,
    UNCERTAINTY_RAW_MAX,
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
from aiedge.signals.summary import _generate_summary

# Magnitude filter thresholds used by score_gap's urgency caps.
# MAGNITUDE_FLOOR is used by scan_universe and lives in aiedge.runners.batch.
MAGNITUDE_CAP_9 = 0.7    # Min move/ATR for urgency > 9.0
MAGNITUDE_CAP_10 = 1.0   # Min move/ATR for urgency > 9.5 (stacks with EMA rule)


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
