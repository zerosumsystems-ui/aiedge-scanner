"""Signal aggregator — combines urgency, uncertainty, R:R, BPA overlay,
and always-in state into the final signal label.

The aggregator is the final decision step. Inputs are already-normalized
urgency and uncertainty scores (0-10) plus the BPA alignment score and
risk/reward ratio. Output is one of:

    BUY_PULLBACK / BUY_SPIKE     — long setups
    SELL_PULLBACK / SELL_SPIKE   — short setups
    SELL_PULLBACK_INTRADAY       — bear-flip on a gap-up day
    BUY_PULLBACK_INTRADAY        — bull-flip on a gap-down day
    WAIT / FOG / PASS / AVOID    — non-actionable states

Also contains `_detect_phase`, the rule-based phase classifier used
by `score_gap` to tag the current market state (spike / channel /
trading_range / failed_gap). This is distinct from the probabilistic
cycle-phase classifier in `context/phase.py`.

Extracted from shared/brooks_score.py (Phase 3h).
"""

import numpy as np
import pandas as pd

from aiedge.context.daytype import SPIKE_MIN_BARS
from aiedge.signals.bpa import BPA_INTEGRATION_ENABLED


# ── Signal decision thresholds ─────────────────────────────────────
URGENCY_HIGH = 7
UNCERTAINTY_LOW = 3
UNCERTAINTY_MED = 5
UNCERTAINTY_HIGH = 5
UNCERTAINTY_TRAP = 7

# Minimum |gap| / ADR to classify a day as a "gap event" (Brooks:
# opening_gap_behavior.md §Special cases — "Small gap (<25% of ADR): not a
# 'gap event.' Treat the first bar like any other opening bar"). Below this
# threshold, the `failed_gap` phase is suppressed — intrabar noise on a sub-
# threshold gap would otherwise misclassify normal opening bars as failed gaps.
FAILED_GAP_MIN_FRAC_ADR = 0.25

# ── Intraday flip signal labels ──
SIGNAL_SELL_INTRADAY = "SELL_PULLBACK_INTRADAY"
SIGNAL_BUY_INTRADAY = "BUY_PULLBACK_INTRADAY"


def _detect_phase(df: pd.DataFrame, spike_bars: int, uncertainty: float,
                  gap_direction: str, gap_held: bool,
                  abs_gap_frac_adr: float = 0.0) -> str:
    """Determine the current market phase (rule-based).

    abs_gap_frac_adr: |gap_size| / ADR. Used to gate `failed_gap`: Brooks
    treats gaps < 25% of ADR as "not a gap event" (opening_gap_behavior.md
    §Special cases). On a sub-threshold gap the `gap_held` flag is noise —
    flipping between True/False on single-cent intrabar excursions — so we
    suppress the `failed_gap` phase and fall through to normal classification.
    """
    if spike_bars >= SPIKE_MIN_BARS and len(df) <= spike_bars + 2:
        return "spike"

    if (not gap_held
            and uncertainty > UNCERTAINTY_HIGH
            and abs_gap_frac_adr >= FAILED_GAP_MIN_FRAC_ADR):
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


def _determine_signal(urgency: float, uncertainty: float, gap_held: bool,
                      gap_direction: str, rr_ratio: float,
                      spike_bars: int, pullback_exists: bool,
                      bpa_alignment: float = 0.0,
                      always_in: str = "unclear",
                      gap_fill_status: str = "held") -> str:
    """Decision matrix for the final signal.

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
