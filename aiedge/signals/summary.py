"""One-line Brooks-style summary for a scored signal.

Consumes the already-computed signal, urgency, uncertainty, phase, and
supporting metadata, and returns a short human-readable sentence for
dashboard / log / alert display.

Extracted from shared/brooks_score.py (Phase 3h).
"""

from typing import Optional

from aiedge.signals.aggregator import (
    SIGNAL_BUY_INTRADAY,
    SIGNAL_SELL_INTRADAY,
    UNCERTAINTY_TRAP,
    URGENCY_HIGH,
)


def _generate_summary(signal: str, urgency: float, uncertainty: float,
                      phase: str, always_in: str, gap_direction: str,
                      spike_bars: int, pullback_depth_pct: float,
                      gap_held: bool = True,
                      bpa_active_setups: Optional[list] = None) -> str:
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
