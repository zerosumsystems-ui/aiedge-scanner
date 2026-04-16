#!/usr/bin/env python3
"""Unit test for SPT (Small Pullback Trend) fix."""

import pandas as pd
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared.brooks_score import _score_small_pullback_trend

def test_spt_current_vs_peak():
    """
    Test that _score_small_pullback_trend tracks current SPT correctly
    in a synthetic series where SPT peaks at 3.0 then drops to 0.5.
    """
    # Synthetic 5-min bars: SPT rises, peaks at 3.0, then drops to 0.5
    # We want to verify that current_spt reflects the LATEST 15-bar window,
    # not the peak from earlier in the session.

    bars_data = [
        # Early bars: start building trend (SPT=0.0 here, not in 15-bar window yet)
        {"open": 100.0, "high": 100.5, "low": 99.9, "close": 100.3},  # bar 0: trend up
        {"open": 100.3, "high": 100.8, "low": 100.2, "close": 100.6},  # bar 1: trend up
        {"open": 100.6, "high": 101.0, "low": 100.5, "close": 100.9},  # bar 2: trend up
    ]

    # Build out 15 more "perfect SPT" bars (clean pullbacks, high trend bar %)
    # Each bar has body ratio > 0.4 and shallow pullbacks
    for i in range(15):
        last_close = bars_data[-1]["close"]
        open_price = last_close
        close_price = last_close + 0.15  # small up move, good body ratio
        high_price = close_price + 0.05  # small upper tail
        low_price = open_price - 0.01    # minimal lower tail (clean)

        bars_data.append({
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        })

    # At this point, we have 18 bars. The last 15 (indices 3-17) should score high SPT.
    df_high_spt = pd.DataFrame(bars_data)
    spt_high = _score_small_pullback_trend(df_high_spt, "up")
    print(f"SPT at high point (18 bars, last 15 are clean trend): {spt_high:.2f}")
    assert 2.0 <= spt_high <= 3.0, f"Expected SPT ~3.0, got {spt_high}"
    # Record peak
    peak_spt = spt_high

    # Now add 15 more "bad" bars (wide pullbacks, fail the SPT check)
    # This pushes out the good bars from the 15-bar window
    for i in range(15):
        last_close = bars_data[-1]["close"]
        # Create bad bars: deep pullback or poor body ratio
        open_price = last_close
        if i % 2 == 0:
            # Pullback day
            close_price = last_close - 0.40  # deep pullback
            low_price = last_close - 0.50    # very deep
            high_price = last_close + 0.05
        else:
            # Wide-range bar with poor body ratio
            close_price = last_close + 0.10
            high_price = close_price + 0.30  # wide tail
            low_price = close_price - 0.30   # wide tail both sides

        bars_data.append({
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        })

    df_low_spt = pd.DataFrame(bars_data)
    spt_low = _score_small_pullback_trend(df_low_spt, "up")
    print(f"SPT after bad bars (33 bars, last 15 are bad): {spt_low:.2f}")
    # Should be much lower since the last 15-bar window has poor SPT characteristics
    assert spt_low < 1.0, f"Expected SPT < 1.0 after bad bars, got {spt_low}"

    print("\n✓ Test passed: SPT correctly reflects CURRENT 15-bar window, not peak")
    print(f"  Peak SPT: {peak_spt:.2f}")
    print(f"  Current SPT: {spt_low:.2f}")
    print(f"  current != peak (as expected): {spt_low != peak_spt}")

if __name__ == "__main__":
    test_spt_current_vs_peak()
    print("\n✓ All tests passed")
