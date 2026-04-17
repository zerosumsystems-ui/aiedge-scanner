"""Unit tests for aiedge.features.swings."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.swings import _find_swing_highs, _find_swing_lows


def bars(rows):
    """Build a DataFrame from a list of (open, high, low, close) tuples."""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


class SwingLowTests(unittest.TestCase):

    def test_simple_valley(self):
        # lows: 10, 8, 10 → bar 1 is a swing low
        df = bars([(0, 11, 10, 10), (0, 9, 8, 9), (0, 11, 10, 10)])
        swings = _find_swing_lows(df)
        self.assertEqual(swings, [(1, 8)])

    def test_no_swing_on_monotonic_down(self):
        df = bars([(0, 12, 11, 11), (0, 11, 10, 10), (0, 10, 9, 9)])
        self.assertEqual(_find_swing_lows(df), [])

    def test_no_swing_on_equal_lows(self):
        # Strict inequality: 10, 10, 10 → no swing
        df = bars([(0, 11, 10, 10), (0, 11, 10, 10), (0, 11, 10, 10)])
        self.assertEqual(_find_swing_lows(df), [])

    def test_endpoints_never_swing(self):
        # First and last bars can't be swings by construction
        df = bars([(0, 11, 8, 9), (0, 11, 10, 10), (0, 11, 8, 9)])
        swings = _find_swing_lows(df)
        self.assertTrue(all(1 <= i <= len(df) - 2 for i, _ in swings))

    def test_multiple_swings(self):
        df = bars([
            (0, 11, 10, 10),
            (0, 9, 8, 9),    # swing low at 8
            (0, 11, 10, 10),
            (0, 9, 7, 8),    # swing low at 7
            (0, 11, 10, 10),
        ])
        swings = _find_swing_lows(df)
        self.assertEqual(swings, [(1, 8), (3, 7)])


class SwingHighTests(unittest.TestCase):

    def test_simple_peak(self):
        # highs: 10, 12, 10 → bar 1 is swing high
        df = bars([(0, 10, 9, 10), (0, 12, 11, 12), (0, 10, 9, 10)])
        swings = _find_swing_highs(df)
        self.assertEqual(swings, [(1, 12)])

    def test_no_swing_on_monotonic_up(self):
        df = bars([(0, 10, 9, 9), (0, 11, 10, 10), (0, 12, 11, 11)])
        self.assertEqual(_find_swing_highs(df), [])

    def test_multiple_peaks(self):
        df = bars([
            (0, 10, 9, 10),
            (0, 12, 11, 12),  # peak 12
            (0, 10, 9, 10),
            (0, 13, 12, 13),  # peak 13
            (0, 10, 9, 10),
        ])
        swings = _find_swing_highs(df)
        self.assertEqual(swings, [(1, 12), (3, 13)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
