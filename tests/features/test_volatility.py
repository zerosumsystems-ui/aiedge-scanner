"""Unit tests for aiedge.features.volatility."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.volatility import _compute_daily_atr


class DailyAtrTests(unittest.TestCase):

    def test_none_input_returns_zero(self):
        self.assertEqual(_compute_daily_atr(None), 0.0)

    def test_too_short_returns_zero(self):
        df = pd.DataFrame([{"high": 100, "low": 99}])
        self.assertEqual(_compute_daily_atr(df), 0.0)

    def test_simple_mean_of_ranges(self):
        # ranges: 2, 3, 4. mean = 3.
        df = pd.DataFrame([
            {"high": 102, "low": 100},
            {"high": 103, "low": 100},
            {"high": 104, "low": 100},
        ])
        self.assertAlmostEqual(_compute_daily_atr(df), 3.0)

    def test_period_caps_sample_count(self):
        # Build 30 bars with range=1 for first 10, range=5 for last 20.
        # tail(20) should take only the last-20 range=5 bars → ATR = 5.
        rows = [{"high": 100 + 1, "low": 100} for _ in range(10)]
        rows += [{"high": 100 + 5, "low": 100} for _ in range(20)]
        df = pd.DataFrame(rows)
        self.assertAlmostEqual(_compute_daily_atr(df, period=20), 5.0)

    def test_uses_high_minus_low_not_true_range(self):
        # If a gap exists, true-range would include it but ADR ignores it.
        # Bar 0: 100-99 = 1. Bar 1: 200-199 = 1 (huge gap up but range=1)
        df = pd.DataFrame([
            {"high": 100, "low": 99},
            {"high": 200, "low": 199},
        ])
        self.assertAlmostEqual(_compute_daily_atr(df), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
