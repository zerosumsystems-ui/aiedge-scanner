"""Unit tests for aiedge.features.session."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.session import OPENING_RANGE_BARS, _opening_range


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


class OpeningRangeTests(unittest.TestCase):

    def test_empty_df_safe_default(self):
        out = _opening_range(bars([]))
        self.assertEqual(out["range_high"], 0.0)
        self.assertEqual(out["range_low"], 0.0)
        self.assertEqual(out["range_size"], 0.0)
        self.assertEqual(out["range_pct"], 0.5)

    def test_basic_opening_range(self):
        # 6 bars, highs 100-105, lows 99-104 → or_high=105, or_low=99, size=6
        rows = [(0, 100 + i, 99 + i, 100 + i) for i in range(6)]
        out = _opening_range(bars(rows), avg_daily_range=10.0)
        self.assertAlmostEqual(out["range_high"], 105.0)
        self.assertAlmostEqual(out["range_low"], 99.0)
        self.assertAlmostEqual(out["range_size"], 6.0)
        self.assertAlmostEqual(out["range_pct"], 0.6)

    def test_range_pct_uses_provided_avg(self):
        rows = [(0, 101, 100, 100) for _ in range(6)]  # range = 1
        out = _opening_range(bars(rows), avg_daily_range=10.0)
        self.assertAlmostEqual(out["range_pct"], 0.1)

    def test_fewer_bars_than_requested(self):
        # Only 3 bars available, n_bars=6 → should use 3
        rows = [(0, 101, 100, 100), (0, 102, 100, 101), (0, 103, 100, 102)]
        out = _opening_range(bars(rows), avg_daily_range=10.0)
        self.assertAlmostEqual(out["range_high"], 103.0)
        self.assertAlmostEqual(out["range_low"], 100.0)

    def test_estimate_fallback_avoids_div_zero(self):
        # No avg_daily_range provided → estimator kicks in. Must not crash.
        rows = [(0, 101, 100, 100) for _ in range(6)]
        out = _opening_range(bars(rows))
        self.assertGreaterEqual(out["range_pct"], 0.0)

    def test_default_constant(self):
        self.assertEqual(OPENING_RANGE_BARS, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
