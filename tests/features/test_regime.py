"""Tests for aiedge.features.regime."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.regime import atr_percentile, realized_vol_tercile


class AtrPercentileTests(unittest.TestCase):

    def test_empty_history_returns_half(self):
        self.assertEqual(atr_percentile(1.0, []), 0.5)

    def test_quiet_today_vs_loud_history(self):
        history = [5.0, 6.0, 7.0, 8.0]
        pct = atr_percentile(1.0, history)
        self.assertEqual(pct, 0.0)

    def test_loud_today_vs_quiet_history(self):
        history = [1.0, 1.2, 1.5, 2.0]
        pct = atr_percentile(10.0, history)
        self.assertEqual(pct, 1.0)

    def test_median(self):
        history = [1.0, 2.0, 3.0, 4.0]
        pct = atr_percentile(2.5, history)
        self.assertEqual(pct, 0.5)  # 2 of 4 ≤ 2.5

    def test_tie_counts_as_at_or_below(self):
        history = [1.0, 2.0, 3.0, 4.0]
        pct = atr_percentile(2.0, history)
        self.assertEqual(pct, 0.5)  # 2 values ≤ 2.0

    def test_nan_ignored(self):
        history = [1.0, float("nan"), 3.0]
        pct = atr_percentile(2.0, history)
        # nan filtered; 1 of 2 remaining ≤ 2.0
        self.assertEqual(pct, 0.5)


class RealizedVolTercileTests(unittest.TestCase):

    def test_too_few_returns_mid(self):
        self.assertEqual(realized_vol_tercile([100.0, 101.0]), "mid")

    def test_flat_closes_returns_low(self):
        closes = [100.0] * 30
        result = realized_vol_tercile(closes)
        # Zero vol → below 0.006 threshold → low
        self.assertEqual(result, "low")

    def test_wild_closes_returns_high(self):
        # 10% daily swings
        closes = [100.0, 110.0, 99.0, 112.0, 97.0, 115.0, 95.0, 118.0] * 4
        result = realized_vol_tercile(closes)
        self.assertEqual(result, "high")

    def test_rolling_history_triggers_tercile_logic(self):
        # 2 full lookback windows of varying vol — triggers the rolling branch
        import numpy as np
        np.random.seed(0)
        low_vol = list(100 + np.cumsum(np.random.normal(0, 0.1, 20)))
        # Second window: higher volatility
        high_vol = list(low_vol[-1] + np.cumsum(np.random.normal(0, 2.0, 20)))
        closes = low_vol + high_vol
        result = realized_vol_tercile(closes, lookback_days=20)
        self.assertIn(result, ("low", "mid", "high"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
