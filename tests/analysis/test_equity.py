"""Tests for aiedge.analysis.equity."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.analysis.equity import (
    equity_curve, max_drawdown, sharpe, sortino, summary_stats,
)


class EquityCurveTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(len(equity_curve([])), 0)

    def test_cumsum(self):
        curve = equity_curve([1.0, -0.5, 0.7])
        self.assertEqual(list(curve), [1.0, 0.5, 1.2])


class SharpeTests(unittest.TestCase):

    def test_empty_zero(self):
        self.assertEqual(sharpe([]), 0.0)

    def test_short_zero(self):
        self.assertEqual(sharpe([1.0]), 0.0)

    def test_zero_stddev_zero(self):
        self.assertEqual(sharpe([1.0, 1.0, 1.0]), 0.0)

    def test_positive_sharpe(self):
        pnls = [0.01, 0.02, 0.015, 0.018, 0.022]
        self.assertGreater(sharpe(pnls), 0)

    def test_negative_mean_negative_sharpe(self):
        pnls = [-0.01, -0.02, -0.015, -0.018, -0.022]
        self.assertLess(sharpe(pnls), 0)


class SortinoTests(unittest.TestCase):

    def test_empty_zero(self):
        self.assertEqual(sortino([]), 0.0)

    def test_no_losses_zero(self):
        self.assertEqual(sortino([0.01, 0.02, 0.015]), 0.0)

    def test_positive_with_some_losses(self):
        pnls = [0.02, 0.03, -0.01, 0.025, -0.005, 0.02]
        # Avg positive, downside dev > 0 → positive sortino
        self.assertGreater(sortino(pnls), 0)


class MaxDrawdownTests(unittest.TestCase):

    def test_empty_zero(self):
        self.assertEqual(max_drawdown([]), 0.0)

    def test_monotonic_up_zero_dd(self):
        equity = [100, 101, 102, 103]
        self.assertEqual(max_drawdown(equity), 0.0)

    def test_peak_to_trough(self):
        equity = [100, 110, 90, 95]
        # Peak 110 → trough 90 = 20/110 ≈ 0.1818
        self.assertAlmostEqual(max_drawdown(equity), 20 / 110, places=4)


class SummaryStatsTests(unittest.TestCase):

    def test_empty_returns_zeros(self):
        stats = summary_stats([])
        self.assertEqual(stats["n_trades"], 0)
        self.assertEqual(stats["total_pnl"], 0.0)

    def test_returns_expected_keys(self):
        stats = summary_stats([1.0, -0.5, 0.7, -0.2, 1.5])
        for key in ("total_pnl", "win_rate", "avg_win", "avg_loss",
                    "expectancy", "sharpe", "sortino", "max_drawdown", "n_trades"):
            self.assertIn(key, stats)

    def test_win_rate(self):
        # 3 wins, 2 losses
        stats = summary_stats([1.0, -0.5, 0.7, -0.2, 1.5])
        self.assertEqual(stats["win_rate"], 0.6)
        self.assertEqual(stats["n_trades"], 5)

    def test_total_pnl(self):
        stats = summary_stats([1.0, 2.0, -0.5])
        self.assertAlmostEqual(stats["total_pnl"], 2.5, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
