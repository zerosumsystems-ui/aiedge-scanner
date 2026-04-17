"""Tests for aiedge.analysis.reliability."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.analysis.reliability import (
    brier_score, expected_calibration_error, reliability_table,
)


class ReliabilityTableTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        out = reliability_table([], [])
        self.assertEqual(len(out), 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            reliability_table([0.5], [1, 0])

    def test_perfect_calibration_bins(self):
        # Predictions cluster in two bins; each should show ~100% match
        # between mean_predicted and empirical_hit_rate within bin.
        predicted = [0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9]
        outcomes = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]  # exact match
        table = reliability_table(predicted, outcomes, n_bins=10)
        # 0.1 group: 5 samples, empirical = 0.0 (all losses)
        # 0.9 group: 5 samples, empirical = 1.0 (all wins)
        # Both bins should hit 0 diff
        diffs = (table["mean_predicted"] - table["empirical_hit_rate"]).abs()
        for diff in diffs:
            self.assertLess(diff, 0.15)

    def test_columns_present(self):
        table = reliability_table([0.3, 0.7, 0.5], [1, 0, 1], n_bins=5)
        for col in ("bin_lo", "bin_hi", "n", "mean_predicted",
                    "empirical_hit_rate", "bin_midpoint"):
            self.assertIn(col, table.columns)


class BrierScoreTests(unittest.TestCase):

    def test_perfect_prediction_zero(self):
        # predicted=0 when outcome=0, predicted=1 when outcome=1
        self.assertEqual(brier_score([0, 0, 1, 1], [0, 0, 1, 1]), 0.0)

    def test_all_wrong_is_one(self):
        self.assertEqual(brier_score([1, 1, 0, 0], [0, 0, 1, 1]), 1.0)

    def test_coin_flip_is_quarter(self):
        self.assertAlmostEqual(brier_score([0.5] * 4, [0, 1, 0, 1]), 0.25, places=4)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            brier_score([0.5], [1, 0])


class ExpectedCalibrationErrorTests(unittest.TestCase):

    def test_well_calibrated_is_low_ece(self):
        # 0.1-predictions lose, 0.9-predictions win → very low ECE
        predicted = [0.1, 0.1, 0.9, 0.9] * 5
        outcomes = [0, 0, 1, 1] * 5
        ece = expected_calibration_error(predicted, outcomes)
        # Bin midpoint != mean predicted exactly, but diff stays small
        self.assertLess(ece, 0.15)

    def test_bounded_zero_one(self):
        import random
        random.seed(0)
        predicted = [random.random() for _ in range(50)]
        outcomes = [random.randint(0, 1) for _ in range(50)]
        ece = expected_calibration_error(predicted, outcomes)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
