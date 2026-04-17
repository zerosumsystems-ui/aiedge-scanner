"""Unit tests for aiedge.features.ema."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.ema import EMA_PERIOD, _compute_ema


class EmaTests(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        out = _compute_ema(np.array([]))
        self.assertEqual(len(out), 0)

    def test_output_length_matches_input(self):
        closes = np.arange(1, 21, dtype=float)
        out = _compute_ema(closes, period=10)
        self.assertEqual(len(out), len(closes))

    def test_first_value_seeds_from_first_close(self):
        closes = np.array([100.0, 101.0, 102.0])
        out = _compute_ema(closes, period=5)
        self.assertAlmostEqual(out[0], 100.0)

    def test_flat_prices_produce_flat_ema(self):
        closes = np.full(20, 100.0)
        out = _compute_ema(closes, period=10)
        # After seeding with 100.0, EMA of flat 100s stays at 100.0.
        np.testing.assert_allclose(out, 100.0, atol=1e-10)

    def test_monotonic_up_produces_monotonic_up_ema(self):
        closes = np.arange(1, 51, dtype=float)
        out = _compute_ema(closes)
        diffs = np.diff(out)
        self.assertTrue((diffs >= 0).all())

    def test_default_period_is_20(self):
        self.assertEqual(EMA_PERIOD, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
