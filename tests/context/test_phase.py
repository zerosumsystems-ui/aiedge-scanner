"""Unit tests for aiedge.context.phase."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.context import phase as p
from aiedge.context.phase import (
    CYCLE_PHASES,
    _cycle_bear_channel_raw,
    _cycle_bear_spike_raw,
    _cycle_bull_channel_raw,
    _cycle_bull_spike_raw,
    _cycle_trading_range_raw,
    _softmax,
    classify_cycle_phase,
)


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


class SoftmaxTests(unittest.TestCase):

    def test_output_sums_to_one(self):
        out = _softmax([1.0, 2.0, 3.0], temperature=0.5)
        self.assertAlmostEqual(sum(out), 1.0)

    def test_larger_input_gets_larger_prob(self):
        out = _softmax([0.0, 1.0, 0.0], temperature=0.5)
        self.assertGreater(out[1], out[0])
        self.assertGreater(out[1], out[2])

    def test_tiny_temperature_is_stable(self):
        # Would overflow without subtracting max.
        out = _softmax([1000.0, 1001.0], temperature=0.001)
        self.assertAlmostEqual(sum(out), 1.0)


class RawScorerRangeTests(unittest.TestCase):
    """Every raw scorer must return a value in [0, 1]."""

    def _random_session(self, n: int = 20) -> pd.DataFrame:
        import random
        rng = random.Random(42)
        rows = []
        price = 100.0
        for _ in range(n):
            o = price
            h = o + rng.uniform(0, 2)
            l = o - rng.uniform(0, 2)
            c = rng.uniform(l, h)
            price = c
            rows.append((o, h, l, c))
        return bars(rows)

    def test_all_scorers_in_range(self):
        df = self._random_session()
        for fn in (
            _cycle_bull_spike_raw,
            _cycle_bear_spike_raw,
            _cycle_bull_channel_raw,
            _cycle_bear_channel_raw,
            _cycle_trading_range_raw,
        ):
            val = fn(df)
            self.assertGreaterEqual(val, 0.0, fn.__name__)
            self.assertLessEqual(val, 1.0, fn.__name__)

    def test_empty_input_returns_zero(self):
        df = bars([])
        for fn in (
            _cycle_bull_spike_raw,
            _cycle_bear_spike_raw,
            _cycle_bull_channel_raw,
            _cycle_bear_channel_raw,
            _cycle_trading_range_raw,
        ):
            self.assertEqual(fn(df), 0.0, fn.__name__)


class ClassifyCyclePhaseTests(unittest.TestCase):

    def test_disabled_returns_empty(self):
        p.CYCLE_PHASE_CLASSIFIER_ENABLED = False
        try:
            self.assertEqual(classify_cycle_phase(bars([])), {})
        finally:
            p.CYCLE_PHASE_CLASSIFIER_ENABLED = True

    def test_short_df_returns_uniform(self):
        # <3 bars → uniform 0.2 across 5 phases.
        df = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
        result = classify_cycle_phase(df)
        self.assertEqual(result["top"], "trading_range")
        self.assertAlmostEqual(result["confidence"], 0.2)
        self.assertEqual(set(result["probs"].keys()), set(CYCLE_PHASES))

    def test_bull_spike_wins_on_obvious_bull_spike(self):
        # 5 consecutive fat bull bars, closing at highs.
        rows = [
            (100, 102, 99.9, 102),
            (102, 104, 101.9, 104),
            (104, 106, 103.9, 106),
            (106, 108, 105.9, 108),
            (108, 110, 107.9, 110),
        ]
        result = classify_cycle_phase(bars(rows))
        # Top should be bull_spike or at least bull-flavored.
        self.assertIn(result["top"], {"bull_spike", "bull_channel"})

    def test_returns_expected_keys(self):
        rows = [(100, 101, 99, 100)] * 20
        result = classify_cycle_phase(bars(rows))
        self.assertIn("probs", result)
        self.assertIn("top", result)
        self.assertIn("confidence", result)
        self.assertIn("raw", result)

    def test_probs_sum_to_one(self):
        rows = [(100, 101, 99, 100)] * 20
        result = classify_cycle_phase(bars(rows))
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
