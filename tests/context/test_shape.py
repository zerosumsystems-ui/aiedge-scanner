"""Smoke tests for aiedge.context.shape.

Session-shape scoring is complex and inherently probabilistic — these
tests check shape invariants (range, keys, disabled state) rather than
exact values.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.context import shape as s
from aiedge.context.shape import (
    SESSION_SHAPES,
    _shape_opening_reversal_raw,
    _shape_spike_and_channel_raw,
    _shape_trend_from_open_raw,
    _shape_trend_resumption_raw,
    _shape_trend_reversal_raw,
    classify_session_shape,
)


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def flat_session(n: int = 30, price: float = 100.0) -> pd.DataFrame:
    return bars([(price, price + 0.5, price - 0.5, price) for _ in range(n)])


class RawScorerRangeTests(unittest.TestCase):

    def test_all_scorers_return_in_range(self):
        df = flat_session(30)
        checks = [
            (_shape_trend_from_open_raw, (df, "up")),
            (_shape_spike_and_channel_raw, (df, "up", 3)),
            (_shape_trend_reversal_raw, (df, "up")),
            (_shape_trend_resumption_raw, (df, "up")),
            (_shape_opening_reversal_raw, (df, "up")),
        ]
        for fn, args in checks:
            v = fn(*args)
            self.assertGreaterEqual(v, 0.0, fn.__name__)
            self.assertLessEqual(v, 1.0, fn.__name__)

    def test_short_input_returns_zero(self):
        df = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
        self.assertEqual(_shape_trend_from_open_raw(df, "up"), 0.0)
        self.assertEqual(_shape_spike_and_channel_raw(df, "up", 2), 0.0)
        self.assertEqual(_shape_trend_reversal_raw(df, "up"), 0.0)
        self.assertEqual(_shape_trend_resumption_raw(df, "up"), 0.0)
        self.assertEqual(_shape_opening_reversal_raw(df, "up"), 0.0)


class ClassifySessionShapeTests(unittest.TestCase):

    def test_disabled_returns_empty(self):
        s.SESSION_SHAPE_CLASSIFIER_ENABLED = False
        try:
            self.assertEqual(classify_session_shape(flat_session(), "up"), {})
        finally:
            s.SESSION_SHAPE_CLASSIFIER_ENABLED = True

    def test_short_df_returns_undetermined(self):
        df = bars([(100, 101, 99, 100), (100, 101, 99, 100)])
        result = classify_session_shape(df, "up")
        self.assertEqual(result["top"], "undetermined")
        self.assertFalse(result["show_on_live_card"])
        self.assertEqual(set(result["probs"].keys()), set(SESSION_SHAPES))

    def test_warmup_gate_hides_card_early(self):
        df = flat_session(20)
        result = classify_session_shape(df, "up", session_minutes=30)
        self.assertFalse(result["show_on_live_card"])

    def test_warmup_gate_shows_card_late(self):
        df = flat_session(20)
        result = classify_session_shape(df, "up", session_minutes=90)
        self.assertTrue(result["show_on_live_card"])

    def test_probs_sum_to_one(self):
        df = flat_session(20)
        result = classify_session_shape(df, "up", session_minutes=90)
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0, places=2)

    def test_returns_expected_keys(self):
        df = flat_session(20)
        result = classify_session_shape(df, "up", session_minutes=90)
        for key in ("probs", "top", "confidence", "raw", "show_on_live_card"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
