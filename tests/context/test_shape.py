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


# ── Scenario-based tests: feed each raw scorer a df that should fire it ──

def _strong_bull_open(n: int = 20) -> pd.DataFrame:
    """Open at low, big trend-from-open spike, minimal retrace, sustained."""
    bars_list = []
    price = 100.0
    for i in range(6):
        bars_list.append((price, price + 2.0, price - 0.05, price + 1.8))
        price += 1.8
    # First hour keeps pushing up
    for i in range(6, 12):
        bars_list.append((price, price + 1.0, price - 0.05, price + 0.7))
        price += 0.7
    # Close still well above open
    for i in range(12, n):
        bars_list.append((price, price + 0.5, price - 0.05, price + 0.3))
        price += 0.3
    return bars(bars_list)


def _bull_reversal(n: int = 20) -> pd.DataFrame:
    """Open pushes up, peaks mid-session, then reverses with lower highs."""
    bars_list = []
    price = 100.0
    for i in range(6):  # up leg
        bars_list.append((price, price + 1.5, price - 0.05, price + 1.0))
        price += 1.0
    for i in range(6, 12):  # peak + reversal
        bars_list.append((price, price + 0.2, price - 1.2, price - 1.0))
        price -= 1.0
    for i in range(12, n):  # continued down (lower highs)
        bars_list.append((price, price + 0.3, price - 0.8, price - 0.5))
        price -= 0.5
    return bars(bars_list)


def _opening_reversal(n: int = 20) -> pd.DataFrame:
    """Big thrust up in first 30 min, then reverses past open."""
    bars_list = []
    start = 100.0
    price = start
    for i in range(4):  # strong thrust
        bars_list.append((price, price + 2.5, price - 0.05, price + 2.0))
        price += 2.0
    # Reversal → drops all the way back and beyond
    for i in range(4, 12):
        bars_list.append((price, price + 0.2, price - 2.0, price - 1.5))
        price -= 1.5
    for i in range(12, n):  # continues down
        bars_list.append((price, price + 0.2, price - 0.5, price - 0.3))
        price -= 0.3
    return bars(bars_list)


def _spike_then_channel(n: int = 20) -> pd.DataFrame:
    """Strong opening spike, then shallower channel in same direction."""
    bars_list = []
    price = 100.0
    for i in range(3):  # fast spike
        bars_list.append((price, price + 3.0, price - 0.05, price + 2.5))
        price += 2.5
    for i in range(3, n):  # slow channel up
        bars_list.append((price, price + 0.5, price - 0.3, price + 0.25))
        price += 0.25
    return bars(bars_list)


def _trend_resume(n: int = 20) -> pd.DataFrame:
    """Open leg moves up, mid compresses, late continues."""
    bars_list = []
    price = 100.0
    # Open leg — 6 bars up
    for i in range(6):
        bars_list.append((price, price + 1.5, price - 0.05, price + 1.0))
        price += 1.0
    # Middle compression — 8 bars flat
    flat = price
    for i in range(6, 14):
        bars_list.append((flat - 0.2, flat + 0.3, flat - 0.3, flat + 0.1))
    # Late continuation — 6 bars up
    price = flat
    for i in range(14, n):
        bars_list.append((price, price + 1.2, price - 0.05, price + 0.8))
        price += 0.8
    return bars(bars_list)


class ScenarioScorerTests(unittest.TestCase):
    """Each of the 5 raw scorers should produce a meaningful score
    (>0.2 feels reasonable) on the scenario df crafted for it."""

    def test_trend_from_open_scorer_fires(self):
        df = _strong_bull_open(20)
        self.assertGreater(_shape_trend_from_open_raw(df, "up"), 0.3)

    def test_spike_and_channel_scorer_fires(self):
        df = _spike_then_channel(20)
        self.assertGreater(_shape_spike_and_channel_raw(df, "up", 3), 0.3)

    def test_trend_reversal_scorer_fires(self):
        df = _bull_reversal(20)
        self.assertGreater(_shape_trend_reversal_raw(df, "up"), 0.3)

    def test_trend_resumption_scorer_fires(self):
        df = _trend_resume(20)
        self.assertGreater(_shape_trend_resumption_raw(df, "up"), 0.3)

    def test_opening_reversal_scorer_fires(self):
        df = _opening_reversal(20)
        self.assertGreater(_shape_opening_reversal_raw(df, "up"), 0.3)

    def test_bear_scenarios_mirror_bull(self):
        """Pass `down` direction on a bull scenario — scorers should not fire."""
        df = _strong_bull_open(20)
        self.assertEqual(_shape_trend_from_open_raw(df, "down"), 0.0)


class ClassifyFullScenarioTests(unittest.TestCase):
    """End-to-end: classify_session_shape should pick the right top label
    on our crafted fixtures."""

    def test_classifier_picks_trend_from_open(self):
        df = _strong_bull_open(25)
        out = classify_session_shape(df, "up", session_minutes=90)
        # Shouldn't pick "undetermined" on a clear trend-from-open
        self.assertNotEqual(out["top"], "undetermined")

    def test_raw_scores_bounded_to_unit_interval(self):
        df = _spike_then_channel(20)
        out = classify_session_shape(df, "up", spike_bars=3, session_minutes=90)
        for v in out["raw"].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
