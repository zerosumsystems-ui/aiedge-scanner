"""Unit tests for aiedge.features.candles.

Pure functions → deterministic tests → golden-value checks.
"""

import math
import sys
import unittest
from pathlib import Path

# Make repo root importable when running `python tests/features/test_candles.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.features.candles import (
    MIN_RANGE,
    _body,
    _body_ratio,
    _close_position,
    _is_bear,
    _is_bull,
    _lower_tail_pct,
    _safe_range,
    _upper_tail_pct,
)


def bar(open_: float, high: float, low: float, close: float) -> dict:
    """Helper for building a bar-shaped dict."""
    return {"open": open_, "high": high, "low": low, "close": close}


class SafeRangeTests(unittest.TestCase):

    def test_normal_bar(self):
        self.assertAlmostEqual(_safe_range(bar(100, 102, 99, 101)), 3.0)

    def test_zero_range_uses_floor(self):
        # When high == low (e.g. a single-trade bar), range is floored.
        self.assertEqual(_safe_range(bar(100, 100, 100, 100)), MIN_RANGE)

    def test_sub_floor_range_uses_floor(self):
        # Range smaller than MIN_RANGE is floored.
        self.assertEqual(_safe_range(bar(100, 100.0001, 100, 100)), MIN_RANGE)


class BodyTests(unittest.TestCase):

    def test_bull_body(self):
        self.assertAlmostEqual(_body(bar(100, 102, 99, 101)), 1.0)

    def test_bear_body(self):
        self.assertAlmostEqual(_body(bar(101, 102, 99, 100)), 1.0)

    def test_doji_body_is_zero(self):
        self.assertEqual(_body(bar(100, 101, 99, 100)), 0.0)


class BodyRatioTests(unittest.TestCase):

    def test_full_trend_bar_is_one(self):
        # open at low, close at high → body == range
        self.assertAlmostEqual(_body_ratio(bar(99, 101, 99, 101)), 1.0)

    def test_doji_is_zero(self):
        self.assertAlmostEqual(_body_ratio(bar(100, 101, 99, 100)), 0.0)

    def test_half_body(self):
        # open=100, close=102, range=4, body=2 → 0.5
        self.assertAlmostEqual(_body_ratio(bar(100, 102, 98, 102)), 0.5)
        # open=99, close=101, range=4, body=2 → 0.5
        self.assertAlmostEqual(_body_ratio(bar(99, 102, 98, 101)), 0.5)


class BullBearTests(unittest.TestCase):

    def test_bull(self):
        self.assertTrue(_is_bull(bar(100, 102, 99, 101)))
        self.assertFalse(_is_bear(bar(100, 102, 99, 101)))

    def test_bear(self):
        self.assertTrue(_is_bear(bar(101, 102, 99, 100)))
        self.assertFalse(_is_bull(bar(101, 102, 99, 100)))

    def test_doji_is_neither(self):
        b = bar(100, 101, 99, 100)
        self.assertFalse(_is_bull(b))
        self.assertFalse(_is_bear(b))


class TailTests(unittest.TestCase):

    def test_lower_tail_bull(self):
        # bull bar: body_bottom = open = 100. low = 99. range = 3.
        # lower_tail = (100 - 99) / 3 = 1/3
        self.assertAlmostEqual(
            _lower_tail_pct(bar(100, 102, 99, 101)), 1.0 / 3.0
        )

    def test_upper_tail_bull(self):
        # bull bar: body_top = close = 101. high = 102. range = 3.
        # upper_tail = (102 - 101) / 3 = 1/3
        self.assertAlmostEqual(
            _upper_tail_pct(bar(100, 102, 99, 101)), 1.0 / 3.0
        )

    def test_lower_tail_bear(self):
        # bear bar: body_bottom = close = 100. low = 99. range = 3.
        # lower_tail = (100 - 99) / 3 = 1/3
        self.assertAlmostEqual(
            _lower_tail_pct(bar(101, 102, 99, 100)), 1.0 / 3.0
        )

    def test_full_trend_bar_no_tails(self):
        # open=low, close=high → no tails
        self.assertAlmostEqual(_lower_tail_pct(bar(99, 101, 99, 101)), 0.0)
        self.assertAlmostEqual(_upper_tail_pct(bar(99, 101, 99, 101)), 0.0)


class ClosePositionTests(unittest.TestCase):

    def test_close_at_high(self):
        self.assertAlmostEqual(_close_position(bar(100, 102, 99, 102)), 1.0)

    def test_close_at_low(self):
        self.assertAlmostEqual(_close_position(bar(100, 102, 99, 99)), 0.0)

    def test_close_at_midpoint(self):
        # range=4, close-low=2 → 0.5
        self.assertAlmostEqual(_close_position(bar(100, 102, 98, 100)), 0.5)


class NaNGuardTests(unittest.TestCase):
    """Safety: none of these functions should return NaN on valid input."""

    def test_no_nan_on_normal_bar(self):
        b = bar(100, 102, 99, 101)
        for fn in (_safe_range, _body, _body_ratio,
                   _lower_tail_pct, _upper_tail_pct, _close_position):
            self.assertFalse(math.isnan(fn(b)), fn.__name__)

    def test_no_nan_on_doji(self):
        b = bar(100, 101, 99, 100)
        for fn in (_safe_range, _body, _body_ratio,
                   _lower_tail_pct, _upper_tail_pct, _close_position):
            self.assertFalse(math.isnan(fn(b)), fn.__name__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
