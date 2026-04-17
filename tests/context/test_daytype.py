"""Unit tests for aiedge.context.daytype.

The classifier is structural and hits many branches. Tests check
behavioral invariants (warmup, trend-from-open at the extreme,
trading-range detection, weight-matrix shape) rather than exact
confidence values.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.context.daytype import (
    DAY_TYPE_WEIGHTS,
    SPIKE_MIN_BARS,
    STRONG_BODY_RATIO,
    WARMUP_BARS,
    _apply_day_type_weight,
    _classify_day_type,
    _compute_two_sided_ratio,
)


def mk_df(opens, closes, highs=None, lows=None, vol=100_000):
    n = len(opens)
    if highs is None:
        highs = [max(o, c) + 0.05 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": [vol] * n,
    })


def trend_up_df(n: int = 14, step: float = 1.0) -> pd.DataFrame:
    opens = [100 + step * i for i in range(n)]
    closes = [100 + step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


def trend_down_df(n: int = 14, step: float = 1.0) -> pd.DataFrame:
    opens = [100 - step * i for i in range(n)]
    closes = [100 - step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


def chop_df(n: int = 14) -> pd.DataFrame:
    opens, closes = [], []
    for i in range(n):
        if i % 2 == 0:
            opens.append(100.0); closes.append(100.5)
        else:
            opens.append(100.5); closes.append(100.0)
    return mk_df(opens, closes)


class ComputeTwoSidedRatioTests(unittest.TestCase):

    def test_tiny_df_returns_half(self):
        df = mk_df([100, 101], [101, 102])
        self.assertEqual(_compute_two_sided_ratio(df, "up"), 0.5)

    def test_pure_bull_is_zero_for_up(self):
        self.assertAlmostEqual(_compute_two_sided_ratio(trend_up_df(10), "up"), 0.0)

    def test_pure_bull_is_one_for_down(self):
        self.assertAlmostEqual(_compute_two_sided_ratio(trend_up_df(10), "down"), 1.0)

    def test_chop_is_roughly_half(self):
        ratio = _compute_two_sided_ratio(chop_df(10), "up")
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)


class ClassifyDayTypeTests(unittest.TestCase):

    def _or(self, pct: float = 0.2) -> dict:
        return {"range_pct": pct}

    def test_warmup_returns_undetermined(self):
        df = mk_df([100, 101], [101, 102])
        result = _classify_day_type(df, self._or(), spike_bars=1,
                                    two_sided_ratio=0.0,
                                    gap_direction="up", gap_held=False)
        self.assertEqual(result["day_type"], "undetermined")
        self.assertEqual(result["confidence"], 0.0)

    def test_warmup_with_strong_start_gives_preliminary_trend(self):
        df = mk_df([100, 101, 102], [101, 102, 103])
        result = _classify_day_type(df, self._or(), spike_bars=3,
                                    two_sided_ratio=0.0,
                                    gap_direction="up", gap_held=True)
        self.assertEqual(result["day_type"], "trend_from_open")
        self.assertEqual(result["confidence"], 0.4)

    def test_trend_from_open_bull(self):
        df = trend_up_df(14)
        result = _classify_day_type(df, self._or(), spike_bars=8,
                                    two_sided_ratio=0.0,
                                    gap_direction="up", gap_held=True)
        self.assertEqual(result["day_type"], "trend_from_open")
        self.assertGreater(result["confidence"], 0.5)

    def test_trend_from_open_bear(self):
        df = trend_down_df(14)
        result = _classify_day_type(df, self._or(), spike_bars=8,
                                    two_sided_ratio=0.0,
                                    gap_direction="down", gap_held=True)
        self.assertEqual(result["day_type"], "trend_from_open")

    def test_returns_expected_keys(self):
        df = trend_up_df(14)
        result = _classify_day_type(df, self._or(), spike_bars=8,
                                    two_sided_ratio=0.0,
                                    gap_direction="up", gap_held=True)
        for key in ("day_type", "confidence", "warning"):
            self.assertIn(key, result)

    def test_day_type_in_allowed_set(self):
        allowed = set(DAY_TYPE_WEIGHTS.keys())
        df = chop_df(14)
        result = _classify_day_type(df, self._or(0.55), spike_bars=0,
                                    two_sided_ratio=0.5,
                                    gap_direction="up", gap_held=False)
        self.assertIn(result["day_type"], allowed)


class DayTypeWeightMatrixTests(unittest.TestCase):

    def test_all_day_types_have_same_components(self):
        component_sets = [set(DAY_TYPE_WEIGHTS[dt].keys())
                          for dt in DAY_TYPE_WEIGHTS]
        first = component_sets[0]
        for cs in component_sets[1:]:
            self.assertEqual(cs, first)

    def test_undetermined_is_all_ones(self):
        for w in DAY_TYPE_WEIGHTS["undetermined"].values():
            self.assertEqual(w, 1.0)

    def test_trend_from_open_amplifies_spike_quality(self):
        self.assertGreater(
            DAY_TYPE_WEIGHTS["trend_from_open"]["spike_quality"], 1.0)

    def test_trading_range_dampens_spike_quality(self):
        self.assertLess(
            DAY_TYPE_WEIGHTS["trading_range"]["spike_quality"], 1.0)

    def test_trend_from_open_dampens_two_sided_ratio(self):
        self.assertLess(
            DAY_TYPE_WEIGHTS["trend_from_open"]["two_sided_ratio"], 1.0)


class ApplyDayTypeWeightTests(unittest.TestCase):

    def test_applies_matrix_weight(self):
        weight = DAY_TYPE_WEIGHTS["trend_from_open"]["spike_quality"]
        self.assertEqual(
            _apply_day_type_weight(2.0, "spike_quality", "trend_from_open"),
            2.0 * weight,
        )

    def test_unknown_component_passes_through(self):
        self.assertEqual(
            _apply_day_type_weight(2.0, "made_up_component", "trend_from_open"),
            2.0,
        )

    def test_unknown_day_type_falls_back_to_undetermined(self):
        self.assertEqual(
            _apply_day_type_weight(2.0, "spike_quality", "bogus_day_type"),
            2.0,
        )


class TunableSanityTests(unittest.TestCase):

    def test_warmup_bars_is_positive(self):
        self.assertGreater(WARMUP_BARS, 0)

    def test_spike_min_bars_is_positive(self):
        self.assertGreater(SPIKE_MIN_BARS, 0)

    def test_strong_body_ratio_in_zero_to_one(self):
        self.assertGreater(STRONG_BODY_RATIO, 0.0)
        self.assertLessEqual(STRONG_BODY_RATIO, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
