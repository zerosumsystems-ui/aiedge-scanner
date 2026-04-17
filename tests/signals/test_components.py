"""Unit tests for aiedge.signals.components (Phase 3f-1 urgency scorers).

Each scorer returns a bounded raw score. Tests check:
  * the bounds are respected on contrived inputs
  * the sign of response matches direction
  * obvious edge cases (short df, wrong direction, flat data) return 0

These are correctness smoke tests, not calibration — fine-grained
thresholds are covered by the scanner-level regression in the
QC harness.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals.components import (
    LIQUIDITY_MIN_DOLLAR_VOL,
    SPT_LOOKBACK_BARS,
    UNCERTAINTY_RAW_MAX,
    URGENCY_RAW_MAX,
    _check_liquidity,
    _find_first_pullback,
    _score_body_gaps,
    _score_failed_counter_setups,
    _score_follow_through,
    _score_gap_integrity,
    _score_levels_broken,
    _score_liquidity_gaps,
    _score_ma_separation,
    _score_majority_trend_bars,
    _score_micro_gaps,
    _score_small_pullback_trend,
    _score_spike_duration,
    _score_spike_quality,
    _score_tail_quality,
    _score_trending_everything,
    _score_trending_swings,
    _score_two_sided_ratio,
    _score_uncertainty,
    _score_volume_confirmation,
)


def mk_df(opens, closes, highs=None, lows=None, vol=100_000):
    n = len(opens)
    if highs is None:
        highs = [max(o, c) + 0.05 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    if isinstance(vol, int):
        vol_list = [vol] * n
    else:
        vol_list = list(vol)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": vol_list,
    })


def trend_up_df(n: int = 30, step: float = 1.0) -> pd.DataFrame:
    opens = [100 + step * i for i in range(n)]
    closes = [100 + step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


def trend_down_df(n: int = 30, step: float = 1.0) -> pd.DataFrame:
    opens = [200 - step * i for i in range(n)]
    closes = [200 - step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


def chop_df(n: int = 20) -> pd.DataFrame:
    opens, closes = [], []
    for i in range(n):
        if i % 2 == 0:
            opens.append(100.0); closes.append(100.5)
        else:
            opens.append(100.5); closes.append(100.0)
    return mk_df(opens, closes)


class SpikeQualityTests(unittest.TestCase):

    def test_pure_bull_spike_scores_high(self):
        df = trend_up_df(6)
        score, bars = _score_spike_quality(df, "up")
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(bars, 1)

    def test_bull_trend_scored_down_direction_is_zero(self):
        df = trend_up_df(6)
        score, bars = _score_spike_quality(df, "down")
        self.assertEqual(score, 0.0)
        self.assertEqual(bars, 0)

    def test_capped_at_four(self):
        df = trend_up_df(20)
        score, _ = _score_spike_quality(df, "up")
        self.assertLessEqual(score, 4.0)


class GapIntegrityTests(unittest.TestCase):

    def test_bull_gap_held_scores_plus_two(self):
        df = trend_up_df(10)
        score, status = _score_gap_integrity(df, prior_close=99.0, gap_direction="up")
        self.assertEqual(score, 2.0)
        self.assertEqual(status, "held")

    def test_invalid_bull_gap_zero(self):
        df = trend_up_df(10)
        # prior close above open = no gap up
        score, status = _score_gap_integrity(df, prior_close=110.0, gap_direction="up")
        self.assertEqual(score, 0.0)
        self.assertEqual(status, "held")

    def test_bear_gap_held(self):
        df = trend_down_df(10)
        score, status = _score_gap_integrity(df, prior_close=201.0, gap_direction="down")
        self.assertEqual(score, 2.0)
        self.assertEqual(status, "held")


class FindFirstPullbackTests(unittest.TestCase):

    def test_no_bars_post_spike_with_many_spike_bars(self):
        df = trend_up_df(6)
        depth, score, exists = _find_first_pullback(df, spike_bars=8, gap_direction="up")
        self.assertEqual(exists, False)
        self.assertEqual(score, 1.0)

    def test_post_spike_bars_present_returns_exists(self):
        df = trend_up_df(6)
        _depth, _score, exists = _find_first_pullback(df, spike_bars=2, gap_direction="up")
        self.assertTrue(exists)


class FollowThroughTests(unittest.TestCase):

    def test_returns_zero_for_short_df(self):
        df = trend_up_df(3)
        self.assertEqual(_score_follow_through(df, 1, "up"), 0.0)

    def test_bounded_between_neg15_and_pos2(self):
        for df_fn in (trend_up_df, trend_down_df, chop_df):
            for dirn in ("up", "down"):
                val = _score_follow_through(df_fn(15), 3, dirn)
                self.assertGreaterEqual(val, -1.5)
                self.assertLessEqual(val, 2.0)


class TailQualityTests(unittest.TestCase):

    def test_zero_for_zero_spike_bars(self):
        self.assertEqual(_score_tail_quality(trend_up_df(10), 0, "up"), 0.0)

    def test_bounded(self):
        df = trend_up_df(10)
        v = _score_tail_quality(df, 5, "up")
        self.assertGreaterEqual(v, -0.5)
        self.assertLessEqual(v, 1.0)


class BodyGapsTests(unittest.TestCase):

    def test_below_two_spike_bars_is_zero(self):
        self.assertEqual(_score_body_gaps(trend_up_df(5), 1, "up"), 0.0)

    def test_capped_at_one(self):
        df = trend_up_df(10, step=2.0)
        # gappy open/close — lots of body gaps
        v = _score_body_gaps(df, 8, "up")
        self.assertLessEqual(v, 1.0)


class MaSeparationTests(unittest.TestCase):

    def test_returns_zero_for_too_short(self):
        df = trend_up_df(5)
        self.assertEqual(_score_ma_separation(df, "up"), 0.0)

    def test_strong_bull_scores_non_negative(self):
        df = trend_up_df(40)
        self.assertGreaterEqual(_score_ma_separation(df, "up"), 0.0)


class FailedCounterSetupsTests(unittest.TestCase):

    def test_short_df_returns_zero(self):
        self.assertEqual(_score_failed_counter_setups(trend_up_df(2), "up"), 0.0)

    def test_bounded(self):
        v = _score_failed_counter_setups(trend_up_df(15), "up")
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


class VolumeConfirmationTests(unittest.TestCase):

    def test_no_volume_col_returns_zero(self):
        df = trend_up_df(10).drop(columns=["volume"])
        self.assertEqual(_score_volume_confirmation(df, 3), 0.0)

    def test_all_zero_volume_returns_zero(self):
        df = trend_up_df(10)
        df["volume"] = 0
        self.assertEqual(_score_volume_confirmation(df, 3), 0.0)

    def test_high_spike_volume_scores_positive(self):
        df = trend_up_df(15)
        vols = [1_000_000] * 3 + [100_000] * 12  # 10x spike
        df["volume"] = vols
        self.assertGreater(_score_volume_confirmation(df, 3), 0.0)


class MajorityTrendBarsTests(unittest.TestCase):

    def test_short_df_zero(self):
        self.assertEqual(_score_majority_trend_bars(trend_up_df(2), "up"), 0.0)

    def test_pure_bull_positive(self):
        self.assertGreater(_score_majority_trend_bars(trend_up_df(10), "up"), 0.0)

    def test_pure_bull_scored_bear_is_negative(self):
        self.assertEqual(_score_majority_trend_bars(trend_up_df(10), "down"), -1.0)


class MicroGapsTests(unittest.TestCase):

    def test_short_df_zero(self):
        self.assertEqual(_score_micro_gaps(trend_up_df(2), "up"), 0.0)

    def test_bounded(self):
        v = _score_micro_gaps(trend_up_df(15), "up")
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 2.0)


class TrendingEverythingTests(unittest.TestCase):

    def test_short_df_zero(self):
        self.assertEqual(_score_trending_everything(trend_up_df(3), "up"), 0.0)

    def test_strong_bull_trend_hits_cap(self):
        v = _score_trending_everything(trend_up_df(30), "up")
        self.assertEqual(v, 2.0)

    def test_strong_bull_trend_scored_bear_is_negative(self):
        v = _score_trending_everything(trend_up_df(30), "down")
        self.assertEqual(v, -1.0)


class LevelsBrokenTests(unittest.TestCase):

    def test_short_df_zero(self):
        self.assertEqual(_score_levels_broken(trend_up_df(3), "up", 100.0), 0.0)

    def test_strong_bull_trend_breaks_levels(self):
        df = trend_up_df(30)
        v = _score_levels_broken(df, "up", prior_close=99.0)
        self.assertGreater(v, 0.0)


class SmallPullbackTrendTests(unittest.TestCase):

    def test_invalid_direction_returns_zero(self):
        self.assertEqual(_score_small_pullback_trend(trend_up_df(15), "bogus"), 0.0)

    def test_too_few_bars_returns_zero(self):
        self.assertEqual(_score_small_pullback_trend(trend_up_df(2), "up"), 0.0)

    def test_bounded_at_three(self):
        v = _score_small_pullback_trend(trend_up_df(SPT_LOOKBACK_BARS), "up")
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 3.0)


class TrendingSwingsTests(unittest.TestCase):

    def test_too_few_swings_returns_zero(self):
        self.assertEqual(_score_trending_swings(trend_up_df(3), "up"), 0.0)

    def test_bounded(self):
        v = _score_trending_swings(trend_up_df(30), "up")
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 2.0)


class SpikeDurationTests(unittest.TestCase):

    def test_zero_spike_bars_is_minus_half(self):
        self.assertEqual(_score_spike_duration(trend_up_df(10), 0, "up"), -0.5)

    def test_long_sustained_bull_hits_cap(self):
        df = trend_up_df(12)
        v = _score_spike_duration(df, 8, "up")
        self.assertEqual(v, 2.0)


class UrgencyRawMaxTests(unittest.TestCase):

    def test_raw_max_positive(self):
        self.assertGreater(URGENCY_RAW_MAX, 0)


# ── Phase 3f-2 uncertainty scorer tests ──

class CheckLiquidityTests(unittest.TestCase):

    def test_no_volume_col_fails(self):
        df = trend_up_df(10).drop(columns=["volume"])
        result = _check_liquidity(df)
        self.assertFalse(result["passed"])

    def test_too_few_bars_fails(self):
        df = trend_up_df(2)
        result = _check_liquidity(df)
        self.assertFalse(result["passed"])

    def test_high_volume_passes(self):
        df = trend_up_df(20)
        df["volume"] = 1_000_000  # $100 × 1M = $100M per bar
        result = _check_liquidity(df)
        self.assertTrue(result["passed"])
        self.assertGreater(result["avg_dollar_vol"], LIQUIDITY_MIN_DOLLAR_VOL)

    def test_returns_expected_keys(self):
        df = trend_up_df(20)
        result = _check_liquidity(df)
        for key in ("passed", "avg_dollar_vol", "bars_measured"):
            self.assertIn(key, result)


class LiquidityGapsTests(unittest.TestCase):

    def test_short_df_zero(self):
        self.assertEqual(_score_liquidity_gaps(trend_up_df(3)), 0.0)

    def test_no_gaps_scores_zero(self):
        # close → next open is always the same (no gap) by construction
        opens = [100.0] * 20
        closes = [100.1] * 20
        highs = [100.2] * 20
        lows = [99.9] * 20
        df = mk_df(opens, closes, highs=highs, lows=lows)
        self.assertEqual(_score_liquidity_gaps(df), 0.0)

    def test_bounded(self):
        v = _score_liquidity_gaps(trend_up_df(20))
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 2.0)


class TwoSidedRatioTests(unittest.TestCase):

    def test_pure_bull_scored_up_is_zero(self):
        self.assertEqual(_score_two_sided_ratio(trend_up_df(10), "up"), 0.0)

    def test_pure_bull_scored_down_hits_cap(self):
        # all 10 bars are bull → countertrend ratio = 1.0 for bear direction
        self.assertEqual(_score_two_sided_ratio(trend_up_df(10), "down"), 3.0)


class UncertaintyTests(unittest.TestCase):

    def test_tiny_df_returns_half_max(self):
        df = trend_up_df(2)
        score, dirn = _score_uncertainty(df, "up")
        self.assertAlmostEqual(score, UNCERTAINTY_RAW_MAX * 0.5, places=5)
        self.assertEqual(dirn, "unclear")

    def test_bull_trend_returns_always_in_long(self):
        df = trend_up_df(20)
        _score, direction = _score_uncertainty(df, "up")
        self.assertEqual(direction, "long")

    def test_bear_trend_returns_always_in_short(self):
        df = trend_down_df(20)
        _score, direction = _score_uncertainty(df, "down")
        self.assertEqual(direction, "short")

    def test_score_non_negative(self):
        for df_fn in (trend_up_df, trend_down_df, chop_df):
            for dirn in ("up", "down"):
                score, _ = _score_uncertainty(df_fn(20), dirn)
                self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
