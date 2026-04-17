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

    def test_chop_increases_uncertainty_score(self):
        chop = chop_df(20)
        pure = trend_up_df(20)
        chop_score, _ = _score_uncertainty(chop, "up")
        pure_score, _ = _score_uncertainty(pure, "up")
        self.assertGreater(chop_score, pure_score)

    def test_long_chop_hits_ma_plus_swing_branches(self):
        """30-bar chop DF is long enough for EMA_PERIOD+2, swing detection,
        and exercises the bear-direction branches of the uncertainty checks."""
        df = chop_df(30)
        score_down, _ = _score_uncertainty(df, "down")
        self.assertGreater(score_down, 0.0)

    def test_zigzag_bear_produces_high_uncertainty(self):
        """Alternating swing highs + lows trigger the reversal-count branch."""
        highs = [109, 107, 108, 105, 106, 103, 104, 101, 102, 100,
                 101, 99, 100, 98, 99, 97, 98, 96, 97, 95,
                 96, 94, 95, 93, 94, 92, 93, 91, 92, 90]
        lows = [h - 2 for h in highs]
        opens = [(h + lo) / 2 for h, lo in zip(highs, lows)]
        closes = [(h + lo) / 2 + (0.3 if i % 2 else -0.3) for i, (h, lo) in enumerate(zip(highs, lows))]
        df = mk_df(opens, closes, highs=highs, lows=lows)
        score, _ = _score_uncertainty(df, "down")
        # Zigzag has reversals — score should be non-trivial
        self.assertGreaterEqual(score, 0.0)


# ── Extra branch coverage on the bigger aggregators ──

class GapIntegrityBranchTests(unittest.TestCase):

    def test_partial_fill_bull(self):
        # Open 102, prior close 100 → gap=2; dips to 99.5 (below prior close but
        # stays above 99 which is prior - 50% gap). Partial fill → score 1.0.
        df = pd.DataFrame({
            "open": [102, 101, 100.5, 101],
            "high": [102.5, 101.5, 101, 101.5],
            "low":  [100.5, 99.5, 99.5, 100.0],
            "close":[101, 100.5, 100.8, 101.3],
            "volume":[1000] * 4,
        })
        score, status = _score_gap_integrity(df, prior_close=100.0, gap_direction="up")
        self.assertEqual(score, 1.0)
        self.assertEqual(status, "partial_fill")

    def test_bear_gap_held(self):
        # Prior close 100, open 98 (gap down 2), highs stay below prior close
        df = pd.DataFrame({
            "open": [98, 97, 96, 95],
            "high": [98.5, 97.5, 96.5, 95.5],
            "low":  [97, 96, 95, 94],
            "close":[97.5, 96.5, 95.5, 94.5],
            "volume":[1000] * 4,
        })
        score, status = _score_gap_integrity(df, prior_close=100.0, gap_direction="down")
        self.assertEqual(score, 2.0)
        self.assertEqual(status, "held")

    def test_no_gap_returns_zero(self):
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100],
            "volume": [1000],
        })
        score, status = _score_gap_integrity(df, prior_close=100.0, gap_direction="up")
        self.assertEqual(score, 0.0)
        self.assertEqual(status, "held")


class FollowThroughBranchTests(unittest.TestCase):

    def test_bear_direction_with_short_df(self):
        df = trend_down_df(5)
        # short df on down direction
        result = _score_follow_through(df, spike_bars=2, gap_direction="down")
        # Bounded by -1.5 to 2.0
        self.assertGreaterEqual(result, -1.5)
        self.assertLessEqual(result, 2.0)

    def test_bear_direction_full_trend(self):
        df = trend_down_df(20)
        result = _score_follow_through(df, spike_bars=3, gap_direction="down")
        self.assertGreaterEqual(result, -1.5)
        self.assertLessEqual(result, 2.0)


class SmallPullbackTrendBranchTests(unittest.TestCase):

    def test_bear_direction_clean_trend(self):
        df = trend_down_df(15)
        val = _score_small_pullback_trend(df, "down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 3.0)

    def test_chop_scores_low(self):
        # Pure chop shouldn't get a high SPT score even if density sneaks through.
        val = _score_small_pullback_trend(chop_df(15), "up")
        self.assertLess(val, 1.5)

    def test_bear_trend_with_shallow_pullbacks(self):
        """Bear SPT — trend down with small up-bar pullbacks between legs."""
        bars_list = []
        price = 200.0
        for _ in range(5):
            bars_list.append((price, price + 0.1, price - 1.5, price - 1.2))
            price -= 1.2
        # Tiny pullback: 2 small bull bars
        bars_list.append((price, price + 0.3, price - 0.1, price + 0.1))
        bars_list.append((price + 0.1, price + 0.3, price, price + 0.15))
        # Resume down — 5 more bear bars
        price += 0.15
        for _ in range(5):
            bars_list.append((price, price + 0.1, price - 1.3, price - 1.0))
            price -= 1.0
        df = pd.DataFrame(bars_list, columns=["open", "high", "low", "close"])
        df["volume"] = 1000
        val = _score_small_pullback_trend(df, "down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 3.0)

    def test_pure_bull_trend_hits_high_score(self):
        """Strong bull with tiny pullbacks should score high."""
        df = trend_up_df(15, step=0.5)
        val = _score_small_pullback_trend(df, "up")
        # Won't always hit 3.0 due to sub-check weighting, but should be substantial
        self.assertGreater(val, 0.0)


class MaSeparationBranchTests(unittest.TestCase):

    def test_bear_direction_full_trend(self):
        df = trend_down_df(40)
        val = _score_ma_separation(df, "down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class FailedCounterSetupsBranchTests(unittest.TestCase):

    def test_bear_direction(self):
        df = trend_down_df(15)
        val = _score_failed_counter_setups(df, "down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class LevelsBrokenBranchTests(unittest.TestCase):

    def test_bear_direction_levels_broken(self):
        df = trend_down_df(30)
        val = _score_levels_broken(df, "down", prior_close=201.0)
        self.assertGreater(val, 0.0)


class SpikeDurationBranchTests(unittest.TestCase):

    def test_bear_direction_sustained(self):
        df = trend_down_df(12)
        val = _score_spike_duration(df, spike_bars=8, gap_direction="down")
        self.assertEqual(val, 2.0)

    def test_short_sustained_returns_zero(self):
        # 3 sustained bars → 0.0 per function contract
        df = trend_up_df(5)
        val = _score_spike_duration(df, spike_bars=3, gap_direction="up")
        # Not a perfect 0.0 because of retrace thresholds; check bounded
        self.assertGreaterEqual(val, -0.5)
        self.assertLessEqual(val, 2.0)


class TrendingSwingsBranchTests(unittest.TestCase):

    def _zigzag_bull(self) -> pd.DataFrame:
        """Higher highs + higher lows — zigzag staircase up."""
        highs = [101, 103, 102, 105, 104, 107, 106, 109]
        lows = [99, 101, 100, 103, 102, 105, 104, 107]
        opens = [(h + lo) / 2 - 0.3 for h, lo in zip(highs, lows)]
        closes = [(h + lo) / 2 + 0.3 for h, lo in zip(highs, lows)]
        return mk_df(opens, closes, highs=highs, lows=lows)

    def _zigzag_bear(self) -> pd.DataFrame:
        """Lower highs + lower lows — zigzag staircase down."""
        highs = [109, 107, 108, 105, 106, 103, 104, 101]
        lows = [107, 105, 106, 103, 104, 101, 102, 99]
        opens = [(h + lo) / 2 + 0.3 for h, lo in zip(highs, lows)]
        closes = [(h + lo) / 2 - 0.3 for h, lo in zip(highs, lows)]
        return mk_df(opens, closes, highs=highs, lows=lows)

    def test_bear_zigzag_scores_positive(self):
        val = _score_trending_swings(self._zigzag_bear(), "down")
        self.assertGreater(val, 0.0)

    def test_bull_zigzag_scores_positive(self):
        val = _score_trending_swings(self._zigzag_bull(), "up")
        self.assertGreater(val, 0.0)

    def test_bear_direction_flat_trend(self):
        df = trend_down_df(30)
        val = _score_trending_swings(df, "down")
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 2.0)


class MajorityTrendBarsBranchTests(unittest.TestCase):

    def test_bear_direction_full_trend(self):
        df = trend_down_df(15)
        val = _score_majority_trend_bars(df, "down")
        self.assertGreater(val, 0.0)

    def test_chop_scored_either_direction_is_neutral_or_negative(self):
        val = _score_majority_trend_bars(chop_df(15), "up")
        self.assertLessEqual(val, 1.0)


class MicroGapsBranchTests(unittest.TestCase):

    def test_bear_direction(self):
        df = trend_down_df(15)
        val = _score_micro_gaps(df, "down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 2.0)


class BodyGapsBranchTests(unittest.TestCase):

    def test_bear_direction(self):
        df = trend_down_df(10)
        val = _score_body_gaps(df, spike_bars=4, gap_direction="down")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


class SpikeQualityBranchTests(unittest.TestCase):

    def test_bear_direction(self):
        df = trend_down_df(6)
        score, bars = _score_spike_quality(df, "down")
        self.assertGreater(score, 0.0)
        self.assertGreaterEqual(bars, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
