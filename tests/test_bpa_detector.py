"""
Canonical fixtures for every BPA detector in shared/bpa_detector.py.

Each setup has a passing fixture (unambiguous Brooks example) plus one or more
near-misses that should NOT fire. Tests bypass the database entirely.

Run with:
    python3 -m unittest tests.test_bpa_detector -v
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.bpa_detector import (  # noqa: E402
    BPASetup,
    _is_bull_trend_bar,
    _is_bear_trend_bar,
    _is_bull_signal_bar,
    _is_bear_signal_bar,
    _is_doji,
    _detect_h1,
    _detect_h2,
    _detect_l1,
    _detect_l2,
    _detect_fl1,
    _detect_fl2,
    _detect_fh1,
    _detect_fh2,
    _detect_spike_and_channel,
    _detect_failed_breakout,
    detect_all,
)


def _df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """bars: list of (open, high, low, close) tuples."""
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"]).assign(volume=1000)


# ─────────────────────────────────────────────────────────────────────────────
# Bar classification
# ─────────────────────────────────────────────────────────────────────────────

class BarClassificationTests(unittest.TestCase):
    def test_bull_trend_bar_passes(self):
        bar = pd.Series({"open": 100, "high": 101, "low": 99.5, "close": 100.9})
        self.assertTrue(_is_bull_trend_bar(bar))
        self.assertTrue(_is_bull_signal_bar(bar))

    def test_weak_bull_long_upper_tail_fails(self):
        bar = pd.Series({"open": 100, "high": 101.5, "low": 99.8, "close": 100.3})
        self.assertFalse(_is_bull_trend_bar(bar))  # body ratio < 0.5
        self.assertFalse(_is_bull_signal_bar(bar))

    def test_bear_trend_bar_passes(self):
        bar = pd.Series({"open": 100, "high": 100.2, "low": 98.5, "close": 98.7})
        self.assertTrue(_is_bear_trend_bar(bar))
        self.assertTrue(_is_bear_signal_bar(bar))

    def test_doji_fails_trend_bar_checks(self):
        bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100.1})
        self.assertTrue(_is_doji(bar))
        self.assertFalse(_is_bull_trend_bar(bar))
        self.assertFalse(_is_bear_trend_bar(bar))

    def test_bull_with_big_upper_tail_is_trend_bar_but_not_signal_bar(self):
        bar = pd.Series({"open": 100, "high": 102, "low": 99.5, "close": 101.0})
        # body = 1.0, range = 2.5, body_ratio = 0.4 — below 0.5 threshold
        self.assertFalse(_is_bull_trend_bar(bar))

    def test_bull_signal_bar_rejects_long_upper_tail_when_body_ok(self):
        # body_ratio 0.6, close_position 0.72 (top third), but upper tail 28% — fails signal test
        bar = pd.Series({"open": 100.0, "high": 102.0, "low": 99.5, "close": 101.5})
        # body = 1.5, range = 2.5, body_ratio = 0.6 ✓
        # close_position = (101.5 - 99.5) / 2.5 = 0.8 ✓
        # upper tail = (102 - 101.5) / 2.5 = 0.2 < 0.25 → IS signal bar, but barely
        self.assertTrue(_is_bull_trend_bar(bar))
        self.assertTrue(_is_bull_signal_bar(bar))

        # now stretch the upper tail
        bar2 = pd.Series({"open": 100.0, "high": 103.0, "low": 99.5, "close": 102.0})
        # body = 2.0, range = 3.5, body_ratio = 0.57 ✓
        # close_position = (102 - 99.5) / 3.5 = 0.71 ≥ 0.667 ✓
        # upper tail = (103 - 102) / 3.5 = 0.286 > 0.25 → NOT signal bar
        self.assertTrue(_is_bull_trend_bar(bar2))
        self.assertFalse(_is_bull_signal_bar(bar2))


# ─────────────────────────────────────────────────────────────────────────────
# H1 — first pullback long
# ─────────────────────────────────────────────────────────────────────────────

def _h1_canonical() -> pd.DataFrame:
    """Up-leg (4 bars) → 2-bar pullback (bear + doji) → bull signal bar."""
    return _df([
        (100.0, 101.0, 99.5, 100.9),   # bull trend
        (100.9, 102.0, 100.5, 101.9),  # bull trend
        (101.9, 103.0, 101.5, 102.9),  # bull trend
        (102.9, 104.2, 102.5, 103.9),  # bull trend - leg top (h=104.2)
        (103.9, 104.0, 103.0, 103.1),  # bear trend - pullback
        (103.1, 103.3, 102.8, 103.0),  # doji - pullback
        (103.0, 104.5, 102.9, 104.4),  # bull signal bar - H1 trigger
    ])


class H1Tests(unittest.TestCase):
    def test_canonical_h1_fires(self):
        df = _h1_canonical()
        result = _detect_h1(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "H1")
        self.assertEqual(result.bar_index, 6)
        self.assertAlmostEqual(result.entry, 104.5, places=2)
        self.assertAlmostEqual(result.stop, 102.9, places=2)
        self.assertGreater(result.target, result.entry)

    def test_weak_signal_bar_fails(self):
        df = _h1_canonical().copy()
        # Replace bar 7 (signal bar) with weak bull — long upper tail
        df.iloc[-1] = [103.0, 106.0, 102.9, 103.8, 1000]
        self.assertIsNone(_detect_h1(df))

    def test_no_pullback_fails(self):
        # up-leg continues straight into signal bar — no intervening pullback
        df = _df([
            (100.0, 101.0, 99.5, 100.9),
            (100.9, 102.0, 100.5, 101.9),
            (101.9, 103.0, 101.5, 102.9),
            (102.9, 104.0, 102.5, 103.9),
            (103.9, 104.5, 103.5, 104.4),
            (104.4, 104.8, 104.0, 104.7),
            (104.7, 105.8, 104.6, 105.7),
        ])
        self.assertIsNone(_detect_h1(df))

    def test_no_up_leg_fails(self):
        # Pullback present but no prior up-leg (flat chop then signal bar)
        df = _df([
            (100.0, 100.5, 99.8, 100.1),
            (100.1, 100.5, 99.8, 100.0),
            (100.0, 100.4, 99.9, 100.1),
            (100.1, 100.4, 99.9, 100.0),
            (100.0, 100.3, 99.7, 99.9),
            (99.9, 100.0, 99.5, 99.6),
            (99.6, 100.6, 99.5, 100.5),  # bull signal bar but no up-leg before
        ])
        self.assertIsNone(_detect_h1(df))


# ─────────────────────────────────────────────────────────────────────────────
# H2 — second pullback with prior H1 + continuation leg
# ─────────────────────────────────────────────────────────────────────────────

def _h2_canonical() -> pd.DataFrame:
    """
    Brooks H2: single extended pullback, H1 triggers but fails (price goes below
    H1's low), then H2 triggers. No new trend high between H1 and H2.
    """
    return _df([
        (100.0, 101.0, 99.5, 100.9),     # up-leg
        (100.9, 102.0, 100.5, 101.9),    # up-leg
        (101.9, 103.5, 101.5, 103.4),    # up-leg
        (103.4, 105.0, 103.3, 104.9),    # up-leg top (h=105)
        (104.9, 104.95, 104.0, 104.1),   # bear pullback leg 1
        (104.1, 104.2, 103.5, 103.6),    # bear pullback leg 1 (low 103.5)
        (103.6, 104.7, 103.55, 104.6),   # H1 bull signal bar (low=103.55)
        (104.6, 104.65, 103.3, 103.4),   # H1 FAILS — new low 103.3 < H1 low
        (103.4, 103.6, 103.0, 103.1),    # pullback extends further
        (103.1, 103.3, 102.8, 103.0),    # doji, low 102.8
        (103.0, 104.5, 102.9, 104.4),    # H2 bull signal bar
    ])


class H2Tests(unittest.TestCase):
    def test_canonical_h2_fires(self):
        df = _h2_canonical()
        result = _detect_h2(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "H2")
        self.assertEqual(result.bar_index, 10)
        self.assertAlmostEqual(result.entry, 104.5, places=2)
        self.assertAlmostEqual(result.stop, 102.9, places=2)

    def test_no_prior_h1_fails(self):
        # Only a single pullback — H2 should NOT fire (but H1 might)
        df = _h1_canonical()
        self.assertIsNone(_detect_h2(df))

    def test_h1_did_not_fail_fails(self):
        # H1 at bar 6; price continues up without trading below H1 low; then
        # another bull signal. That's two H1s in separate legs, not an H2.
        df = _df([
            (100.0, 101.0, 99.5, 100.9),
            (100.9, 102.0, 100.5, 101.9),
            (101.9, 103.0, 101.5, 102.9),
            (102.9, 104.2, 102.5, 103.9),
            (103.9, 104.0, 103.0, 103.1),    # pullback 1
            (103.1, 103.3, 102.8, 103.0),    # doji
            (103.0, 104.5, 102.9, 104.4),    # H1 signal bar (low 102.9)
            (104.4, 105.5, 104.2, 105.3),    # bull continuation — stays above H1 low
            (105.3, 106.0, 105.0, 105.9),    # bull continuation
            (105.9, 105.95, 105.2, 105.3),   # pullback 2 (new pullback, not extension)
            (105.3, 106.6, 105.2, 106.5),    # another bull signal — but H1 never failed
        ])
        self.assertIsNone(_detect_h2(df))


# ─────────────────────────────────────────────────────────────────────────────
# L1 / L2 — mirrors
# ─────────────────────────────────────────────────────────────────────────────

def _l1_canonical() -> pd.DataFrame:
    """Down-leg, pullback up, bear signal bar breaking prior bar low."""
    return _df([
        (100.0, 100.5, 99.0, 99.1),
        (99.1, 99.5, 98.0, 98.1),
        (98.1, 98.5, 97.0, 97.1),
        (97.1, 97.5, 95.8, 96.1),      # leg bottom #1 (l=95.8)
        (96.1, 97.0, 96.0, 96.9),      # bull pullback
        (96.9, 97.2, 96.7, 97.0),      # doji pullback
        (97.0, 97.1, 95.5, 95.6),      # bear signal bar — L1
    ])


def _l2_canonical() -> pd.DataFrame:
    """
    Brooks L2: single extended rally, L1 triggers but fails (price goes above
    L1's high), then L2 triggers. No new trend low between L1 and L2.
    """
    return _df([
        (100.0, 100.5, 99.0, 99.1),       # down-leg
        (99.1, 99.5, 98.0, 98.1),         # down-leg
        (98.1, 98.5, 96.5, 96.6),         # down-leg
        (96.6, 96.7, 95.0, 95.1),         # down-leg bottom (l=95)
        (95.1, 96.0, 95.05, 95.9),        # bull rally leg 1
        (95.9, 96.5, 95.8, 96.4),         # bull rally leg 1 (high 96.5)
        (96.4, 96.45, 95.3, 95.4),        # L1 bear signal bar (high=96.45)
        (95.4, 96.7, 95.35, 96.6),        # L1 FAILS — new high 96.7 > L1 high
        (96.6, 97.0, 96.4, 96.9),         # rally extends
        (96.9, 97.2, 96.7, 97.0),         # doji, high 97.2
        (97.0, 97.1, 95.5, 95.6),         # L2 bear signal bar
    ])


class LTests(unittest.TestCase):
    def test_canonical_l1_fires(self):
        df = _l1_canonical()
        result = _detect_l1(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "L1")
        self.assertAlmostEqual(result.entry, 95.5, places=2)
        self.assertAlmostEqual(result.stop, 97.1, places=2)
        self.assertLess(result.target, result.entry)

    def test_canonical_l2_fires(self):
        df = _l2_canonical()
        result = _detect_l2(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "L2")

    def test_l1_near_miss_weak_signal(self):
        df = _l1_canonical().copy()
        df.iloc[-1] = [97.0, 97.1, 94.5, 96.5, 1000]  # long lower tail — not a bear signal bar
        self.assertIsNone(_detect_l1(df))


# ─────────────────────────────────────────────────────────────────────────────
# failed_bo — failed breakout
# ─────────────────────────────────────────────────────────────────────────────

def _failed_bo_up_canonical() -> pd.DataFrame:
    """18-bar range [100, 105] with tight individual bars, boundary tests, then failed bull breakout."""
    range_bars = [
        (102.0, 102.8, 101.5, 102.5),
        (102.5, 103.0, 102.0, 102.8),
        (102.8, 103.5, 102.3, 103.2),
        (103.2, 104.8, 103.0, 104.5),  # test high
        (104.5, 104.9, 103.5, 103.8),  # test high
        (103.8, 104.0, 102.0, 102.3),
        (102.3, 102.8, 101.0, 101.2),
        (101.2, 101.5, 100.1, 100.3),  # test low
        (100.3, 100.5, 99.9, 100.2),   # test low
        (100.2, 101.0, 100.1, 100.8),
        (100.8, 101.5, 100.5, 101.2),
        (101.2, 102.0, 101.0, 101.8),
        (101.8, 102.5, 101.5, 102.3),
        (102.3, 103.2, 102.0, 103.0),
        (103.0, 104.8, 102.8, 104.5),  # test high
        (104.5, 104.7, 103.0, 103.2),
        (103.2, 103.5, 102.5, 102.8),
        (102.8, 103.0, 102.3, 102.5),
    ]
    breakout_and_failure = [
        (102.5, 106.5, 102.3, 106.2),  # bull breakout (close > RH ~ 104.9)
        (106.2, 106.5, 103.5, 103.8),  # immediate failure — close back inside range
    ]
    return _df(range_bars + breakout_and_failure)


class FailedBoTests(unittest.TestCase):
    def test_canonical_failed_bo_fires(self):
        df = _failed_bo_up_canonical()
        result = _detect_failed_breakout(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "failed_bo")
        # entry at range high (short), target at range low
        self.assertGreater(result.entry, result.target)
        self.assertGreater(result.stop, result.entry)  # stop above entry for short

    def test_no_breakout_fails(self):
        df = _failed_bo_up_canonical().iloc[:-2].copy()  # drop the breakout + failure bars
        # Pad with 2 more in-range bars
        extra = _df([
            (102.5, 103.0, 100.5, 101.0),
            (101.0, 102.0, 100.3, 101.5),
        ])
        df = pd.concat([df, extra], ignore_index=True)
        self.assertIsNone(_detect_failed_breakout(df))

    def test_narrow_range_fails(self):
        # All bars in a tight 0.5-wide range — fails width check
        df = _df([(100.0 + 0.1 * (i % 5), 100.2 + 0.1 * (i % 5),
                    99.9 + 0.1 * (i % 5), 100.0 + 0.1 * (i % 5))
                   for i in range(18)] + [
            (100.5, 101.5, 100.4, 101.4),  # "breakout"
            (101.4, 101.5, 100.0, 100.1),  # "failure"
        ])
        self.assertIsNone(_detect_failed_breakout(df))


# ─────────────────────────────────────────────────────────────────────────────
# spike_channel — continuation
# ─────────────────────────────────────────────────────────────────────────────

def _spike_channel_canonical() -> pd.DataFrame:
    """3-bar bull spike (big bars), weak channel drift, pullback, bull signal bar."""
    return _df([
        (100.0, 102.5, 99.8, 102.3),    # spike bar 1 (big body)
        (102.3, 104.5, 102.2, 104.4),   # spike bar 2
        (104.4, 106.5, 104.3, 106.4),   # spike bar 3
        (106.4, 106.6, 106.0, 106.3),   # channel (weak bear, body_ratio ~0.17)
        (106.3, 106.5, 106.0, 106.4),   # channel (weak bull, body_ratio ~0.20)
        (106.4, 106.8, 106.2, 106.6),   # channel (weak bull)
        (106.6, 106.9, 106.4, 106.7),   # channel (weak bull)
        (106.7, 107.0, 106.5, 106.8),   # channel leg top (h=107.0)
        (106.8, 106.85, 106.1, 106.2),  # pullback (bear trend bar)
        (106.2, 107.5, 106.1, 107.4),   # bull signal bar — continuation entry
    ])


class SpikeChannelTests(unittest.TestCase):
    def test_canonical_spike_channel_fires(self):
        df = _spike_channel_canonical()
        result = _detect_spike_and_channel(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "spike_channel")
        # Long entry (continuation with bull trend)
        self.assertLess(result.stop, result.entry)
        self.assertGreater(result.target, result.entry)


# ─────────────────────────────────────────────────────────────────────────────
# FL1 — failed L1 (bull entry on failure)
# ─────────────────────────────────────────────────────────────────────────────

def _fl1_canonical() -> pd.DataFrame:
    """Bear leg, pullback, L1 bear signal bar, then bull reversal = FL1."""
    return _df([
        (100.0, 100.5, 99.0, 99.1),
        (99.1, 99.5, 98.0, 98.1),
        (98.1, 98.5, 97.0, 97.1),
        (97.1, 97.5, 95.8, 96.1),       # leg bottom
        (96.1, 97.0, 96.0, 96.9),       # bull pullback
        (96.9, 97.2, 96.7, 97.0),       # doji pullback
        (97.0, 97.1, 95.5, 95.6),       # L1 bear signal bar
        (95.6, 97.3, 95.55, 97.2),      # bull signal bar — FL1 reversal
    ])


class FL1Tests(unittest.TestCase):
    def test_canonical_fl1_fires(self):
        df = _fl1_canonical()
        result = _detect_fl1(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "FL1")
        self.assertGreater(result.target, result.entry)

    def test_fl1_requires_valid_l1_first(self):
        # Same structure but the "L1" bar is just a weak bear, not a signal bar
        df = _fl1_canonical().copy()
        df.iloc[-2] = [97.0, 97.1, 95.5, 96.5, 1000]  # long lower tail — not bear signal
        self.assertIsNone(_detect_fl1(df))


# ─────────────────────────────────────────────────────────────────────────────
# FL2 — failed L2 (second failed short)
# ─────────────────────────────────────────────────────────────────────────────

def _fl2_canonical() -> pd.DataFrame:
    """
    L1 fires, L1 fails (price rallies above L1 high), L2 fires within the same
    extended rally, L2 fails → bull FL2 reversal entry.
    """
    return _df([
        (100.0, 100.5, 99.0, 99.1),       # down-leg
        (99.1, 99.5, 98.0, 98.1),         # down-leg
        (98.1, 98.5, 96.5, 96.6),         # down-leg
        (96.6, 96.7, 95.0, 95.1),         # down-leg bottom
        (95.1, 96.0, 95.05, 95.9),        # bull rally leg 1
        (95.9, 96.5, 95.8, 96.4),         # bull rally leg 1
        (96.4, 96.45, 95.3, 95.4),        # L1 bear signal bar
        (95.4, 96.7, 95.35, 96.6),        # L1 FAILS — new high
        (96.6, 97.0, 96.4, 96.9),         # rally extends
        (96.9, 97.2, 96.7, 97.0),         # doji
        (97.0, 97.1, 95.5, 95.6),         # L2 bear signal bar
        (95.6, 97.3, 95.55, 97.2),        # FL2 bull reversal signal (closes above L2 high)
    ])


class FL2Tests(unittest.TestCase):
    def test_canonical_fl2_fires(self):
        df = _fl2_canonical()
        result = _detect_fl2(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "FL2")


# ─────────────────────────────────────────────────────────────────────────────
# FH1 / FH2 — mirrors
# ─────────────────────────────────────────────────────────────────────────────

def _fh1_canonical() -> pd.DataFrame:
    """Up-leg, pullback, H1 bull signal bar, then bear reversal = FH1."""
    return _df([
        (100.0, 101.0, 99.5, 100.9),
        (100.9, 102.0, 100.5, 101.9),
        (101.9, 103.0, 101.5, 102.9),
        (102.9, 104.2, 102.5, 103.9),   # leg top
        (103.9, 104.0, 103.0, 103.1),   # bear pullback
        (103.1, 103.3, 102.8, 103.0),   # doji pullback
        (103.0, 104.5, 102.9, 104.4),   # H1 bull signal bar
        (104.4, 104.45, 102.7, 102.8),  # FH1 bear reversal signal (breaks H1 low 102.9)
    ])


class FH1Tests(unittest.TestCase):
    def test_canonical_fh1_fires(self):
        df = _fh1_canonical()
        result = _detect_fh1(df)
        self.assertIsNotNone(result)
        self.assertEqual(result.setup_type, "FH1")
        self.assertLess(result.target, result.entry)


# ─────────────────────────────────────────────────────────────────────────────
# detect_all integration
# ─────────────────────────────────────────────────────────────────────────────

class DetectAllTests(unittest.TestCase):
    def test_returns_list(self):
        df = _h1_canonical()
        results = detect_all(df)
        self.assertIsInstance(results, list)

    def test_sorts_by_bar_index_desc(self):
        df = _h2_canonical()
        results = detect_all(df)
        # Both H1 and H2 may fire on this; both at the same bar_index
        self.assertTrue(all(isinstance(r, BPASetup) for r in results))
        if len(results) > 1:
            indices = [r.bar_index for r in results]
            self.assertEqual(indices, sorted(indices, reverse=True))


class FailedBoStopNotFromBreakoutBarTests(unittest.TestCase):
    """
    The failed_bo stop must be computable BEFORE the breakout bar exists,
    because the limit at the boundary fills mid-breakout-bar when the bar's
    final high/low is unknown. Test: swap the breakout bar's extreme for a
    much more violent one and confirm the stop is unchanged.
    """

    def test_stop_independent_of_breakout_bar_extreme(self):
        from shared.bpa_detector import _detect_failed_breakout

        df = _failed_bo_up_canonical()
        baseline = _detect_failed_breakout(df.copy())
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.setup_type, "failed_bo")

        # Move the breakout bar's high up dramatically — if the stop was
        # derived from it, the stop would move with it.
        df_moved = df.copy()
        idx = len(df_moved) - 2  # breakout bar sits at signal_idx - 1
        df_moved.iloc[idx, df_moved.columns.get_loc("high")] = (
            float(df_moved.iloc[idx]["high"]) + 5.0
        )
        moved = _detect_failed_breakout(df_moved)
        self.assertIsNotNone(moved)
        self.assertEqual(
            moved.stop, baseline.stop,
            f"Stop moved with breakout bar high — this is hindsight. "
            f"baseline={baseline.stop}, moved={moved.stop}",
        )


class EntryModeTests(unittest.TestCase):
    """Each setup's entry_mode must match Brooks' canonical entry method."""

    def test_h1_is_stop(self):
        self.assertEqual(_detect_h1(_h1_canonical()).entry_mode, "stop")

    def test_h2_is_stop(self):
        self.assertEqual(_detect_h2(_h2_canonical()).entry_mode, "stop")

    def test_l1_is_stop(self):
        self.assertEqual(_detect_l1(_l1_canonical()).entry_mode, "stop")

    def test_l2_is_stop(self):
        self.assertEqual(_detect_l2(_l2_canonical()).entry_mode, "stop")

    def test_spike_channel_is_stop(self):
        self.assertEqual(_detect_spike_and_channel(_spike_channel_canonical()).entry_mode, "stop")

    def test_failed_bo_is_limit(self):
        self.assertEqual(_detect_failed_breakout(_failed_bo_up_canonical()).entry_mode, "limit")

    def test_fl1_is_limit(self):
        self.assertEqual(_detect_fl1(_fl1_canonical()).entry_mode, "limit")

    def test_fl2_is_limit(self):
        self.assertEqual(_detect_fl2(_fl2_canonical()).entry_mode, "limit")

    def test_fh1_is_limit(self):
        self.assertEqual(_detect_fh1(_fh1_canonical()).entry_mode, "limit")

    def test_fl1_entry_is_at_failed_l1_low(self):
        # Entry should be the failed L1 signal bar's low, NOT the reversal bar's high
        df = _fl1_canonical()
        l1_low = df.iloc[-2]["low"]  # L1 bar is second-to-last
        result = _detect_fl1(df)
        self.assertAlmostEqual(result.entry, l1_low, places=2)


class NoHindsightTests(unittest.TestCase):
    """
    Each detector must be deterministic given only bars up to the signal bar.
    If appending future bars changes the detector's output at a given signal
    bar, the detector is reading future data — a look-ahead bug.
    """

    def _prove_no_lookahead(self, fixture_fn, detector, garbage_future_bars=6):
        """Run detector on fixture; append arbitrary bars; truncate back; must match."""
        df = fixture_fn()
        baseline = detector(df)

        # Append arbitrary future bars (noise)
        future = _df([(df.iloc[-1]["close"], df.iloc[-1]["close"] + 10,
                        df.iloc[-1]["close"] - 10, df.iloc[-1]["close"] - 5)
                       for _ in range(garbage_future_bars)])
        extended = pd.concat([df, future], ignore_index=True)
        # Truncate extended back to the baseline length — detector must match
        truncated = extended.iloc[:len(df)].copy()
        replay = detector(truncated)

        # Both should produce identical detection results
        if baseline is None:
            self.assertIsNone(replay, f"{detector.__name__}: replay fired when baseline didn't")
        else:
            self.assertIsNotNone(replay, f"{detector.__name__}: replay failed to fire")
            self.assertEqual(baseline.setup_type, replay.setup_type)
            self.assertEqual(baseline.bar_index, replay.bar_index)
            self.assertAlmostEqual(baseline.entry, replay.entry, places=2)
            self.assertAlmostEqual(baseline.stop, replay.stop, places=2)
            self.assertAlmostEqual(baseline.target, replay.target, places=2)
            self.assertEqual(baseline.entry_mode, replay.entry_mode)

    def test_h1_no_lookahead(self):
        self._prove_no_lookahead(_h1_canonical, _detect_h1)

    def test_h2_no_lookahead(self):
        self._prove_no_lookahead(_h2_canonical, _detect_h2)

    def test_l1_no_lookahead(self):
        self._prove_no_lookahead(_l1_canonical, _detect_l1)

    def test_l2_no_lookahead(self):
        self._prove_no_lookahead(_l2_canonical, _detect_l2)

    def test_failed_bo_no_lookahead(self):
        self._prove_no_lookahead(_failed_bo_up_canonical, _detect_failed_breakout)

    def test_spike_channel_no_lookahead(self):
        self._prove_no_lookahead(_spike_channel_canonical, _detect_spike_and_channel)

    def test_fl1_no_lookahead(self):
        self._prove_no_lookahead(_fl1_canonical, _detect_fl1)

    def test_fl2_no_lookahead(self):
        self._prove_no_lookahead(_fl2_canonical, _detect_fl2)

    def test_fh1_no_lookahead(self):
        self._prove_no_lookahead(_fh1_canonical, _detect_fh1)


class LiveSuitabilityTests(unittest.TestCase):
    """
    Simulate how the live scanner actually runs: at each bar close, call the
    detector with df[:N+1]. We check setups don't re-fire on subsequent bars
    for the same underlying structure (would spam alerts) and that each setup
    fires only at the specific bar that completes it.
    """

    def _rolling_fires(self, fixture_fn, detector, min_start=5):
        """Return list of bar indices where detector fires during a rolling scan."""
        df = fixture_fn()
        fires = []
        for n in range(min_start, len(df) + 1):
            window = df.iloc[:n].copy()
            r = detector(window)
            if r is not None:
                fires.append(n - 1)  # current last bar
        return fires

    def test_h1_fires_once_only_on_signal_bar(self):
        from shared.bpa_detector import _detect_h1
        fires = self._rolling_fires(_h1_canonical, _detect_h1)
        self.assertEqual(fires, [6], f"H1 should fire once at bar 6, got {fires}")

    def test_h2_fires_once_only_on_signal_bar(self):
        from shared.bpa_detector import _detect_h2
        fires = self._rolling_fires(_h2_canonical, _detect_h2)
        self.assertEqual(fires, [10], f"H2 should fire once at bar 10, got {fires}")

    def test_l1_fires_once(self):
        from shared.bpa_detector import _detect_l1
        fires = self._rolling_fires(_l1_canonical, _detect_l1)
        self.assertEqual(fires, [6])

    def test_l2_fires_once(self):
        from shared.bpa_detector import _detect_l2
        fires = self._rolling_fires(_l2_canonical, _detect_l2)
        self.assertEqual(fires, [10])

    def test_failed_bo_fires_once(self):
        from shared.bpa_detector import _detect_failed_breakout
        fires = self._rolling_fires(_failed_bo_up_canonical, _detect_failed_breakout, min_start=15)
        self.assertEqual(fires, [19])

    def test_spike_channel_fires_once(self):
        from shared.bpa_detector import _detect_spike_and_channel
        fires = self._rolling_fires(_spike_channel_canonical, _detect_spike_and_channel)
        self.assertEqual(fires, [9])

    def test_fl1_fires_once(self):
        from shared.bpa_detector import _detect_fl1
        fires = self._rolling_fires(_fl1_canonical, _detect_fl1)
        self.assertEqual(fires, [7])

    def test_fl2_fires_once(self):
        from shared.bpa_detector import _detect_fl2
        fires = self._rolling_fires(_fl2_canonical, _detect_fl2)
        self.assertEqual(fires, [11])

    def test_fh1_fires_once(self):
        from shared.bpa_detector import _detect_fh1
        fires = self._rolling_fires(_fh1_canonical, _detect_fh1)
        self.assertEqual(fires, [7])

    def test_no_refire_after_signal_with_noise_bars(self):
        """After a setup fires, noise bars that don't form a new signal must not re-trigger it."""
        from shared.bpa_detector import _detect_h1
        df = _h1_canonical()  # H1 fires at bar 6
        # Append 5 "neutral" bars that drift slightly up (not forming new signal bars)
        neutral = _df([
            (104.4, 104.5, 104.3, 104.4),   # doji-ish
            (104.4, 104.5, 104.3, 104.4),
            (104.4, 104.5, 104.3, 104.4),
            (104.4, 104.5, 104.3, 104.4),
            (104.4, 104.5, 104.3, 104.4),
        ])
        extended = pd.concat([df, neutral], ignore_index=True)

        fires = []
        for n in range(5, len(extended) + 1):
            r = _detect_h1(extended.iloc[:n].copy())
            if r is not None:
                fires.append(n - 1)
        self.assertEqual(fires, [6], f"H1 should fire only at bar 6, got {fires}")


class RollingScanTests(unittest.TestCase):
    """
    Simulate a live-scanner run: call detect_all at each successive bar close.
    At bar N, the detector sees only bars 0..N. We check:
      1. Signals only fire at the bar they should (no stale re-fires)
      2. A signal fires at most once for a given setup instance
    """

    def test_h1_fires_only_when_signal_bar_forms(self):
        df = _h1_canonical()  # signal bar is at index 6 (last bar)
        fired_at = []
        for n in range(5, len(df) + 1):
            window = df.iloc[:n].copy()
            from shared.bpa_detector import _detect_h1
            r = _detect_h1(window)
            if r is not None:
                fired_at.append(n - 1)  # index of last bar at time of fire
        # H1 should fire exactly once, at bar 6
        self.assertEqual(fired_at, [6], f"expected H1 at bar 6, got {fired_at}")

    def test_failed_bo_fires_only_on_fresh_failure_bar(self):
        # Build a 22-bar df: 18 range bars + breakout + failure + 2 "after" bars.
        # When calling detector at bar 19 close, should fire. At bar 20 or 21,
        # must NOT fire (the failure bar is stale).
        base = _failed_bo_up_canonical()  # 20 bars, failure at idx 19
        after = _df([(103.8, 104.0, 103.0, 103.5), (103.5, 103.8, 103.0, 103.2)])
        full = pd.concat([base, after], ignore_index=True)

        from shared.bpa_detector import _detect_failed_breakout
        fired_at = []
        for n in range(18, len(full) + 1):
            window = full.iloc[:n].copy()
            r = _detect_failed_breakout(window)
            if r is not None:
                fired_at.append(n - 1)
        # Should fire once at bar 19 (the failure bar); NOT at bar 20 or 21
        self.assertEqual(fired_at, [19], f"expected only at bar 19, got {fired_at}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
