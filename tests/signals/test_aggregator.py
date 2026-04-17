"""Unit tests for aiedge.signals.aggregator."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals.aggregator import (
    FAILED_GAP_MIN_FRAC_ADR,
    SIGNAL_BUY_INTRADAY,
    SIGNAL_SELL_INTRADAY,
    UNCERTAINTY_HIGH,
    UNCERTAINTY_TRAP,
    URGENCY_HIGH,
    _detect_phase,
    _determine_signal,
)


def mk_df(n: int = 15) -> pd.DataFrame:
    opens = [100 + i for i in range(n)]
    closes = [100 + i + 1 for i in range(n)]
    highs = [c + 0.2 for c in closes]
    lows = [o - 0.2 for o in opens]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


class DetectPhaseTests(unittest.TestCase):

    def test_spike_phase_when_short_df(self):
        df = mk_df(5)
        self.assertEqual(_detect_phase(df, spike_bars=4, uncertainty=0.0,
                                        gap_direction="up", gap_held=True), "spike")

    def test_failed_gap_requires_adr_threshold(self):
        df = mk_df(15)
        # gap_held=False + high uncertainty + adr frac > threshold → failed_gap
        self.assertEqual(_detect_phase(df, spike_bars=0, uncertainty=UNCERTAINTY_HIGH + 1,
                                        gap_direction="up", gap_held=False,
                                        abs_gap_frac_adr=FAILED_GAP_MIN_FRAC_ADR + 0.01),
                         "failed_gap")

    def test_failed_gap_suppressed_below_adr_threshold(self):
        df = mk_df(15)
        # Below ADR threshold → falls through to trading_range (high uncertainty)
        result = _detect_phase(df, spike_bars=0, uncertainty=UNCERTAINTY_HIGH + 1,
                                gap_direction="up", gap_held=False,
                                abs_gap_frac_adr=0.01)
        self.assertNotEqual(result, "failed_gap")

    def test_trading_range_when_high_uncertainty(self):
        df = mk_df(15)
        self.assertEqual(_detect_phase(df, spike_bars=0, uncertainty=UNCERTAINTY_HIGH + 1,
                                        gap_direction="up", gap_held=True), "trading_range")

    def test_channel_when_spike_present(self):
        df = mk_df(15)
        result = _detect_phase(df, spike_bars=4, uncertainty=0.0,
                                gap_direction="up", gap_held=True)
        self.assertEqual(result, "channel")

    def test_trading_range_fallback(self):
        df = mk_df(15)
        result = _detect_phase(df, spike_bars=0, uncertainty=0.0,
                                gap_direction="up", gap_held=True)
        self.assertEqual(result, "trading_range")


class DetermineSignalTests(unittest.TestCase):

    def test_failed_gap_returns_avoid(self):
        signal = _determine_signal(urgency=8.0, uncertainty=2.0, gap_held=False,
                                    gap_direction="up", rr_ratio=2.0,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "AVOID")

    def test_trap_returns_avoid(self):
        signal = _determine_signal(urgency=URGENCY_HIGH, uncertainty=UNCERTAINTY_TRAP,
                                    gap_held=True, gap_direction="up", rr_ratio=2.0,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "AVOID")

    def test_strong_bull_produces_buy_pullback(self):
        signal = _determine_signal(urgency=8.0, uncertainty=1.0, gap_held=True,
                                    gap_direction="up", rr_ratio=2.5,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "BUY_PULLBACK")

    def test_strong_bull_no_pullback_produces_buy_spike(self):
        signal = _determine_signal(urgency=8.0, uncertainty=1.0, gap_held=True,
                                    gap_direction="up", rr_ratio=2.5,
                                    spike_bars=4, pullback_exists=False)
        self.assertEqual(signal, "BUY_SPIKE")

    def test_bear_mirror_produces_sell_pullback(self):
        signal = _determine_signal(urgency=8.0, uncertainty=1.0, gap_held=True,
                                    gap_direction="down", rr_ratio=2.5,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "SELL_PULLBACK")

    def test_low_rr_downgrades_buy_to_wait(self):
        signal = _determine_signal(urgency=8.0, uncertainty=1.0, gap_held=True,
                                    gap_direction="up", rr_ratio=0.5,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "WAIT")

    def test_high_rr_upgrades_wait_to_buy(self):
        # urgency=6, uncertainty=4 → "WAIT" baseline; rr=2.5 should upgrade
        signal = _determine_signal(urgency=6.0, uncertainty=4.0, gap_held=True,
                                    gap_direction="up", rr_ratio=2.5,
                                    spike_bars=4, pullback_exists=True)
        self.assertEqual(signal, "BUY_PULLBACK")

    def test_bpa_intraday_bear_flip_on_gap_up(self):
        signal = _determine_signal(urgency=6.0, uncertainty=3.0, gap_held=True,
                                    gap_direction="up", rr_ratio=1.5,
                                    spike_bars=4, pullback_exists=True,
                                    bpa_alignment=-1.5, always_in="short")
        self.assertEqual(signal, SIGNAL_SELL_INTRADAY)

    def test_bpa_intraday_bull_flip_on_gap_down(self):
        signal = _determine_signal(urgency=6.0, uncertainty=3.0, gap_held=True,
                                    gap_direction="down", rr_ratio=1.5,
                                    spike_bars=4, pullback_exists=True,
                                    bpa_alignment=1.8, always_in="long")
        self.assertEqual(signal, SIGNAL_BUY_INTRADAY)

    def test_failed_gap_recovered_with_bpa_returns_wait_not_avoid(self):
        signal = _determine_signal(urgency=6.0, uncertainty=3.0, gap_held=False,
                                    gap_direction="up", rr_ratio=1.5,
                                    spike_bars=4, pullback_exists=True,
                                    bpa_alignment=1.5,
                                    gap_fill_status="filled_recovered")
        # With BPA confirmation on recovered gap, shouldn't AVOID
        self.assertNotEqual(signal, "AVOID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
