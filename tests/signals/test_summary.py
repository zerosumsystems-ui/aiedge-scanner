"""Unit tests for aiedge.signals.summary."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals.aggregator import SIGNAL_BUY_INTRADAY, SIGNAL_SELL_INTRADAY
from aiedge.signals.summary import _generate_summary


class GenerateSummaryTests(unittest.TestCase):

    def _common(self):
        return dict(urgency=7.0, uncertainty=2.0, phase="channel",
                    always_in="long", gap_direction="up",
                    spike_bars=4, pullback_depth_pct=0.25)

    def test_buy_pullback_summary_contains_direction(self):
        s = _generate_summary(signal="BUY_PULLBACK", **self._common())
        self.assertIn("bull", s)

    def test_sell_pullback_summary_contains_bear(self):
        args = self._common()
        args["gap_direction"] = "down"
        s = _generate_summary(signal="SELL_PULLBACK", **args)
        self.assertIn("bear", s)

    def test_intraday_sell_flip_returns_distinct_message(self):
        s = _generate_summary(signal=SIGNAL_SELL_INTRADAY, **self._common())
        self.assertIn("bear flip", s)

    def test_intraday_buy_flip_returns_distinct_message(self):
        args = self._common()
        args["gap_direction"] = "down"
        s = _generate_summary(signal=SIGNAL_BUY_INTRADAY, **args)
        self.assertIn("bull flip", s)

    def test_avoid_trap_summary(self):
        args = self._common()
        args["urgency"] = 8.0
        args["uncertainty"] = 8.0
        s = _generate_summary(signal="AVOID", **args)
        self.assertIn("Trap", s)

    def test_avoid_gap_filled_summary(self):
        args = self._common()
        args["gap_held"] = False
        s = _generate_summary(signal="AVOID", **args)
        self.assertIn("Gap filled", s)

    def test_fog_summary(self):
        s = _generate_summary(signal="FOG", **self._common())
        self.assertIn("Can't read", s)

    def test_wait_summary(self):
        s = _generate_summary(signal="WAIT", **self._common())
        self.assertIn("Promising", s)

    def test_pass_summary(self):
        s = _generate_summary(signal="PASS", **self._common())
        self.assertIn("Pass", s)

    def test_unknown_signal_returns_generic_summary(self):
        s = _generate_summary(signal="UNKNOWN", **self._common())
        self.assertIn("Phase: channel", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
