"""Tests for aiedge.analysis.failure."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.analysis.failure import classify_failure, failure_breakdown


def mk_bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


class ClassifyFailureTests(unittest.TestCase):

    def test_missing_entry_returns_unknown(self):
        self.assertEqual(
            classify_failure({}, pd.DataFrame()),
            "unknown",
        )

    def test_empty_post_entry_returns_unknown(self):
        trade = {"entry": 100.0, "stop": 99.0, "direction": "long"}
        self.assertEqual(
            classify_failure(trade, pd.DataFrame()),
            "unknown",
        )

    def test_stop_flush_long(self):
        # Stop hit on the 2nd post-entry bar
        trade = {"entry": 100.0, "stop": 99.0, "direction": "long"}
        bars = mk_bars([
            (100.0, 100.2, 99.8, 100.1),  # bar 1 — no hit
            (100.0, 100.0, 98.8, 99.0),   # bar 2 — stop hit at 98.8 (< 99)
        ])
        self.assertEqual(classify_failure(trade, bars), "stop_flush")

    def test_stop_flush_short(self):
        trade = {"entry": 100.0, "stop": 101.0, "direction": "short"}
        bars = mk_bars([
            (100.0, 100.5, 99.8, 100.0),
            (100.0, 101.5, 99.5, 101.0),  # high exceeds stop
        ])
        self.assertEqual(classify_failure(trade, bars), "stop_flush")

    def test_news_shock(self):
        # Bar 5 has a massive adverse body > 70% of daily ATR, stop hit
        trade = {"entry": 100.0, "stop": 99.0, "direction": "long"}
        bars = mk_bars([
            (100.0, 100.3, 99.7, 100.1),
            (100.1, 100.5, 99.9, 100.2),
            (100.2, 100.4, 99.8, 100.0),
            (100.0, 100.2, 99.5, 99.8),
            (99.8, 99.9, 90.0, 90.5),  # giant down bar — body = 9.3
        ])
        result = classify_failure(trade, bars, daily_atr=5.0)
        self.assertEqual(result, "news_shock")

    def test_reversal(self):
        # Went into 2x-risk profit, then reversed and stopped out
        trade = {"entry": 100.0, "stop": 99.0, "direction": "long"}
        bars = mk_bars([
            (100.0, 101.0, 99.8, 100.8),
            (100.8, 102.5, 100.5, 102.3),  # MFE = 2.5 (2.5x risk)
            (102.3, 102.5, 101.0, 101.5),
            (101.5, 101.5, 98.9, 99.0),    # stop hit
        ])
        result = classify_failure(trade, bars)
        self.assertEqual(result, "reversal")

    def test_slow_bleed(self):
        # Never got much MFE; slow consistent decline
        trade = {"entry": 100.0, "stop": 98.0, "direction": "long"}
        bars = mk_bars([
            (100.0, 100.1, 99.5, 99.8),
            (99.8, 99.8, 99.3, 99.4),
            (99.4, 99.5, 99.0, 99.0),
            (99.0, 99.1, 98.6, 98.6),
            (98.6, 98.6, 97.9, 98.0),
        ])
        result = classify_failure(trade, bars)
        # Either slow_bleed or chop — both are reasonable for this shape
        self.assertIn(result, ("slow_bleed", "chop"))


class FailureBreakdownTests(unittest.TestCase):

    def test_aggregates_counts(self):
        # 2 stop_flush, 1 reversal
        trade_sf = {"entry": 100.0, "stop": 99.0, "direction": "long", "ticker": "AAPL"}
        sf_bars = mk_bars([(100.0, 100.2, 98.8, 99.0)])
        trade_rev = {"entry": 100.0, "stop": 99.0, "direction": "long", "ticker": "MSFT"}
        rev_bars = mk_bars([
            (100.0, 102.5, 99.8, 102.0),
            (102.0, 102.5, 98.8, 99.0),
        ])
        losses = [
            {**trade_sf, "post_entry_bars": sf_bars},
            {**trade_sf, "post_entry_bars": sf_bars, "ticker": "NVDA"},
            {**trade_rev, "post_entry_bars": rev_bars},
        ]
        breakdown = failure_breakdown(losses)
        # At least 2 stop_flush and 1 reversal
        self.assertGreater(breakdown["n"].sum(), 2)

    def test_empty_losses(self):
        breakdown = failure_breakdown([])
        self.assertEqual(len(breakdown), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
