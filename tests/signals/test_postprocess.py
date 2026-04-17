"""Unit tests for aiedge.signals.postprocess."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals.postprocess import (
    ETF_FAMILIES,
    SAME_COMPANY,
    _TICKER_TO_FAMILY,
    _compute_movement,
    _dedup_etf_families,
    _fmt_delta,
    _fmt_movement,
    annotate_adr_multiple,
)


class AnnotateAdrMultipleTests(unittest.TestCase):

    def test_writes_three_keys(self):
        df = pd.DataFrame({"high": [101.0, 102.0], "low": [99.0, 100.0]})
        score = {}
        annotate_adr_multiple(score, df, "AAPL", {"AAPL": 3.0})
        self.assertEqual(set(score.keys()),
                         {"today_range", "adr_20", "adr_multiple"})

    def test_today_range_is_high_minus_low(self):
        df = pd.DataFrame({"high": [105.0, 110.0], "low": [100.0, 102.0]})
        score = {}
        annotate_adr_multiple(score, df, "AAPL", {"AAPL": 5.0})
        self.assertEqual(score["today_range"], 10.0)  # 110 - 100

    def test_adr_multiple_is_range_over_adr(self):
        df = pd.DataFrame({"high": [110.0], "low": [100.0]})
        score = {}
        annotate_adr_multiple(score, df, "AAPL", {"AAPL": 5.0})
        self.assertEqual(score["adr_multiple"], 2.0)  # 10 / 5

    def test_zero_adr_avoids_divzero(self):
        df = pd.DataFrame({"high": [110.0], "low": [100.0]})
        score = {}
        annotate_adr_multiple(score, df, "AAPL", {})
        self.assertEqual(score["adr_multiple"], 0.0)


class DedupEtfFamiliesTests(unittest.TestCase):

    def test_non_etf_tickers_pass_through(self):
        results = [
            {"ticker": "AAPL", "urgency": 8.0},
            {"ticker": "MSFT", "urgency": 7.0},
        ]
        out = _dedup_etf_families(results)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["ticker"], "AAPL")

    def test_family_leader_kept_siblings_dropped(self):
        results = [
            {"ticker": "QQQ", "urgency": 8.0},
            {"ticker": "TQQQ", "urgency": 7.5},
            {"ticker": "SQQQ", "urgency": 7.0},
        ]
        out = _dedup_etf_families(results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ticker"], "QQQ")
        self.assertEqual(out[0]["family"], "NDX")
        self.assertTrue(out[0]["family_leader"])
        self.assertEqual(out[0]["family_sibling_count"], 2)
        self.assertIn("TQQQ", out[0]["family_siblings"])

    def test_google_siblings_collapsed(self):
        results = [
            {"ticker": "GOOG", "urgency": 6.5},
            {"ticker": "GOOGL", "urgency": 6.0},
        ]
        out = _dedup_etf_families(results)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["family"], "GOOG_FAMILY")

    def test_mixed_etf_and_equity(self):
        results = [
            {"ticker": "QQQ", "urgency": 8.0},
            {"ticker": "AAPL", "urgency": 7.5},
            {"ticker": "TQQQ", "urgency": 7.0},
        ]
        out = _dedup_etf_families(results)
        self.assertEqual([r["ticker"] for r in out], ["QQQ", "AAPL"])


class ComputeMovementTests(unittest.TestCase):

    def test_first_scan_all_have_dash(self):
        results = [
            {"ticker": "AAPL", "urgency": 8.0},
            {"ticker": "MSFT", "urgency": 7.0},
        ]
        out = _compute_movement(results, prior={})
        self.assertEqual(out[0]["rank"], 1)
        self.assertTrue(out[0]["_first_scan"])
        self.assertIsNone(out[0]["prev_rank"])

    def test_unchanged_rank(self):
        prior = {"AAPL": {"rank": 1, "urgency": 8.0}}
        results = [{"ticker": "AAPL", "urgency": 8.0}]
        out = _compute_movement(results, prior)
        self.assertEqual(out[0]["rank_change"], 0)
        self.assertEqual(out[0]["urgency_delta"], 0.0)

    def test_moved_up_in_rank(self):
        # Previously rank 3, now rank 1 → rank_change = +2
        prior = {"AAPL": {"rank": 3, "urgency": 5.0}}
        results = [{"ticker": "AAPL", "urgency": 7.0}]
        out = _compute_movement(results, prior)
        self.assertEqual(out[0]["rank_change"], 2)
        self.assertEqual(out[0]["urgency_delta"], 2.0)

    def test_new_ticker_not_in_prior(self):
        prior = {"AAPL": {"rank": 1, "urgency": 8.0}}
        results = [{"ticker": "MSFT", "urgency": 7.0}]
        out = _compute_movement(results, prior)
        self.assertIsNone(out[0]["prev_rank"])
        self.assertIsNone(out[0]["rank_change"])


class FmtMovementTests(unittest.TestCase):

    def test_first_scan_returns_dash(self):
        self.assertEqual(_fmt_movement({"_first_scan": True}), "—")

    def test_new_ticker(self):
        self.assertEqual(_fmt_movement({"_first_scan": False, "rank_change": None}), "NEW")

    def test_unchanged_ranking(self):
        self.assertEqual(
            _fmt_movement({"_first_scan": False, "rank_change": 0, "prev_rank": 3}),
            "was #3  (=)",
        )

    def test_moved_up(self):
        self.assertIn("+2",
                      _fmt_movement({"_first_scan": False, "rank_change": 2, "prev_rank": 3}))

    def test_moved_down(self):
        self.assertIn("-2",
                      _fmt_movement({"_first_scan": False, "rank_change": -2, "prev_rank": 3}))


class FmtDeltaTests(unittest.TestCase):

    def test_first_scan_dash(self):
        self.assertEqual(_fmt_delta({"_first_scan": True}), "—")

    def test_none_delta_dash(self):
        self.assertEqual(_fmt_delta({"_first_scan": False, "urgency_delta": None}), "—")

    def test_positive_delta_has_plus_sign(self):
        self.assertEqual(
            _fmt_delta({"_first_scan": False, "urgency_delta": 1.5}),
            "U +1.5",
        )

    def test_negative_delta(self):
        self.assertEqual(
            _fmt_delta({"_first_scan": False, "urgency_delta": -2.0}),
            "U -2.0",
        )


class EtfFamilyConstantsTests(unittest.TestCase):

    def test_same_company_merged_into_etf_families(self):
        for fam in SAME_COMPANY:
            self.assertIn(fam, ETF_FAMILIES)

    def test_ticker_to_family_has_goog(self):
        self.assertEqual(_TICKER_TO_FAMILY.get("GOOG"), "GOOG_FAMILY")
        self.assertEqual(_TICKER_TO_FAMILY.get("GOOGL"), "GOOG_FAMILY")

    def test_ticker_to_family_has_qqq(self):
        self.assertEqual(_TICKER_TO_FAMILY.get("QQQ"), "NDX")


if __name__ == "__main__":
    unittest.main(verbosity=2)
