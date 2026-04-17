"""Tests for aiedge.analysis.correlation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.analysis.correlation import (
    cluster_by_threshold, correlation_matrix, dedup_correlated, log_returns,
)


def mk_closes(n: int = 30) -> pd.DataFrame:
    """Three ticker prices: two highly correlated, one independent."""
    np.random.seed(0)
    common = np.cumsum(np.random.normal(0, 1.0, n))
    independent = np.cumsum(np.random.normal(0, 1.0, n))
    return pd.DataFrame({
        "QQQ":  100 + common,
        "TQQQ": 100 + common * 3 + np.random.normal(0, 0.1, n),  # highly corr
        "XLE":  100 + independent,
    })


class LogReturnsTests(unittest.TestCase):

    def test_log_returns_drops_first_row(self):
        closes = mk_closes(20)
        rets = log_returns(closes)
        self.assertEqual(len(rets), 19)

    def test_log_returns_handles_zero(self):
        closes = pd.DataFrame({"A": [100, 0, 101, 102, 103]})
        rets = log_returns(closes)
        # Zero becomes NaN, not an error
        self.assertTrue(rets.isna().any().any() or len(rets) < 4)


class CorrelationMatrixTests(unittest.TestCase):

    def test_diagonal_is_one(self):
        rets = log_returns(mk_closes(30))
        corr = correlation_matrix(rets)
        for t in corr.columns:
            self.assertAlmostEqual(corr.loc[t, t], 1.0, places=5)

    def test_high_corr_detected(self):
        rets = log_returns(mk_closes(50))
        corr = correlation_matrix(rets)
        self.assertGreater(corr.loc["QQQ", "TQQQ"], 0.9)


class ClusterByThresholdTests(unittest.TestCase):

    def test_separates_correlated_from_independent(self):
        rets = log_returns(mk_closes(60))
        corr = correlation_matrix(rets)
        clusters = cluster_by_threshold(corr, threshold=0.8)
        # QQQ + TQQQ should cluster together, XLE alone
        self.assertEqual(len(clusters), 2)
        # Find the cluster containing QQQ
        for cluster in clusters:
            if "QQQ" in cluster:
                self.assertIn("TQQQ", cluster)

    def test_high_threshold_isolates_everyone(self):
        rets = log_returns(mk_closes(60))
        corr = correlation_matrix(rets)
        clusters = cluster_by_threshold(corr, threshold=0.99999)
        self.assertEqual(len(clusters), 3)


class DedupCorrelatedTests(unittest.TestCase):

    def test_keeps_highest_ranked_per_cluster(self):
        rets = log_returns(mk_closes(60))
        corr = correlation_matrix(rets)
        candidates = [
            {"ticker": "QQQ",  "urgency": 8.0},
            {"ticker": "TQQQ", "urgency": 7.5},
            {"ticker": "XLE",  "urgency": 6.0},
        ]
        out = dedup_correlated(candidates, corr, threshold=0.8)
        tickers = [c["ticker"] for c in out]
        self.assertIn("QQQ", tickers)
        self.assertNotIn("TQQQ", tickers)
        self.assertIn("XLE", tickers)

    def test_unknown_ticker_passes_through(self):
        rets = log_returns(mk_closes(30))
        corr = correlation_matrix(rets)
        candidates = [
            {"ticker": "AAPL", "urgency": 9.0},   # not in corr matrix
            {"ticker": "QQQ",  "urgency": 7.0},
        ]
        out = dedup_correlated(candidates, corr)
        self.assertEqual(len(out), 2)

    def test_empty_candidates(self):
        corr = pd.DataFrame()
        self.assertEqual(dedup_correlated([], corr), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
