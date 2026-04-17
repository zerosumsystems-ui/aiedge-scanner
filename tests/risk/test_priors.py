"""Tests for aiedge.risk.priors (fallback lookup)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.risk.priors import (
    DEFAULT_PRIOR_PWIN, PriorLookup, p_win, trader_equation_edge,
)
from aiedge.storage.priors_store import PriorsStore


class PriorsLookupTests(unittest.TestCase):

    def test_no_data_falls_back_to_default(self):
        with PriorsStore(":memory:") as store:
            result = p_win(store, "H2", "mid", "aligned", "tfo", min_samples=10)
            self.assertEqual(result.p_win, DEFAULT_PRIOR_PWIN)
            self.assertEqual(result.matched_level, "default")

    def test_exact_match_used_when_enough_samples(self):
        with PriorsStore(":memory:") as store:
            # 20 wins, 10 losses at the exact key
            for _ in range(20):
                store.record_outcome("H2", "mid", "aligned", "tfo", won=True)
            for _ in range(10):
                store.record_outcome("H2", "mid", "aligned", "tfo", won=False)
            result = p_win(store, "H2", "mid", "aligned", "tfo", min_samples=20)
            self.assertEqual(result.matched_level, "exact")
            self.assertAlmostEqual(result.p_win, 20 / 30, places=4)

    def test_fallback_to_regime_plus_align(self):
        # Exact key has < min_samples; broader (drop day_type) has enough
        with PriorsStore(":memory:") as store:
            store.record_outcome("H2", "mid", "aligned", "tfo", won=True)  # just 1 at exact
            for _ in range(30):
                store.record_outcome("H2", "mid", "aligned", "range", won=True)
            for _ in range(20):
                store.record_outcome("H2", "mid", "aligned", "range", won=False)
            result = p_win(store, "H2", "mid", "aligned", "tfo", min_samples=30)
            self.assertEqual(result.matched_level, "regime+align")

    def test_fallback_cascades_all_the_way_to_default(self):
        with PriorsStore(":memory:") as store:
            store.record_outcome("L2", "high", "opposed", "trending_tr", won=True)
            result = p_win(store, "H2", "mid", "aligned", "tfo", min_samples=30)
            self.assertEqual(result.matched_level, "default")


class TraderEquationEdgeTests(unittest.TestCase):

    def test_positive_edge(self):
        prior = PriorLookup(p_win=0.6, wins=60, losses=40, matched_level="exact")
        edge = trader_equation_edge(prior, reward=2.0, risk=1.0)
        # 0.6 * 2 - 0.4 * 1 = 0.8
        self.assertAlmostEqual(edge, 0.8, places=4)

    def test_negative_edge(self):
        prior = PriorLookup(p_win=0.3, wins=30, losses=70, matched_level="exact")
        edge = trader_equation_edge(prior, reward=1.0, risk=1.0)
        # 0.3 * 1 - 0.7 * 1 = -0.4
        self.assertAlmostEqual(edge, -0.4, places=4)

    def test_zero_risk_returns_zero(self):
        prior = PriorLookup(p_win=0.5, wins=50, losses=50, matched_level="exact")
        self.assertEqual(trader_equation_edge(prior, 1.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
