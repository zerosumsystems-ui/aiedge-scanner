"""Tests for aiedge.storage.priors_store."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.storage.priors_store import PriorsStore


class PriorsStoreTests(unittest.TestCase):

    def test_empty_lookup_returns_zero(self):
        with PriorsStore(":memory:") as store:
            wins, losses = store.get("H2", "mid", "aligned", "trend_from_open")
            self.assertEqual((wins, losses), (0, 0))

    def test_record_win_increments_wins(self):
        with PriorsStore(":memory:") as store:
            store.record_outcome("H2", "mid", "aligned", "trend_from_open", won=True)
            wins, losses = store.get("H2", "mid", "aligned", "trend_from_open")
            self.assertEqual((wins, losses), (1, 0))

    def test_record_multiple_outcomes(self):
        with PriorsStore(":memory:") as store:
            for _ in range(5):
                store.record_outcome("H2", "mid", "aligned", "trend_from_open", won=True)
            for _ in range(3):
                store.record_outcome("H2", "mid", "aligned", "trend_from_open", won=False)
            wins, losses = store.get("H2", "mid", "aligned", "trend_from_open")
            self.assertEqual((wins, losses), (5, 3))

    def test_different_keys_kept_separate(self):
        with PriorsStore(":memory:") as store:
            store.record_outcome("H2", "mid", "aligned", "trend_from_open", won=True)
            store.record_outcome("L2", "mid", "aligned", "trend_from_open", won=True)
            self.assertEqual(store.get("H2", "mid", "aligned", "trend_from_open"), (1, 0))
            self.assertEqual(store.get("L2", "mid", "aligned", "trend_from_open"), (1, 0))

    def test_all_by_setup(self):
        with PriorsStore(":memory:") as store:
            store.record_outcome("H2", "mid", "aligned", "tfo", won=True)
            store.record_outcome("H2", "high", "aligned", "tfo", won=True)
            store.record_outcome("H2", "high", "aligned", "tfo", won=False)
            rows = store.all_by_setup("H2")
            self.assertEqual(len(rows), 2)
            # Row with 2 observations sorted before row with 1
            self.assertEqual(rows[0]["regime"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
