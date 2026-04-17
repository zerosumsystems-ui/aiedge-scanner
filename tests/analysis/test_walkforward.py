"""Tests for aiedge.analysis.walkforward."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.analysis.walkforward import by_date, expanding_window, rolling_window


class RollingWindowTests(unittest.TestCase):

    def test_basic_split(self):
        splits = list(rolling_window(n=100, train_size=50, test_size=10))
        # Step defaults to test_size=10; so 50 + 10, 60 + 10, 70 + 10, 80 + 10, 90 + 10
        self.assertEqual(len(splits), 5)

    def test_disjoint_train_test(self):
        for train_idx, test_idx in rolling_window(100, 30, 10):
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

    def test_fixed_sizes(self):
        for train_idx, test_idx in rolling_window(100, 30, 10):
            self.assertEqual(len(train_idx), 30)
            self.assertEqual(len(test_idx), 10)

    def test_custom_step(self):
        # step=5 means 6 splits from n=100, train=50, test=10: starts 0,5,10,15,20,25,30,35,40
        splits = list(rolling_window(100, 50, 10, step=5))
        self.assertGreater(len(splits), 5)

    def test_too_large_raises(self):
        with self.assertRaises(ValueError):
            list(rolling_window(10, 20, 5))

    def test_negative_train_raises(self):
        with self.assertRaises(ValueError):
            list(rolling_window(10, -1, 5))


class ExpandingWindowTests(unittest.TestCase):

    def test_train_grows(self):
        splits = list(expanding_window(n=100, min_train=30, test_size=10))
        prev_train_end = -1
        for train_idx, _ in splits:
            self.assertGreater(len(train_idx), prev_train_end)
            prev_train_end = len(train_idx)

    def test_train_starts_at_zero(self):
        for train_idx, _ in expanding_window(100, 30, 10):
            self.assertEqual(train_idx[0], 0)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            list(expanding_window(10, 0, 5))


class ByDateTests(unittest.TestCase):

    def test_basic_split_by_day(self):
        # 30 unique dates, 2 rows per date = 60 rows
        dates = pd.to_datetime(
            [f"2026-01-{i:02d}" for i in range(1, 31)] * 2
        )
        splits = list(by_date(dates, train_days=5, test_days=2))
        # First split: train days 1-5, test days 6-7
        self.assertGreater(len(splits), 0)

    def test_too_few_days_returns_nothing(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        splits = list(by_date(dates, train_days=5, test_days=2))
        self.assertEqual(splits, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
