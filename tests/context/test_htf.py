"""Tests for aiedge.context.htf."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.context.htf import classify_htf_alignment


class HtfAlignmentTests(unittest.TestCase):

    def test_both_up_with_long_is_aligned(self):
        daily = [100 + i for i in range(30)]    # trending up
        weekly = [100 + i * 5 for i in range(15)]
        out = classify_htf_alignment(daily, weekly, "long")
        self.assertEqual(out["alignment"], "aligned")
        self.assertEqual(out["daily_bias"], "up")

    def test_both_down_with_short_is_aligned(self):
        daily = [200 - i for i in range(30)]
        weekly = [200 - i * 5 for i in range(15)]
        out = classify_htf_alignment(daily, weekly, "short")
        self.assertEqual(out["alignment"], "aligned")

    def test_both_up_with_short_is_opposed(self):
        daily = [100 + i for i in range(30)]
        weekly = [100 + i * 5 for i in range(15)]
        out = classify_htf_alignment(daily, weekly, "short")
        self.assertEqual(out["alignment"], "opposed")

    def test_no_data_when_insufficient_history(self):
        out = classify_htf_alignment([100, 101], [100, 101], "long")
        self.assertEqual(out["alignment"], "no_data")

    def test_returns_expected_keys(self):
        daily = [100 + i for i in range(30)]
        weekly = [100 + i for i in range(15)]
        out = classify_htf_alignment(daily, weekly, "long")
        for key in ("daily_bias", "weekly_bias", "setup_direction", "alignment"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
