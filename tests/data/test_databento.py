"""Unit tests for aiedge.data.databento.

The external-facing fetchers all hit the Databento Historical REST API
and cannot be exercised offline. These tests cover the pure utilities
(_prev_trading_days, with_timeout) and the module-level constants —
enough to catch breakage from the refactor without standing up network
mocks.
"""

import signal
import sys
import time
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.data.databento import (
    DATASET,
    ET,
    SCHEMA,
    _prev_trading_days,
    with_timeout,
)


class PrevTradingDaysTests(unittest.TestCase):

    def test_returns_a_date(self):
        self.assertIsInstance(_prev_trading_days(1), date)

    def test_skips_weekends(self):
        # N trading days ago should land on Mon-Fri
        for n in (1, 3, 5, 20):
            self.assertLess(_prev_trading_days(n).weekday(), 5)

    def test_is_strictly_in_the_past(self):
        self.assertLess(_prev_trading_days(1), date.today())

    def test_more_days_are_further_back(self):
        self.assertLessEqual(_prev_trading_days(5), _prev_trading_days(1))


class WithTimeoutTests(unittest.TestCase):

    def test_fast_function_completes(self):
        @with_timeout(2)
        def fast():
            return 42
        self.assertEqual(fast(), 42)

    def test_slow_function_raises_timeout(self):
        @with_timeout(1)
        def slow():
            time.sleep(3)

        with self.assertRaises(TimeoutError):
            slow()

    def test_alarm_is_cleared_after_exception(self):
        @with_timeout(1)
        def boom():
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            boom()
        # If the alarm wasn't cleared, this assert would be interrupted
        self.assertEqual(signal.alarm(0), 0)


class ConstantsTests(unittest.TestCase):

    def test_dataset(self):
        self.assertEqual(DATASET, "EQUS.MINI")

    def test_schema_matches_scan_schema(self):
        self.assertEqual(SCHEMA, "ohlcv-1m")

    def test_et_timezone(self):
        self.assertEqual(str(ET), "America/New_York")


if __name__ == "__main__":
    unittest.main(verbosity=2)
