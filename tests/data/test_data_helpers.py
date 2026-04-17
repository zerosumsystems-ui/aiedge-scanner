"""Unit tests for aiedge.data.{normalize,resample,universe}."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.data.normalize import _normalize_databento_df
from aiedge.data.resample import (
    ET,
    SCAN_BAR_SCHEMA,
    SCAN_RESAMPLE,
    _resample_to_5min,
    resample_to_5min,
)
from aiedge.data.universe import _get_default_universe


class NormalizeDatabentoTests(unittest.TestCase):

    def test_lowercases_columns(self):
        df = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1000],
        }, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
        out = _normalize_databento_df(df)
        self.assertIn("open", out.columns)
        self.assertNotIn("Open", out.columns)

    def test_sets_datetime_index_from_ts_event(self):
        df = pd.DataFrame({
            "ts_event": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "open": [1.0, 2.0, 3.0],
        })
        out = _normalize_databento_df(df)
        self.assertIsInstance(out.index, pd.DatetimeIndex)

    def test_converts_naive_index_to_utc(self):
        idx = pd.date_range("2024-01-01", periods=3)  # naive
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx)
        out = _normalize_databento_df(df)
        self.assertEqual(str(out.index.tz), "UTC")

    def test_converts_other_tz_to_utc(self):
        idx = pd.date_range("2024-01-01", periods=3, tz="US/Eastern")
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=idx)
        out = _normalize_databento_df(df)
        self.assertEqual(str(out.index.tz), "UTC")


class ResampleTests(unittest.TestCase):

    def test_five_one_minute_bars_produce_one_five_minute_bar(self):
        idx = pd.date_range("2024-01-01 09:30", periods=5, freq="1min", tz="UTC")
        df1m = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low":  [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [1000] * 5,
        }, index=idx)
        df5m = _resample_to_5min(df1m)
        self.assertEqual(len(df5m), 1)
        self.assertEqual(df5m.iloc[0]["open"], 100)
        self.assertEqual(df5m.iloc[0]["high"], 105)
        self.assertEqual(df5m.iloc[0]["low"], 99)
        self.assertEqual(df5m.iloc[0]["close"], 105)
        self.assertEqual(df5m.iloc[0]["volume"], 5000)

    def test_handles_missing_volume_column(self):
        idx = pd.date_range("2024-01-01 09:30", periods=5, freq="1min", tz="UTC")
        df1m = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low":  [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
        }, index=idx)
        df5m = _resample_to_5min(df1m)
        self.assertEqual(len(df5m), 1)
        self.assertNotIn("volume", df5m.columns)


class LiveResampleTests(unittest.TestCase):
    """Cover the live-scanner variant: datetime column, ET localization,
    partial-bar drop."""

    def _bars(self, n: int, start: str = "2026-04-17 09:30") -> pd.DataFrame:
        return pd.DataFrame({
            "datetime": pd.date_range(start, periods=n, freq="1min"),
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.1] * n,
            "volume": [1000] * n,
        })

    def test_five_bars_make_one_closed_window(self):
        from datetime import datetime
        df = self._bars(5, "2026-04-17 09:30")
        out = resample_to_5min(df, now=ET.localize(datetime(2026, 4, 17, 10, 0)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["volume"], 5000)

    def test_drops_half_open_last_bar(self):
        from datetime import datetime
        # 7 bars @ 09:30 → closed [09:30,09:35) + open [09:35,09:40)
        df = self._bars(7, "2026-04-17 09:30")
        out = resample_to_5min(df, now=ET.localize(datetime(2026, 4, 17, 9, 37)))
        # 09:35 window is still forming → dropped; only 09:30 survives
        self.assertEqual(len(out), 1)

    def test_keeps_all_bars_when_now_past_last_window(self):
        from datetime import datetime
        df = self._bars(7, "2026-04-17 09:30")
        out = resample_to_5min(df, now=datetime(2026, 4, 17, 9, 45, tzinfo=ET))
        self.assertEqual(len(out), 2)

    def test_localizes_naive_datetime_to_et(self):
        from datetime import datetime
        df = self._bars(5)  # naive datetimes
        out = resample_to_5min(df, now=ET.localize(datetime(2026, 4, 17, 10, 0)))
        # Output datetime col should be tz-aware ET
        first = out.iloc[0]["datetime"]
        self.assertIsNotNone(first.tz)

    def test_forward_fills_gap_bars(self):
        # Bars at 09:30, 09:34 only — 09:31/32/33 missing from this 5-min block
        from datetime import datetime
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2026-04-17 09:30", "2026-04-17 09:34"]),
            "open": [100.0, 101.0],
            "high": [100.5, 101.5],
            "low": [99.5, 100.5],
            "close": [100.1, 101.1],
            "volume": [1000, 2000],
        })
        out = resample_to_5min(df, now=ET.localize(datetime(2026, 4, 17, 10, 0)))
        # Single 5-min bar covering [09:30, 09:35)
        self.assertEqual(len(out), 1)


class SchemaConstantsTests(unittest.TestCase):

    def test_scan_bar_schema(self):
        self.assertEqual(SCAN_BAR_SCHEMA, "ohlcv-1m")

    def test_scan_resample(self):
        self.assertEqual(SCAN_RESAMPLE, "5min")


class UniverseTests(unittest.TestCase):

    def test_returns_non_empty_list(self):
        universe = _get_default_universe()
        self.assertIsInstance(universe, list)
        self.assertGreater(len(universe), 0)

    def test_returns_ticker_strings(self):
        universe = _get_default_universe()
        for ticker in universe[:10]:
            self.assertIsInstance(ticker, str)
            self.assertGreater(len(ticker), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
