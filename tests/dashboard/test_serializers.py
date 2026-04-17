"""Unit tests for aiedge.dashboard.serializers."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.dashboard.serializers import _serialize_scan_payload


def _base_result(**overrides) -> dict:
    """A minimal result dict matching the shape produced by score_gap."""
    r = {
        "ticker": "FAKE",
        "rank": 1,
        "urgency": 7.5,
        "uncertainty": 2.0,
        "signal": "BUY_INTRADAY",
        "day_type": "trend_from_open",
        "phase": "trend_from_open",
        "daily_atr": 1.23,
        "move_ratio": 1.0,
        "adr_multiple": 1.2,
        "summary": "test",
        "details": {
            "spike_quality": 1.0, "gap_integrity": 1.0, "pullback_quality": 1.0,
            "follow_through": 1.0, "ma_separation": 1.0, "volume_conf": 1.0,
            "tail_quality": 1.0, "small_pullback_trend": 1.0, "bpa_alignment": 1.0,
            "bpa_setups": [],
        },
    }
    r.update(overrides)
    return r


def _serialize_one(result: dict) -> dict:
    payload = _serialize_scan_payload(
        results=[result],
        now_et=datetime(2026, 4, 17, 10, 30),
        total_symbols=1,
        passed=1,
        elapsed=0.1,
        interval_min=5,
        intraday_levels={},
        first_scan_hour=9,
        first_scan_min=35,
    )
    return payload["results"][0]


class BpaActiveSetupsTests(unittest.TestCase):

    def test_empty_list_when_no_setups(self):
        entry = _serialize_one(_base_result())
        self.assertEqual(entry["bpaActiveSetups"], [])

    def test_extracts_type_names_from_details_dicts(self):
        result = _base_result()
        result["details"]["bpa_setups"] = [
            {"type": "H2", "entry": 100.0, "stop": 99.0, "target": 102.0,
             "confidence": 0.8, "bar_index": 12, "entry_mode": "stop"},
            {"type": "spike_channel", "entry": 100.0, "stop": 99.0, "target": 102.0,
             "confidence": 0.7, "bar_index": 11, "entry_mode": "stop"},
        ]
        entry = _serialize_one(result)
        self.assertEqual(entry["bpaActiveSetups"], ["H2", "spike_channel"])

    def test_skips_non_string_types(self):
        result = _base_result()
        result["details"]["bpa_setups"] = [
            {"type": "H1"},
            {"not_type": "garbage"},
            {"type": None},
            {"type": "L2"},
        ]
        entry = _serialize_one(result)
        self.assertEqual(entry["bpaActiveSetups"], ["H1", "L2"])


class PhaseFieldTests(unittest.TestCase):

    def test_uses_top_level_phase_when_present(self):
        entry = _serialize_one(_base_result(phase="trading_range", day_type="trend_from_open"))
        self.assertEqual(entry["phase"], "trading_range")

    def test_falls_back_to_day_type(self):
        r = _base_result()
        r.pop("phase")
        r["day_type"] = "spike and channel"
        entry = _serialize_one(r)
        # spaces normalized to underscores, like dayType
        self.assertEqual(entry["phase"], "spike_and_channel")

    def test_defaults_to_undetermined(self):
        r = _base_result()
        r.pop("phase")
        r["day_type"] = ""
        entry = _serialize_one(r)
        self.assertEqual(entry["phase"], "undetermined")


if __name__ == "__main__":
    unittest.main()
