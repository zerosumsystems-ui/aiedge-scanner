#!/usr/bin/env python3
"""
Pattern Lab API — JSON output for the tradescope dashboard.

Usage:
  python3 pattern_lab_api.py --stats      # full stats payload
  python3 pattern_lab_api.py --recent 20  # last N detections
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.pattern_lab import (
    win_rate_by_setup,
    win_rate_by_context,
    win_rate_by_time_of_day,
    recent_detections,
    detection_count,
    _connect,
)


def _summary() -> dict:
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        dates = conn.execute(
            "SELECT COUNT(DISTINCT detection_date) FROM detections"
        ).fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(detection_date), MAX(detection_date) FROM detections"
        ).fetchone()
        return {
            "totalDetections": total,
            "datesTracked": dates,
            "dateRange": {
                "from": date_range[0] or "",
                "to": date_range[1] or "",
            },
        }
    finally:
        conn.close()


def full_stats() -> dict:
    return {
        "summary": _summary(),
        "bySetup": win_rate_by_setup(),
        "byContext": {
            "cycle_phase": win_rate_by_context("cycle_phase"),
            "always_in": win_rate_by_context("always_in"),
            "day_type": win_rate_by_context("day_type"),
        },
        "byTimeOfDay": win_rate_by_time_of_day(bucket_size=6),
        "recentDetections": [
            {
                "ticker": d["ticker"],
                "setupType": d["setup_type"],
                "direction": d["direction"],
                "detectedAt": d["detected_at"],
                "confidence": d["confidence"],
                "result": d["result"],
                "mfe": d["mfe"],
                "mae": d["mae"],
                "cyclePhase": d["cycle_phase"],
                "signal": d["signal"],
                "urgency": d["urgency"],
            }
            for d in recent_detections(limit=50)
        ],
    }


def push_to_dashboard(url: str = "http://localhost:3000/api/patterns") -> None:
    """POST current stats to the tradescope dashboard API."""
    import urllib.request

    payload = json.dumps(full_stats(), default=str).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"POST {url} → {resp.status}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Print full stats JSON")
    parser.add_argument("--recent", type=int, default=0, help="Print last N detections")
    parser.add_argument("--push", metavar="URL", nargs="?", const="http://localhost:3000/api/patterns",
                        help="POST stats to dashboard (default: localhost:3000)")
    args = parser.parse_args()

    if args.push:
        push_to_dashboard(args.push)
    elif args.stats:
        print(json.dumps(full_stats(), default=str))
    elif args.recent > 0:
        rows = recent_detections(limit=args.recent)
        print(json.dumps(rows, default=str))
    else:
        parser.print_help()
