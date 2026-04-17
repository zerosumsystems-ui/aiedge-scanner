#!/usr/bin/env python3
"""
Pattern Lab API — JSON output for the aiedge.trade dashboard.

Usage:
  python3 pattern_lab_api.py --stats                  # live-only stats payload
  python3 pattern_lab_api.py --recent 20              # last N live detections
  python3 pattern_lab_api.py --push [URL]             # POST live stats to /api/patterns
  python3 pattern_lab_api.py --run-stats <run_id>     # stats scoped to a backtest run
  python3 pattern_lab_api.py --push-run <run_id> [URL]# POST backtest-run stats to /api/patterns/run/<id>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.pattern_lab import (
    LIVE_ONLY,
    _connect,
    detection_count,
    detections_by_day,
    get_backtest_run,
    list_backtest_runs,
    purge_polluted_detections,
    recent_detections,
    win_rate_by_context,
    win_rate_by_setup,
    win_rate_by_time_of_day,
)


# ── Shared builders ───────────────────────────────────────────────────────────

def _summary(run_id=None) -> dict:
    """Totals scoped by run_id (LIVE_ONLY, specific id, or None=all)."""
    conn = _connect()
    try:
        clauses: list[str] = []
        params: list = []
        if run_id is LIVE_ONLY:
            clauses.append("run_id IS NULL")
        elif run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        total = conn.execute(
            f"SELECT COUNT(*) FROM detections {where}", params
        ).fetchone()[0]
        dates = conn.execute(
            f"SELECT COUNT(DISTINCT detection_date) FROM detections {where}", params
        ).fetchone()[0]
        date_range = conn.execute(
            f"SELECT MIN(detection_date), MAX(detection_date) FROM detections {where}",
            params,
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


def _detection_payload(d: dict) -> dict:
    """Shape a DB row into the RecentDetection type the site consumes."""
    chart = None
    raw = d.get("chart_json")
    if raw:
        try:
            chart = json.loads(raw)
        except (TypeError, ValueError):
            chart = None
    return {
        "id": d.get("id"),
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
        # entry_mode column added 2026-04-17 alongside bpa_detector rewrite;
        # pre-migration rows return None.
        "entryMode": d["entry_mode"] if "entry_mode" in d.keys() else None,
        "chart": chart,
    }


def _build_stats(run_id, recent_limit: int = 50) -> dict:
    """Core stats dict — reused for live and per-run payloads."""
    return {
        "summary": _summary(run_id=run_id),
        "bySetup": win_rate_by_setup(run_id=run_id),
        "byContext": {
            "cycle_phase": win_rate_by_context("cycle_phase", run_id=run_id),
            "always_in": win_rate_by_context("always_in", run_id=run_id),
            "day_type": win_rate_by_context("day_type", run_id=run_id),
        },
        "byTimeOfDay": win_rate_by_time_of_day(bucket_size=6, run_id=run_id),
        "recentDetections": [
            _detection_payload(d)
            for d in recent_detections(limit=recent_limit, run_id=run_id)
        ],
    }


# ── Public payloads ───────────────────────────────────────────────────────────

def full_stats() -> dict:
    """Live-only stats (excludes all backtest runs)."""
    return _build_stats(run_id=LIVE_ONLY)


def full_stats_for_run(run_id: str) -> dict:
    """Stats scoped to a single backtest run, with metadata + per-day breakdown."""
    run_meta = get_backtest_run(run_id)
    return {
        "run": run_meta,
        **_build_stats(run_id=run_id, recent_limit=50),
        "byDay": detections_by_day(run_id=run_id),
    }


# ── Push to site ──────────────────────────────────────────────────────────────

def push_to_dashboard(url: str = "http://localhost:3000/api/patterns") -> None:
    """POST live stats to /api/patterns."""
    _post_json(url, full_stats())


def push_run_to_dashboard(
    run_id: str,
    base_url: str = "http://localhost:3000",
) -> str:
    """POST backtest-run stats to /api/patterns/run/<run_id>. Returns URL."""
    url = f"{base_url.rstrip('/')}/api/patterns/run/{run_id}"
    _post_json(url, full_stats_for_run(run_id))
    return url


def _resolve_sync_secret() -> str | None:
    """SYNC_SECRET from env, or parsed out of site/.env.local as a dev fallback."""
    secret = __import__("os").environ.get("SYNC_SECRET")
    if secret:
        return secret
    dotenv = Path.home() / "code" / "aiedge" / "site" / ".env.local"
    if dotenv.exists():
        import re
        m = re.search(r"^SYNC_SECRET=(.*)$", dotenv.read_text(), re.M)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None


def _post_json(url: str, payload: dict) -> None:
    import urllib.request

    data = json.dumps(payload, default=str).encode()
    headers = {"Content-Type": "application/json"}
    secret = _resolve_sync_secret()
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"POST {url} → {resp.status}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Print live stats JSON")
    parser.add_argument(
        "--recent", type=int, default=0, help="Print last N live detections"
    )
    parser.add_argument(
        "--push", metavar="URL", nargs="?", const="http://localhost:3000/api/patterns",
        help="POST live stats to dashboard (default: localhost:3000)",
    )
    parser.add_argument(
        "--run-stats", metavar="RUN_ID",
        help="Print stats for a specific backtest run",
    )
    parser.add_argument(
        "--push-run", metavar="RUN_ID",
        help="POST backtest-run stats to /api/patterns/run/<id>",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:3000",
        help="Base URL for --push-run (default: localhost:3000)",
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List backtest runs as JSON",
    )
    parser.add_argument(
        "--purge", choices=["live", "run", "all"],
        help="Retroactively drop vetoed + counter-trend rows. Scope: live (default, run_id IS NULL), run (requires --purge-run-id), or all.",
    )
    parser.add_argument("--purge-run-id", help="Required when --purge run")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --purge: report what WOULD be deleted without deleting",
    )
    args = parser.parse_args()

    if args.purge:
        result = purge_polluted_detections(
            scope=args.purge,
            run_id=args.purge_run_id,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, default=str, indent=2))
    elif args.list_runs:
        print(json.dumps(list_backtest_runs(), default=str, indent=2))
    elif args.push_run:
        url = push_run_to_dashboard(args.push_run, base_url=args.base_url)
        print(url)
    elif args.run_stats:
        print(json.dumps(full_stats_for_run(args.run_stats), default=str))
    elif args.push:
        push_to_dashboard(args.push)
    elif args.stats:
        print(json.dumps(full_stats(), default=str))
    elif args.recent > 0:
        rows = [_detection_payload(d) for d in recent_detections(limit=args.recent, run_id=LIVE_ONLY)]
        print(json.dumps(rows, default=str))
    else:
        parser.print_help()
