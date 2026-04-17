#!/usr/bin/env python3
"""
Claude-driven backtest.

Wraps the historical Databento backfill logic so Claude can invoke it with
natural filters and get back a compact JSON result bundle. Each run writes
to `pattern_lab.sqlite` scoped by a unique run_id — live stats are not
contaminated (partial unique indexes keep dedup scoped per-run).

Usage:
  python3 claude_backtest.py --setups H2,H1 --symbols TSLA,SPY --days 30
  python3 claude_backtest.py --setups L2 --symbols TSLA --start 2026-01-01 --end 2026-02-01
  python3 claude_backtest.py --list-runs
  python3 claude_backtest.py --run-id <id>
  python3 claude_backtest.py --run-id <id> --delete
  python3 claude_backtest.py --setups H2 --symbols TSLA --days 30 --push
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import databento as db
import pandas as pd

# Project path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backfill_historical_databento import (  # noqa: E402
    DATASET,
    DEFAULT_SYMBOLS,
    _rth_bounds,
    _trading_days,
    process_symbol_day,
)
from pattern_lab_api import full_stats_for_run, push_run_to_dashboard  # noqa: E402
from shared.pattern_lab import (  # noqa: E402
    delete_backtest_run,
    delete_run_detections_outside_setups,
    detection_count,
    get_backtest_run,
    init_db,
    list_backtest_runs,
    register_backtest_run,
    update_backtest_run_total,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Databento client ──────────────────────────────────────────────────────────

def _load_databento_key() -> str:
    """Populate DATABENTO_API_KEY from ~/keys/databento.env if not set."""
    if os.environ.get("DATABENTO_API_KEY"):
        return os.environ["DATABENTO_API_KEY"]
    key_path = Path.home() / "keys" / "databento.env"
    if key_path.exists():
        for line in key_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
    if not os.environ.get("DATABENTO_API_KEY"):
        raise RuntimeError(
            "DATABENTO_API_KEY not found — set it or populate ~/keys/databento.env"
        )
    return os.environ["DATABENTO_API_KEY"]


# ── Date resolution ───────────────────────────────────────────────────────────

def _resolve_range(args) -> tuple[date, date]:
    today = date.today()
    end_date = date.fromisoformat(args.end) if args.end else (
        today - timedelta(days=1)
    )
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        td = end_date
        count = 0
        while count < args.days:
            if td.weekday() < 5:
                count += 1
            if count < args.days:
                td -= timedelta(days=1)
        start_date = td
    return start_date, end_date


# ── Daily bar fetch (for prior_close) ─────────────────────────────────────────

def _fetch_daily_closes(client, sym: str, start: date, end: date, et_tz) -> pd.DataFrame:
    daily_start_utc = datetime(
        start.year, start.month, start.day, 0, 0, 0, tzinfo=timezone.utc
    ) - timedelta(days=5)
    daily_end_utc = datetime(
        end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc
    )
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[sym],
        schema="ohlcv-1d",
        start=daily_start_utc.isoformat(),
        end=daily_end_utc.isoformat(),
    )
    df = data.to_df()
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(et_tz)
    df["_date"] = df.index.date
    return df


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_backtest(
    run_id: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    setup_filter: list[str] | None,
    dry_run: bool = False,
) -> int:
    """Execute the backtest for one run. Returns total detections logged."""
    from zoneinfo import ZoneInfo
    et_tz = ZoneInfo("America/New_York")

    _load_databento_key()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    trading_days = _trading_days(start_date, end_date)
    logger.info(
        f"Backtest run {run_id}: {len(symbols)} symbols × {len(trading_days)} days"
    )

    total = 0
    for sym in symbols:
        logger.info(f"\n── {sym} ──")
        try:
            daily_df = _fetch_daily_closes(client, sym, start_date, end_date, et_tz)
        except Exception as e:
            logger.warning(f"  daily bars failed for {sym}: {e}")
            continue
        if daily_df.empty:
            logger.warning(f"  no daily data for {sym}")
            continue

        def prior_close_for(d: date) -> float:
            candidates = daily_df[daily_df["_date"] < d]
            if candidates.empty:
                return 0.0
            return float(candidates.iloc[-1]["close"])

        for d in trading_days:
            date_str = d.isoformat()
            pc = prior_close_for(d)
            if pc <= 0:
                logger.debug(f"  {date_str} {sym}: no prior close — skipping")
                continue

            open_utc, close_utc = _rth_bounds(d)
            try:
                intra_data = client.timeseries.get_range(
                    dataset=DATASET,
                    symbols=[sym],
                    schema="ohlcv-1m",
                    start=open_utc.isoformat(),
                    end=close_utc.isoformat(),
                )
                df1m = intra_data.to_df()
            except Exception as e:
                logger.warning(f"  {date_str} {sym}: intraday fetch failed — {e}")
                continue

            if df1m.empty:
                continue

            n = process_symbol_day(
                sym=sym, date_str=date_str, df1m=df1m,
                prior_close=pc, dry_run=dry_run, run_id=run_id,
            )
            total += n
            logger.info(f"  {date_str} {sym}: {n} detections (pc={pc:.2f})")
            time.sleep(0.2)  # rate limit

    # Post-filter by setup type
    if setup_filter and not dry_run:
        removed = delete_run_detections_outside_setups(run_id, setup_filter)
        if removed:
            logger.info(f"Post-filter: dropped {removed} detections outside {setup_filter}")

    return total


# ── Results builder ───────────────────────────────────────────────────────────

def build_result_bundle(run_id: str, base_url: str = "http://localhost:3000") -> dict:
    """Compact JSON summary for Claude to present in chat."""
    stats = full_stats_for_run(run_id)
    return {
        "runId": run_id,
        "run": stats.get("run"),
        "summary": stats.get("summary"),
        "bySetup": stats.get("bySetup"),
        "byCyclePhase": stats.get("byContext", {}).get("cycle_phase", {}),
        "byAlwaysIn": stats.get("byContext", {}).get("always_in", {}),
        "byDay": stats.get("byDay", []),
        "detections": [
            {
                "id": d.get("id"),
                "ticker": d["ticker"],
                "setupType": d["setupType"],
                "direction": d["direction"],
                "detectedAt": d["detectedAt"],
                "result": d["result"],
                "mfe": d["mfe"],
                "mae": d["mae"],
                "cyclePhase": d["cyclePhase"],
            }
            for d in (stats.get("recentDetections") or [])[:20]
        ],
        "viewerUrl": f"{base_url.rstrip('/')}/patterns/run/{run_id}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Claude-driven backtest")
    parser.add_argument("--setups", help="Comma-separated (H2,H1,L2,FL1...). Omit = all.")
    parser.add_argument("--symbols", help="Comma-separated tickers. Default: top-9.")
    parser.add_argument("--start", help="YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--days", type=int, default=30, help="Trading days back (default 30)")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    parser.add_argument("--list-runs", action="store_true", help="List backtest runs as JSON")
    parser.add_argument("--run-id", help="Existing run to inspect or delete")
    parser.add_argument("--delete", action="store_true", help="With --run-id: delete run")
    parser.add_argument(
        "--push", action="store_true",
        help="POST results to /api/patterns/run/<id> after run completes",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:3000",
        help="Base URL for --push (default localhost:3000)",
    )
    args = parser.parse_args()

    init_db()

    # Read-only modes first
    if args.list_runs:
        print(json.dumps(list_backtest_runs(), default=str, indent=2))
        return

    if args.run_id and args.delete:
        n = delete_backtest_run(args.run_id)
        print(json.dumps({"deleted": args.run_id, "detections_removed": n}))
        return

    if args.run_id and not args.delete:
        if not get_backtest_run(args.run_id):
            print(json.dumps({"error": f"run_id not found: {args.run_id}"}))
            sys.exit(1)
        bundle = build_result_bundle(args.run_id, base_url=args.base_url)
        print(json.dumps(bundle, default=str, indent=2))
        return

    # Running a new backtest
    symbols = [
        s.strip().upper() for s in (args.symbols or ",".join(DEFAULT_SYMBOLS)).split(",")
        if s.strip()
    ]
    setups = [s.strip().upper() for s in args.setups.split(",")] if args.setups else None
    start_date, end_date = _resolve_range(args)

    run_id = datetime.utcnow().strftime("bt-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    if not args.dry_run:
        register_backtest_run(
            run_id=run_id,
            symbols=symbols,
            date_from=start_date.isoformat(),
            date_to=end_date.isoformat(),
            setup_filter=",".join(setups) if setups else None,
            args_json=json.dumps(vars(args)),
        )

    before = detection_count()
    t0 = time.monotonic()
    try:
        total = run_backtest(
            run_id=run_id,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            setup_filter=setups,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Backtest failed mid-run: {e}")
        if not args.dry_run:
            update_backtest_run_total(run_id)
        raise

    elapsed = time.monotonic() - t0
    logger.info(f"\nTotal: {total} detections in {elapsed:.1f}s")

    if args.dry_run:
        print(json.dumps({
            "runId": run_id, "dryRun": True,
            "symbols": symbols,
            "start": start_date.isoformat(), "end": end_date.isoformat(),
            "setupFilter": setups, "detectionsCounted": total,
        }, default=str, indent=2))
        return

    kept = update_backtest_run_total(run_id)
    after = detection_count()
    logger.info(f"Pattern Lab: {before:,} → {after:,} rows (run kept: {kept})")

    if args.push:
        try:
            url = push_run_to_dashboard(run_id, base_url=args.base_url)
            logger.info(f"Pushed to {url}")
        except Exception as e:
            logger.warning(f"Push failed (continuing with CLI output): {e}")

    bundle = build_result_bundle(run_id, base_url=args.base_url)
    print(json.dumps(bundle, default=str, indent=2))


if __name__ == "__main__":
    main()
