#!/usr/bin/env python3
"""
Historical BPA Backtest via Databento
======================================
Fetches ohlcv-1m from Databento (EQUS.MINI) for a set of equity symbols
over a date range, runs BPA detection at regular intervals through each
trading day, and stores all detections + outcomes in pattern_lab.sqlite.

Mirrors the logic in backfill_pattern_lab.py but pulls from Databento
instead of requiring saved session pickles.

Usage:
  python3 backfill_historical_databento.py                          # default: top-9, 60 days
  python3 backfill_historical_databento.py --days 30               # last 30 trading days
  python3 backfill_historical_databento.py --symbols SPY QQQ TSLA  # specific symbols
  python3 backfill_historical_databento.py --start 2026-01-01 --end 2026-03-31
  python3 backfill_historical_databento.py --dry-run               # count trades, no DB writes
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import databento as db
import pandas as pd

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.brooks_score import score_gap
from shared.pattern_lab import (
    detection_count, finalize_outcome, init_db,
    log_detection, update_checkpoint, build_chart_json,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ET = ZoneInfo("America/New_York")
DATASET = "EQUS.MINI"

# RTH window
RTH_OPEN_H, RTH_OPEN_M   = 9, 30
RTH_CLOSE_H, RTH_CLOSE_M = 16, 0

# Same params as backfill_pattern_lab.py
MIN_BARS       = 30          # min 5-min bars before we start scanning
MIN_PRICE      = 5.0         # skip penny stocks
MIN_RANGE_PCT  = 0.003       # skip dead days
SCAN_EVERY_N   = 6           # scan every 6th 5-min bar (~30 min cadence)
OUTCOME_BARS   = [5, 10, 20, 30]

# Top 9 microgap symbols (ranked by PF from backtest)
DEFAULT_SYMBOLS = ["TSLA", "SPY", "QQQ", "CRM", "TQQQ", "META", "AMD", "IWM", "SQQQ"]

# Dates we already have from session pickles — skip to avoid double-counting
SKIP_DATES = {"2026-04-13", "2026-04-14", "2026-04-15"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trading_days(start: date, end: date) -> list[date]:
    """Return Mon-Fri dates in [start, end] inclusive."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon=0 … Fri=4
            days.append(d)
        d += timedelta(days=1)
    return days


def _rth_bounds(d: date):
    """Return (open_utc, close_utc) for RTH on date d."""
    open_et  = datetime(d.year, d.month, d.day, RTH_OPEN_H,  RTH_OPEN_M,  tzinfo=ET)
    close_et = datetime(d.year, d.month, d.day, RTH_CLOSE_H, RTH_CLOSE_M, tzinfo=ET)
    return open_et.astimezone(timezone.utc), close_et.astimezone(timezone.utc)


def _resample_to_5min(df1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min. Returns DataFrame with a 'datetime' column."""
    df = df1m.copy()
    df.columns = [c.lower() for c in df.columns]

    # Normalise datetime index
    ts_col = next((c for c in ("ts_event", "timestamp", "datetime") if c in df.columns), None)
    if ts_col:
        df = df.set_index(ts_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("No datetime index found after normalisation")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    ohlcv = {c: c for c in ("open", "high", "low", "close", "volume") if c in df.columns}
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in ohlcv:
        agg["volume"] = "sum"

    df5 = df[list(ohlcv.values())].resample("5min", label="left", closed="left").agg(agg)
    df5["open"]  = df5["open"].ffill()
    df5["high"]  = df5["high"].ffill()
    df5["low"]   = df5["low"].ffill()
    df5["close"] = df5["close"].ffill()
    if "volume" in agg:
        df5["volume"] = df5["volume"].fillna(0)
    df5 = df5.dropna(subset=["open", "close"])
    df5 = df5[df5["open"] > 0]
    return df5.reset_index().rename(columns={"index": "datetime", df5.index.name or "index": "datetime"})


def _derive_direction(setup_type: str, entry, stop) -> str:
    if setup_type in ("H1", "H2", "FL1", "FL2"):
        return "long"
    elif setup_type in ("L1", "L2"):
        return "short"
    elif entry and stop:
        return "long" if entry > stop else "short"
    return "unknown"


def _compute_outcome(det_bar_idx: int, direction: str, entry, stop, target,
                     df5_full: pd.DataFrame) -> dict:
    """Compute checkpoints and result — identical to backfill_pattern_lab.py."""
    start_idx = det_bar_idx + 1
    if start_idx >= len(df5_full):
        return {"outcome_status": "complete", "result": "INCOMPLETE"}

    follow = df5_full.iloc[start_idx:]
    if len(follow) == 0:
        return {"outcome_status": "complete", "result": "INCOMPLETE"}

    # MFE / MAE
    mfe = mae = None
    if entry:
        if direction == "long":
            mfe = float(follow["high"].max()) - entry
            mae = entry - float(follow["low"].min())
        elif direction == "short":
            mfe = entry - float(follow["low"].min())
            mae = float(follow["high"].max()) - entry

    # Checkpoints
    checkpoints = {}
    all_filled = True
    for ck in OUTCOME_BARS:
        if len(follow) >= ck:
            sl = follow.iloc[:ck]
            checkpoints[ck] = {
                "high":  float(sl["high"].max()),
                "low":   float(sl["low"].min()),
                "close": float(sl.iloc[-1]["close"]),
            }
        else:
            all_filled = False

    # Target / stop scan
    hit_target_bar = hit_stop_bar = None
    if target and stop and entry:
        for i in range(len(follow)):
            row = follow.iloc[i]
            if direction == "long":
                if row["high"] >= target  and hit_target_bar is None:
                    hit_target_bar = i + 1
                if row["low"]  <= stop    and hit_stop_bar   is None:
                    hit_stop_bar   = i + 1
            elif direction == "short":
                if row["low"]  <= target  and hit_target_bar is None:
                    hit_target_bar = i + 1
                if row["high"] >= stop    and hit_stop_bar   is None:
                    hit_stop_bar   = i + 1

    result = result_bars = None
    if hit_target_bar and hit_stop_bar:
        if hit_target_bar <= hit_stop_bar:
            result, result_bars = "WIN",  hit_target_bar
        else:
            result, result_bars = "LOSS", hit_stop_bar
    elif hit_target_bar:
        result, result_bars = "WIN",  hit_target_bar
    elif hit_stop_bar:
        result, result_bars = "LOSS", hit_stop_bar
    elif all_filled:
        result, result_bars = "SCRATCH", 30
    else:
        result = "INCOMPLETE"

    return {
        "checkpoints":    checkpoints,
        "mfe":            mfe,
        "mae":            mae,
        "result":         result,
        "result_bars":    result_bars,
        "hit_target_bar": hit_target_bar,
        "hit_stop_bar":   hit_stop_bar,
    }


# ── Core per-symbol per-day processor ────────────────────────────────────────

def process_symbol_day(
    sym: str,
    date_str: str,
    df1m: pd.DataFrame,
    prior_close: float,
    dry_run: bool = False,
    run_id: "Optional[str]" = None,
) -> int:
    """
    Run BPA detection through one symbol's trading day.
    Returns number of detections logged.

    When run_id is set, detections are tagged with the run and may overlap
    live data (dedup is scoped to the run via partial unique index).
    """
    if df1m.empty or prior_close <= 0:
        return 0

    try:
        df5_full = _resample_to_5min(df1m)
    except Exception as e:
        logger.debug(f"  {sym} {date_str} resample failed: {e}")
        return 0

    if len(df5_full) < MIN_BARS:
        return 0

    avg_price = float(df5_full["close"].mean())
    if avg_price < MIN_PRICE:
        return 0

    day_range = float(df5_full["high"].max()) - float(df5_full["low"].min())
    if avg_price > 0 and (day_range / avg_price) < MIN_RANGE_PCT:
        return 0

    gap_dir = "up" if float(df5_full.iloc[0]["open"]) > prior_close else "down"

    detections_logged = 0

    for scan_end in range(10, len(df5_full) + 1, SCAN_EVERY_N):
        df5_slice = df5_full.iloc[:scan_end]
        if len(df5_slice) < 2:
            continue

        try:
            score = score_gap(
                df5_slice, prior_close=prior_close, gap_direction=gap_dir,
                ticker=sym,
            )
        except Exception:
            continue

        bpa_setups = score.get("details", {}).get("bpa_setups", [])
        if not bpa_setups:
            continue

        # Extract context
        cp = score.get("details", {}).get("cycle_phase", {})
        cycle_phase_top = cp.get("top") if isinstance(cp, dict) else None

        last_dt = df5_slice.iloc[-1]["datetime"]
        scan_time_str = last_dt.isoformat() if hasattr(last_dt, "isoformat") else str(last_dt)

        for setup in bpa_setups:
            setup_type = setup.get("type", "")
            bar_idx    = setup.get("bar_index", -1)
            direction  = _derive_direction(setup_type, setup.get("entry"), setup.get("stop"))

            price_at = None
            if 0 <= bar_idx < len(df5_slice):
                price_at = float(df5_slice.iloc[bar_idx]["close"])
            elif len(df5_slice) > 0:
                price_at = float(df5_slice.iloc[-1]["close"])

            if dry_run:
                detections_logged += 1
                continue

            chart_json = build_chart_json(
                df5_full=df5_full,
                bar_index=bar_idx,
                entry=setup.get("entry"),
                stop=setup.get("stop"),
                target=setup.get("target"),
                direction=direction,
                prior_close=prior_close,
                cycle_phase=cycle_phase_top,
            )

            rid = log_detection(
                ticker=sym,
                setup_type=setup_type,
                detected_at=scan_time_str,
                detection_date=date_str,
                bar_index=bar_idx,
                bar_count_at_detect=len(df5_slice),
                session_bar_number=bar_idx,
                entry_price=setup.get("entry"),
                stop_price=setup.get("stop"),
                target_price=setup.get("target"),
                entry_mode=setup.get("entry_mode"),
                confidence=setup.get("confidence", 0.0),
                direction=direction,
                price_at_detect=price_at or 0.0,
                urgency=score.get("urgency"),
                uncertainty=score.get("uncertainty"),
                always_in=score.get("always_in"),
                cycle_phase=cycle_phase_top,
                day_type=score.get("day_type"),
                signal=score.get("signal"),
                gap_direction=gap_dir,
                bpa_alignment=score.get("details", {}).get("bpa_alignment"),
                chart_json=chart_json,
                run_id=run_id,
                # Backtests evaluate all buckets — never filter by whitelist
                # (otherwise we couldn't ever rebuild the whitelist from data).
                enforce_whitelist=False,
            )

            if rid is None:
                continue  # dedup — already logged

            outcome = _compute_outcome(
                bar_idx, direction,
                setup.get("entry"), setup.get("stop"), setup.get("target"),
                df5_full,
            )

            for ck, vals in outcome.get("checkpoints", {}).items():
                update_checkpoint(
                    detection_id=rid, checkpoint=ck,
                    high=vals["high"], low=vals["low"], close=vals["close"],
                    mfe=outcome.get("mfe"), mae=outcome.get("mae"),
                )

            finalize_outcome(
                detection_id=rid,
                result=outcome["result"],
                result_bars=outcome.get("result_bars"),
                hit_target_bar=outcome.get("hit_target_bar"),
                hit_stop_bar=outcome.get("hit_stop_bar"),
                mfe=outcome.get("mfe"),
                mae=outcome.get("mae"),
            )

            detections_logged += 1

    return detections_logged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Historical BPA backtest via Databento")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Ticker symbols (default: top-9 microgap symbols)")
    parser.add_argument("--days",  type=int, default=60,
                        help="Number of trading days to look back (default: 60)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count detections only — no DB writes")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip dates already in pattern_lab.sqlite (default: on)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    # ── Load API key ──────────────────────────────────────────────────────────
    key_path = Path.home() / "keys" / "databento.env"
    if key_path.exists():
        for line in key_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not found. Set it or add to ~/keys/databento.env")
        sys.exit(1)

    client = db.Historical(api_key)

    # ── Date range ────────────────────────────────────────────────────────────
    today = date.today()
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        # Default: up through yesterday
        end_date = today - timedelta(days=1)
        while end_date.weekday() >= 5:
            end_date -= timedelta(days=1)

    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        # Walk back N trading days from end_date
        td = end_date
        count = 0
        while count < args.days:
            if td.weekday() < 5:
                count += 1
            if count < args.days:
                td -= timedelta(days=1)
        start_date = td

    trading_days = _trading_days(start_date, end_date)
    # Skip dates we already have from session pickles
    trading_days = [d for d in trading_days if d.isoformat() not in SKIP_DATES]

    logger.info(f"BPA Historical Backtest")
    logger.info(f"  Symbols : {args.symbols}")
    logger.info(f"  Range   : {start_date} → {end_date} ({len(trading_days)} trading days)")
    logger.info(f"  Dry run : {args.dry_run}")
    logger.info(f"  Dataset : {DATASET}")

    if not args.dry_run:
        init_db()
        before = detection_count()
        logger.info(f"  DB before: {before:,} detections")

    total_detections = 0
    t_global = time.monotonic()

    for sym in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Symbol: {sym}")
        logger.info(f"{'='*60}")
        sym_detections = 0
        t_sym = time.monotonic()

        # ── 1. Fetch daily bars for prior-close lookup ─────────────────────
        daily_start_utc = datetime(
            start_date.year, start_date.month, start_date.day,
            0, 0, 0, tzinfo=timezone.utc
        ) - timedelta(days=5)  # buffer for weekends/holidays

        daily_end_utc = datetime(
            end_date.year, end_date.month, end_date.day,
            23, 59, 59, tzinfo=timezone.utc
        )

        logger.info(f"  Fetching daily bars...")
        try:
            daily_data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=[sym],
                schema="ohlcv-1d",
                start=daily_start_utc.isoformat(),
                end=daily_end_utc.isoformat(),
            )
            daily_df = daily_data.to_df()
            daily_df.columns = [c.lower() for c in daily_df.columns]
        except Exception as e:
            logger.warning(f"  Daily bars failed for {sym}: {e}")
            continue

        if daily_df.empty:
            logger.warning(f"  No daily data for {sym}")
            continue

        # Normalise daily index to date
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            for cand in ("ts_event", "timestamp", "date"):
                if cand in daily_df.columns:
                    daily_df = daily_df.set_index(cand)
                    break
        if daily_df.index.tz is None:
            daily_df.index = daily_df.index.tz_localize("UTC")
        daily_df.index = daily_df.index.tz_convert(ET)
        daily_df["_date"] = daily_df.index.date

        def get_prior_close(d: date) -> float:
            """Return prior trading day's close for symbol."""
            candidates = daily_df[daily_df["_date"] < d]
            if candidates.empty:
                return 0.0
            return float(candidates.iloc[-1]["close"])

        # ── 2. Process each trading day ────────────────────────────────────
        for d in trading_days:
            date_str = d.isoformat()

            # Skip if already in DB
            if args.skip_existing and not args.dry_run:
                import sqlite3
                db_path = PROJECT_ROOT / "db" / "pattern_lab.sqlite"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    n = conn.execute(
                        "SELECT COUNT(*) FROM detections WHERE detection_date=? AND ticker=?",
                        (date_str, sym),
                    ).fetchone()[0]
                    conn.close()
                    if n > 0:
                        logger.info(f"  {date_str} {sym}: skipping ({n} detections already in DB)")
                        continue

            prior_close = get_prior_close(d)
            if prior_close <= 0:
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
                logger.debug(f"  {date_str} {sym}: no intraday data")
                continue

            n = process_symbol_day(sym, date_str, df1m, prior_close, dry_run=args.dry_run)
            sym_detections += n
            total_detections += n
            logger.info(f"  {date_str} {sym}: {n} detections (prior_close={prior_close:.2f})")

            # Small pause to avoid hammering the API
            time.sleep(0.2)

        elapsed_sym = time.monotonic() - t_sym
        logger.info(f"  {sym} total: {sym_detections} detections in {elapsed_sym:.1f}s")

    elapsed = time.monotonic() - t_global
    logger.info(f"\n{'='*60}")
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Total detections found: {total_detections:,}")

    if not args.dry_run:
        after = detection_count()
        logger.info(f"Pattern Lab: {before:,} → {after:,} (+{after - before:,})")

    return total_detections


if __name__ == "__main__":
    main()
