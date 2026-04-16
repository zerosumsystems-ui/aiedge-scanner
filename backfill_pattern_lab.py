#!/usr/bin/env python3
"""
Backfill Pattern Lab from saved session pickles.

Replays historical sessions through score_gap() at regular intervals,
logs BPA detections, and immediately resolves outcomes (since we have
the full day's bars).

Usage:
  python3 backfill_pattern_lab.py                    # all available pickles
  python3 backfill_pattern_lab.py --date 2026-04-14  # specific date
"""

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.brooks_score import score_gap
from shared.pattern_lab import (
    log_detection, update_checkpoint, finalize_outcome, detection_count, init_db,
)

LOG_DIR = Path(__file__).parent / "logs" / "live_scanner"
ET = __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("America/New_York")

MIN_BARS = 30
MIN_PRICE = 5.0
MIN_RANGE_PCT = 0.003
SCAN_EVERY_N_BARS = 6  # scan every 6th 5-min bar (~30 min cadence)
OUTCOME_BARS = [5, 10, 20, 30]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min. Mirrors live_scanner.resample_to_5min()."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    df5 = (
        df.resample("5min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"), close=("close", "last"),
             volume=("volume", "sum"))
    )
    df5["open"] = df5["open"].ffill()
    df5["high"] = df5["high"].ffill()
    df5["low"] = df5["low"].ffill()
    df5["close"] = df5["close"].ffill()
    df5["volume"] = df5["volume"].fillna(0)
    df5 = df5.dropna(subset=["open", "close"])
    df5 = df5[df5["open"] > 0]
    return df5.reset_index().rename(columns={"datetime": "datetime"})


def _derive_direction(setup_type: str, entry: float | None, stop: float | None) -> str:
    if setup_type in ("H1", "H2", "FL1", "FL2"):
        return "long"
    elif setup_type in ("L1", "L2"):
        return "short"
    elif entry and stop:
        return "long" if entry > stop else "short"
    return "unknown"


def _compute_outcome(
    det_bar_idx: int, direction: str, entry: float | None,
    stop: float | None, target: float | None, df5_full: pd.DataFrame,
) -> dict:
    """Compute checkpoints and final outcome using the full day's bars."""
    bar_count_at_detect = det_bar_idx + 1  # bars available at detection time
    start_idx = bar_count_at_detect
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
            ck_slice = follow.iloc[:ck]
            checkpoints[ck] = {
                "high": float(ck_slice["high"].max()),
                "low": float(ck_slice["low"].min()),
                "close": float(ck_slice.iloc[-1]["close"]),
            }
        else:
            all_filled = False

    # Target / stop scan
    hit_target_bar = hit_stop_bar = None
    if target and stop and entry:
        for i in range(len(follow)):
            row = follow.iloc[i]
            if direction == "long":
                if row["high"] >= target and hit_target_bar is None:
                    hit_target_bar = i + 1
                if row["low"] <= stop and hit_stop_bar is None:
                    hit_stop_bar = i + 1
            elif direction == "short":
                if row["low"] <= target and hit_target_bar is None:
                    hit_target_bar = i + 1
                if row["high"] >= stop and hit_stop_bar is None:
                    hit_stop_bar = i + 1

    # Result
    result = None
    result_bars = None
    if hit_target_bar and hit_stop_bar:
        if hit_target_bar <= hit_stop_bar:
            result, result_bars = "WIN", hit_target_bar
        else:
            result, result_bars = "LOSS", hit_stop_bar
    elif hit_target_bar:
        result, result_bars = "WIN", hit_target_bar
    elif hit_stop_bar:
        result, result_bars = "LOSS", hit_stop_bar
    elif all_filled:
        result, result_bars = "SCRATCH", 30
    else:
        result = "INCOMPLETE"

    return {
        "checkpoints": checkpoints,
        "mfe": mfe,
        "mae": mae,
        "result": result,
        "result_bars": result_bars,
        "hit_target_bar": hit_target_bar,
        "hit_stop_bar": hit_stop_bar,
    }


def backfill_date(date_str: str) -> int:
    """Backfill Pattern Lab for a single date. Returns detection count."""
    pkl_path = LOG_DIR / f"{date_str}_session.pkl"
    if not pkl_path.exists():
        logger.warning(f"No pickle for {date_str}: {pkl_path}")
        return 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Backfilling {date_str}")
    logger.info(f"{'='*60}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    saved_bars: dict = data["bars"]
    saved_closes: dict = data.get("prior_closes", {})
    saved_atrs: dict = data.get("daily_atrs", {})
    saved_ts: str = data.get("timestamp", date_str)

    logger.info(f"  {len(saved_bars):,} symbols | recorded at {saved_ts}")

    detections_logged = 0
    symbols_with_detections = 0
    t0 = time.monotonic()

    for sym, bar_list in saved_bars.items():
        if len(bar_list) < MIN_BARS:
            continue

        df = pd.DataFrame(bar_list)
        avg_price = df["close"].mean()
        if avg_price < MIN_PRICE:
            continue
        day_range = df["high"].max() - df["low"].min()
        if avg_price > 0 and (day_range / avg_price) < MIN_RANGE_PCT:
            continue

        pc = saved_closes.get(sym)
        if pc is None:
            continue

        df5_full = resample_to_5min(df)
        if len(df5_full) < 10:
            continue

        gap_dir = "up" if df.iloc[0]["open"] > pc else "down"
        sym_detected = False

        # Simulate scanning at regular intervals through the day
        for scan_end in range(10, len(df5_full) + 1, SCAN_EVERY_N_BARS):
            df5_slice = df5_full.iloc[:scan_end]
            if len(df5_slice) < 2:
                continue

            try:
                score = score_gap(
                    df5_slice, prior_close=pc, gap_direction=gap_dir,
                    ticker=sym, daily_atr=saved_atrs.get(sym),
                )
            except Exception:
                continue

            bpa_setups = score.get("details", {}).get("bpa_setups", [])
            if not bpa_setups:
                continue

            # Extract context
            cycle_phase_top = None
            cp = score.get("details", {}).get("cycle_phase", {})
            if isinstance(cp, dict):
                cycle_phase_top = cp.get("top")

            # Approximate scan time from the last bar's datetime
            if "datetime" in df5_slice.columns and len(df5_slice) > 0:
                last_dt = df5_slice.iloc[-1]["datetime"]
                if hasattr(last_dt, "isoformat"):
                    scan_time_str = last_dt.isoformat()
                else:
                    scan_time_str = str(last_dt)
            else:
                scan_time_str = f"{date_str}T12:00:00"

            for setup in bpa_setups:
                setup_type = setup.get("type", "")
                bar_idx = setup.get("bar_index", -1)
                direction = _derive_direction(
                    setup_type, setup.get("entry"), setup.get("stop")
                )

                price_at = None
                if 0 <= bar_idx < len(df5_slice):
                    price_at = float(df5_slice.iloc[bar_idx]["close"])
                elif len(df5_slice) > 0:
                    price_at = float(df5_slice.iloc[-1]["close"])

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
                )

                if rid is None:
                    continue  # dedup — already logged

                # Immediately compute outcome (we have full day's bars)
                outcome = _compute_outcome(
                    bar_idx, direction,
                    setup.get("entry"), setup.get("stop"), setup.get("target"),
                    df5_full,
                )

                # Fill checkpoints
                for ck, vals in outcome.get("checkpoints", {}).items():
                    update_checkpoint(
                        detection_id=rid, checkpoint=ck,
                        high=vals["high"], low=vals["low"], close=vals["close"],
                        mfe=outcome.get("mfe"), mae=outcome.get("mae"),
                    )

                # Finalize
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
                sym_detected = True

        if sym_detected:
            symbols_with_detections += 1

    elapsed = time.monotonic() - t0
    logger.info(f"  {detections_logged} detections across {symbols_with_detections} symbols in {elapsed:.1f}s")
    return detections_logged


def main():
    parser = argparse.ArgumentParser(description="Backfill Pattern Lab from session pickles")
    parser.add_argument("--date", help="Specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    init_db()
    before = detection_count()

    if args.date:
        backfill_date(args.date)
    else:
        # All available pickles
        pickles = sorted(LOG_DIR.glob("*_session.pkl"))
        if not pickles:
            logger.error(f"No session pickles found in {LOG_DIR}")
            sys.exit(1)
        for pkl in pickles:
            date_str = pkl.stem.replace("_session", "")
            backfill_date(date_str)

    after = detection_count()
    logger.info(f"\nPattern Lab: {before} → {after} detections (+{after - before})")


if __name__ == "__main__":
    main()
