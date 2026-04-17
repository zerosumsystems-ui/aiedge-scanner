"""Pattern Lab logging + outcome-update helpers for the live scanner.

Wraps the lower-level `shared.pattern_lab` functions (log_detection,
build_chart_json, get_pending_detections, update_checkpoint,
finalize_outcome). The functions here are called from `run_scan` every
cycle — one to log fresh detections and one to revisit pending
detections and fill in their outcome data as more bars accumulate.

Extracted from live_scanner.py (Phase 4f). The underlying
shared/pattern_lab.py will itself move to aiedge/storage/ later.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

from aiedge.data.resample import resample_to_5min

logger = logging.getLogger(__name__)

# Flipped to False on first `shared.pattern_lab` ImportError — after that,
# every subsequent call short-circuits so we don't spam import errors.
_PATTERN_LAB_OK = True


def log_pattern_lab_detections(
    ticker: str, score: dict, bpa_setups: list[dict],
    df5: pd.DataFrame, scan_time: datetime,
    prior_close: Optional[float] = None,
) -> None:
    """Log BPA detections from one scored ticker to the Pattern Lab database."""
    global _PATTERN_LAB_OK
    if not _PATTERN_LAB_OK:
        return
    try:
        from shared.pattern_lab import log_detection, build_chart_json
    except ImportError:
        _PATTERN_LAB_OK = False
        return

    detection_date = scan_time.strftime("%Y-%m-%d")

    cycle_phase_top = None
    cp = score.get("details", {}).get("cycle_phase", {})
    if isinstance(cp, dict):
        cycle_phase_top = cp.get("top")

    for setup in bpa_setups:
        setup_type = setup.get("type", "")
        bar_idx = setup.get("bar_index", -1)

        # Derive direction from setup type
        if setup_type in ("H1", "H2", "FL1", "FL2"):
            direction = "long"
        elif setup_type in ("L1", "L2", "FH1", "FH2"):
            direction = "short"
        else:
            entry, stop = setup.get("entry"), setup.get("stop")
            if entry and stop:
                direction = "long" if entry > stop else "short"
            else:
                direction = "unknown"

        # Price at detection bar
        price_at = None
        if 0 <= bar_idx < len(df5):
            price_at = float(df5.iloc[bar_idx]["close"])
        elif len(df5) > 0:
            price_at = float(df5.iloc[-1]["close"])

        # Chart window ±30/+20 around signal. At live scan time we only have
        # bars up to 'now', so window_after naturally truncates — later scans
        # will see a longer window if the same detection re-fires (it won't
        # re-insert due to dedup, which is fine).
        chart_json = build_chart_json(
            df5_full=df5,
            bar_index=bar_idx,
            entry=setup.get("entry"),
            stop=setup.get("stop"),
            target=setup.get("target"),
            direction=direction,
            prior_close=prior_close,
            cycle_phase=cycle_phase_top,
        )

        log_detection(
            ticker=ticker,
            setup_type=setup_type,
            detected_at=scan_time.isoformat(),
            detection_date=detection_date,
            bar_index=bar_idx,
            bar_count_at_detect=len(df5),
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
            gap_direction="up" if score.get("gap_held") else score.get("details", {}).get("gap_direction"),
            bpa_alignment=score.get("details", {}).get("bpa_alignment"),
            chart_json=chart_json,
        )


def update_pattern_lab_outcomes(
    scan_time: datetime,
    bars: dict[str, list[dict]],
    bars_lock: threading.Lock,
) -> None:
    """Revisit pending detections and fill in outcome data from current bars.

    `bars` + `bars_lock` are the caller's shared mutable state (the live
    scanner's 1-min bar accumulator). We snapshot under the lock, then
    resample + compare to pending detections without holding it.
    """
    global _PATTERN_LAB_OK
    if not _PATTERN_LAB_OK:
        return
    try:
        from shared.pattern_lab import (
            get_pending_detections, update_checkpoint, finalize_outcome,
        )
    except ImportError:
        _PATTERN_LAB_OK = False
        return

    today = scan_time.strftime("%Y-%m-%d")
    pending = get_pending_detections(detection_date=today)
    if not pending:
        return

    with bars_lock:
        bars_snapshot = {sym: list(bl) for sym, bl in bars.items()}

    # Group by ticker to avoid redundant resamples
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for det in pending:
        by_ticker[det["ticker"]].append(det)

    updated = 0
    for ticker, dets in by_ticker.items():
        bar_list = bars_snapshot.get(ticker)
        if not bar_list:
            continue

        df5 = resample_to_5min(pd.DataFrame(bar_list))
        if len(df5) == 0:
            continue

        for det in dets:
            detect_bar_count = det["bar_count_at_detect"]
            bars_elapsed = len(df5) - detect_bar_count
            if bars_elapsed <= 0:
                continue

            start_idx = detect_bar_count
            if start_idx >= len(df5):
                continue
            follow_bars = df5.iloc[start_idx:]

            entry = det["entry_price"]
            stop = det["stop_price"]
            target = det["target_price"]
            direction = det["direction"]

            # Running MFE / MAE
            mfe = mae = None
            if entry and len(follow_bars) > 0:
                if direction == "long":
                    mfe = float(follow_bars["high"].max()) - entry
                    mae = entry - float(follow_bars["low"].min())
                elif direction == "short":
                    mfe = entry - float(follow_bars["low"].min())
                    mae = float(follow_bars["high"].max()) - entry

            # Fill checkpoints
            checkpoints = [5, 10, 20, 30]
            all_filled = True
            for ck in checkpoints:
                if det.get(f"ck{ck}_high") is not None:
                    continue  # already filled
                if bars_elapsed >= ck:
                    ck_bars = follow_bars.iloc[:ck]
                    update_checkpoint(
                        detection_id=det["id"],
                        checkpoint=ck,
                        high=float(ck_bars["high"].max()),
                        low=float(ck_bars["low"].min()),
                        close=float(ck_bars.iloc[-1]["close"]),
                        mfe=mfe,
                        mae=mae,
                    )
                    updated += 1
                else:
                    all_filled = False

            # Determine final result (target/stop hit scan)
            if target and stop and entry and len(follow_bars) > 0:
                hit_target_bar = hit_stop_bar = None
                for i in range(len(follow_bars)):
                    row = follow_bars.iloc[i]
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

                result = result_bars_val = None
                if hit_target_bar and hit_stop_bar:
                    if hit_target_bar <= hit_stop_bar:
                        result, result_bars_val = "WIN", hit_target_bar
                    else:
                        result, result_bars_val = "LOSS", hit_stop_bar
                elif hit_target_bar:
                    result, result_bars_val = "WIN", hit_target_bar
                elif hit_stop_bar:
                    result, result_bars_val = "LOSS", hit_stop_bar
                elif all_filled:
                    result, result_bars_val = "SCRATCH", 30

                if result:
                    finalize_outcome(
                        detection_id=det["id"],
                        result=result,
                        result_bars=result_bars_val,
                        hit_target_bar=hit_target_bar,
                        hit_stop_bar=hit_stop_bar,
                        mfe=mfe,
                        mae=mae,
                    )

    if updated:
        logger.info(f"Pattern Lab: updated {updated} outcome checkpoints")
