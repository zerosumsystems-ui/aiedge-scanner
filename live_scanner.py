#!/usr/bin/env python3
"""
live_scanner.py — Real-time US equities scanner (Brooks price action)
======================================================================
Streams ALL US equities via Databento Live websocket, accumulates 1-min bars
in memory, and runs Brooks price action scoring every N minutes to find
trend-from-open and small pullback trend stocks.

Schedule:
  9:25 AM ET  — Connect to Live feed, fetch prior closes
  9:35 AM ET  — First scan cycle
  Every 5 min — Subsequent scans (configurable)
  4:05 PM ET  — Clean shutdown, save final results

Usage:
  python live_scanner.py                    # Run with defaults
  python live_scanner.py --test             # Connect, accumulate 2 min, scan once, exit
  python live_scanner.py --scan-interval 3  # Scan every 3 minutes
  python live_scanner.py --top 10           # Show top 10
  python live_scanner.py --min-urgency 7    # Only alert on urgency >= 7
"""

import argparse
import base64
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "credentials" / ".env")

from shared.brooks_score import score_gap
from shared.chart_renderer import render_chart

from aiedge.data.databento import (
    DATASET,
    ET,
    SCHEMA,
    _fetch_ohlcv1m_range,
    _prev_trading_days,
    _timeout_handler,
    backfill_intraday_bars as _backfill_intraday_bars_pure,
    fetch_prior_closes,
    with_timeout,
)
from aiedge.data.levels import fetch_intraday_key_levels
from aiedge.data.resample import resample_to_5min
from aiedge.dashboard.charts import render_chart_base64
from aiedge.dashboard.render import (
    _HTML_FOOT,
    _HTML_HEAD,
    _SIG_CSS,
    _adr_mult_tier,
    _bar_html,
    _build_card_html,
    _build_component_strip,
    _generate_dashboard as _generate_dashboard_pure,
    _movement_html,
    _rank_arrow_html,
    _signal_badge,
)
from aiedge.dashboard.console import (
    _BOLD,
    _COLORS,
    _DOWN,
    _RST,
    _UP,
    _W,
    _format_note_text as _format_note_text_pure,
    _movement_arrow,
    _next_scan_after as _next_scan_after_pure,
    _next_scan_time_str as _next_scan_time_str_pure,
    print_leaderboard as _print_leaderboard_pure,
)
from aiedge.signals.postprocess import (
    ETF_FAMILIES,
    SAME_COMPANY,
    _TICKER_TO_FAMILY,
    _compute_movement,
    _dedup_etf_families,
    _fmt_delta,
    _fmt_movement,
    annotate_adr_multiple as _annotate_adr_multiple_pure,
)
from aiedge.storage.pattern_lab import (
    log_pattern_lab_detections,
    update_pattern_lab_outcomes,
)

import databento as db

# ── Constants ─────────────────────────────────────────────────────────────────
# DATASET, ET, SCHEMA now live in aiedge.data.databento (re-imported above).

FIRST_SCAN_HOUR = 9
FIRST_SCAN_MIN = 35
SHUTDOWN_HOUR = 16
SHUTDOWN_MIN = 5

DEFAULT_SCAN_INTERVAL = 5   # minutes
DEFAULT_TOP_N = 20
DEFAULT_MIN_URGENCY = 0.0

MIN_DOLLAR_VOL = 350_000   # avg bar dollar volume (price × volume)
MIN_PRICE = 5.0
MIN_BARS = 2
MIN_RANGE_PCT = 0.003      # min intraday range as % of price (0.3%) — filters bond/money-market ETFs
MIN_BAR_COMPLETENESS = 0.90  # must have >=90% of expected 1-min bars — filters illiquid tape with gaps

DASHBOARD_CHARTS = 15       # how many stocks get a chart rendered per cycle
DASHBOARD_PATH = ROOT / "logs" / "live_scanner" / "dashboard.html"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs" / "live_scanner"
LOG_DIR.mkdir(parents=True, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"{today_str}.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Global state (thread-safe) ────────────────────────────────────────────────
bars: dict[str, list[dict]] = {}        # {symbol: [bar_dict, ...]}
bars_lock = threading.Lock()

prior_closes: dict[str, float] = {}     # {symbol: prior_close_price}
daily_atrs: dict[str, float] = {}       # {symbol: 20-day avg daily range in $}

instrument_map: dict[int, str] = {}     # {instrument_id: symbol}
instrument_map_lock = threading.Lock()

scan_results_history: list = []         # accumulate all scan results for final save
stop_event = threading.Event()

# Rank/urgency cache from previous scan cycle — resets on process restart (correct)
_prior_scan: dict[str, dict] = {}       # {ticker: {rank, urgency, uncertainty}}

# Intraday key-level cache (PDH/PDL/ONH/ONL/PMH/PML) — populated once at startup
intraday_levels: dict[str, dict[str, float]] = {}   # {symbol: {"pdh":.., "pdl":.., "onh":.., "onl":.., "pmh":.., "pml":..}}

# ── ETF / dual-class family dedup: moved to aiedge.signals.postprocess
# (Phase 4e). Re-imported at top of file — legacy names still resolve
# on `ls.ETF_FAMILIES` etc.


def annotate_adr_multiple(score: dict, df_1m: pd.DataFrame, sym: str) -> None:
    """Backwards-compatible wrapper: binds `daily_atrs` to the module-level
    cache so existing single-arg call sites keep working. See
    `aiedge.signals.postprocess.annotate_adr_multiple` for the pure version.
    """
    _annotate_adr_multiple_pure(score, df_1m, sym, daily_atrs)

# ── Prior Closes ──────────────────────────────────────────────────────────────








# ── Bar Accumulation ──────────────────────────────────────────────────────────

def process_record(record) -> None:
    """Route an incoming DBN record to the appropriate handler."""
    rtype = type(record)

    if rtype is db.SymbolMappingMsg:
        # stype_in_symbol is the subscription input side ("ALL_SYMBOLS" when
        # we subscribe to the entire universe) — useless for attribution.
        # stype_out_symbol is the resolved raw ticker ("AAPL", "MSFT", ...) —
        # this is what we index bars by.
        sym = getattr(record, "stype_out_symbol", None) \
              or getattr(record, "stype_in_symbol", None)
        iid = getattr(record, "instrument_id", None)
        if sym and iid is not None:
            with instrument_map_lock:
                instrument_map[iid] = sym
            # Per-mapping logging at DEBUG only — EQUS.MINI sends ~12k mappings
            # per session and INFO-level spam throttles the stream thread.
            logger.debug(f"Symbol mapping: {iid} → {sym}")
        return

    if rtype is db.OHLCVMsg:
        iid = record.instrument_id
        with instrument_map_lock:
            sym = instrument_map.get(iid)
        if sym is None:
            return

        try:
            bar = {
                "datetime": pd.Timestamp(record.ts_event, unit="ns", tz="UTC").tz_convert(ET),
                "open":   record.pretty_open,
                "high":   record.pretty_high,
                "low":    record.pretty_low,
                "close":  record.pretty_close,
                "volume": int(record.volume),
            }
            with bars_lock:
                if sym not in bars:
                    bars[sym] = []
                bars[sym].append(bar)
        except Exception as e:
            logger.debug(f"Bar ingestion error for instrument {iid}: {e}")
        return

# ── Resampling ────────────────────────────────────────────────────────────────




# ── Chart Rendering ───────────────────────────────────────────────────────────


# ── Intraday Key Levels (PDH/PDL/ONH/ONL/PMH/PML) ─────────────────────────────





# ── ETF Family Dedup ──────────────────────────────────────────────────────────



# ── Rank Movement ─────────────────────────────────────────────────────────────






# ── Pattern Lab Helpers (moved to aiedge.storage.pattern_lab in Phase 4f) ───
# Re-imported above as log_pattern_lab_detections + update_pattern_lab_outcomes.
# Local wrappers below preserve the legacy private names.


def _log_pattern_lab_detections(
    ticker: str, score: dict, bpa_setups: list[dict],
    df5: pd.DataFrame, scan_time: datetime,
    prior_close: "float | None" = None,
) -> None:
    """Backwards-compatible alias for aiedge.storage.pattern_lab."""
    log_pattern_lab_detections(ticker, score, bpa_setups, df5, scan_time, prior_close)


def _update_pattern_lab_outcomes(scan_time: datetime) -> None:
    """Backwards-compatible wrapper — binds this module's bars/bars_lock."""
    update_pattern_lab_outcomes(scan_time, bars, bars_lock)





# ── Scan Cycle ────────────────────────────────────────────────────────────────

def run_scan(args) -> list[dict]:
    """Score all liquid symbols, annotate movement, print leaderboard, update outputs."""
    global _prior_scan
    t0 = time.monotonic()
    now_et = datetime.now(ET)
    logger.info(f"=== Scan cycle starting at {now_et.strftime('%H:%M:%S')} ET ===")

    with bars_lock:
        snapshot = {sym: list(bl) for sym, bl in bars.items()}

    results = []
    # Also keep the 5m DataFrames for chart rendering (top N only)
    df5m_cache: dict[str, pd.DataFrame] = {}
    skipped_no_prior = 0
    skipped_filter = 0

    for sym, bar_list in snapshot.items():
        try:
            if len(bar_list) < MIN_BARS:
                continue

            df = pd.DataFrame(bar_list)

            avg_price = df["close"].mean()
            if avg_price < MIN_PRICE:
                skipped_filter += 1
                continue

            avg_dv = (df["close"] * df["volume"]).mean()
            if avg_dv < MIN_DOLLAR_VOL:
                skipped_filter += 1
                continue

            # Range filter: skip instruments with no meaningful price action
            day_range = df["high"].max() - df["low"].min()
            if avg_price > 0 and (day_range / avg_price) < MIN_RANGE_PCT:
                skipped_filter += 1
                continue

            # Bar completeness: skip illiquid tape with gaps
            if "datetime" in df.columns and len(df) >= 2:
                session_minutes = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 60
                expected_bars = max(session_minutes, 1)
                if len(df) / expected_bars < MIN_BAR_COMPLETENESS:
                    skipped_filter += 1
                    continue

            pc = prior_closes.get(sym)
            if pc is None:
                skipped_no_prior += 1
                continue

            df5 = resample_to_5min(df)
            if len(df5) < 2:
                continue

            gap_dir = "up" if df.iloc[0]["open"] > pc else "down"
            score = score_gap(df5, prior_close=pc, gap_direction=gap_dir, ticker=sym,
                              daily_atr=daily_atrs.get(sym))
            score["_prior_close"] = pc

            # ── SPT Enhancement: Track current (not peak) and sustained count ──
            # Extract current SPT from the last-computed 15-bar window (in 5m bars)
            # This reflects the CURRENT phase of the trend, not peak from earlier in session
            current_spt = score["details"].get("small_pullback_trend", 0.0)
            score["current_spt"] = current_spt

            # Count consecutive 15-bar windows where SPT >= 2.0 ending at close
            # (This would normally track across scans, but for now we report the flag)
            score["sustained_spt_count"] = 1 if current_spt >= 2.0 else 0

            annotate_adr_multiple(score, df, sym)
            results.append(score)
            df5m_cache[sym] = df5

            # Pattern Lab: log any BPA detections for this symbol
            _bpa_setups = score.get("details", {}).get("bpa_setups", [])
            if _bpa_setups:
                _log_pattern_lab_detections(
                    sym, score, _bpa_setups, df5, now_et, prior_close=pc
                )

        except Exception as e:
            logger.debug(f"Scoring error for {sym}: {e}")

    results.sort(key=lambda x: -x.get("urgency", 0))
    # Collapse ETF family repeats (QQQ/TQQQ/SQQQ all on same NDX move → keep one leader)
    results = _dedup_etf_families(results)
    results = _compute_movement(results, _prior_scan)

    elapsed_score = time.monotonic() - t0
    passed = len(results)

    with instrument_map_lock:
        n_mapped = len(instrument_map)
    logger.info(
        f"Scored {passed} symbols | "
        f"filtered {skipped_filter} | no prior close {skipped_no_prior} | "
        f"total streaming {len(snapshot)} | mapped iids {n_mapped} | "
        f"{elapsed_score:.2f}s"
    )

    top = results[: args.top]
    print_leaderboard(top, now_et, len(snapshot), passed, elapsed_score, args.scan_interval)

    # Alerts
    for r in results:
        urg = r.get("urgency", 0)
        sig = r.get("signal", "")
        if urg >= args.min_urgency and sig in (
            "BUY_PULLBACK", "SELL_PULLBACK", "BUY_SPIKE", "SELL_SPIKE",
            "SELL_PULLBACK_INTRADAY", "BUY_PULLBACK_INTRADAY",
        ):
            fire_alert(r)

    scan_results_history.append(
        {"timestamp": now_et.isoformat(), "results": results[:50]}
    )

    # Pattern Lab: update outcomes for pending detections
    _update_pattern_lab_outcomes(now_et)

    # Generate HTML dashboard (with charts) in background thread
    # Pass a snapshot of top-N results + their DataFrames
    chart_candidates = results[:DASHBOARD_CHARTS]
    df5m_for_dash = {
        r["ticker"]: df5m_cache[r["ticker"]]
        for r in chart_candidates
        if r["ticker"] in df5m_cache
    }
    threading.Thread(
        target=_generate_dashboard,
        args=(chart_candidates, df5m_for_dash, now_et, len(snapshot), passed,
              elapsed_score, args.scan_interval),
        daemon=True,
    ).start()

    # POST to aiedge.trade (fire and forget) — raw OHLC bars flow through; site
    # renders interactive charts via lightweight-charts (Phase 6).
    threading.Thread(
        target=_post_to_aiedge,
        args=(chart_candidates, now_et, len(snapshot), passed,
              elapsed_score, args.scan_interval, df5m_for_dash),
        daemon=True,
    ).start()

    # Update Apple Note (fire and forget)
    note_text = _format_note_text(top, now_et, len(snapshot), passed, elapsed_score, args.scan_interval)
    threading.Thread(
        target=update_apple_note, args=(note_text,), daemon=True
    ).start()

    _prior_scan = {
        r["ticker"]: {
            "rank": r["rank"],
            "urgency": r["urgency"],
            "uncertainty": r["uncertainty"],
        }
        for r in results
    }

    return results

# ── Console Output (moved to aiedge.dashboard.console in Phase 4g-1) ──────────
# ANSI color palette + leaderboard + note-text formatters are re-imported above.
# fire_alert() in this file uses the re-exported _COLORS / _RST / _BOLD constants.


def _next_scan_after(now: datetime, interval_min: int) -> datetime:
    """Backwards-compatible alias that binds FIRST_SCAN_HOUR/MIN."""
    return _next_scan_after_pure(now, interval_min, FIRST_SCAN_HOUR, FIRST_SCAN_MIN)


def _next_scan_time_str(now_et: datetime, interval_min: int) -> str:
    """Backwards-compatible alias that binds FIRST_SCAN_HOUR/MIN."""
    return _next_scan_time_str_pure(now_et, interval_min, FIRST_SCAN_HOUR, FIRST_SCAN_MIN)


def print_leaderboard(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> None:
    """Backwards-compatible wrapper that binds FIRST_SCAN_HOUR/MIN."""
    _print_leaderboard_pure(results, now_et, total_symbols, passed, elapsed,
                             interval_min, FIRST_SCAN_HOUR, FIRST_SCAN_MIN)


def _format_note_text(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> str:
    """Backwards-compatible wrapper that binds FIRST_SCAN_HOUR/MIN."""
    return _format_note_text_pure(results, now_et, total_symbols, passed, elapsed,
                                    interval_min, FIRST_SCAN_HOUR, FIRST_SCAN_MIN)


def _generate_dashboard(
    results: list[dict],
    df5m_map: dict[str, pd.DataFrame],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> None:
    """Backwards-compatible wrapper — binds this module's intraday_levels,
    DASHBOARD_PATH, and FIRST_SCAN_HOUR/MIN."""
    _generate_dashboard_pure(
        results, df5m_map, now_et, total_symbols, passed, elapsed, interval_min,
        intraday_levels, DASHBOARD_PATH, FIRST_SCAN_HOUR, FIRST_SCAN_MIN,
    )












# ── Apple Notes Integration ───────────────────────────────────────────────────

def update_apple_note(content: str, note_name: str = "Live Scanner") -> None:
    """Write leaderboard to an Apple Note via AppleScript (overwrites each cycle)."""
    escaped = (
        content
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )
    script = f'''
tell application "Notes"
    set targetNote to missing value
    repeat with n in notes of default account
        if name of n is "{note_name}" then
            set targetNote to n
            exit repeat
        end if
    end repeat
    if targetNote is missing value then
        make new note at default account with properties {{name:"{note_name}", body:"{escaped}"}}
    else
        set body of targetNote to "{escaped}"
    end if
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(f"Apple Notes update failed (rc={result.returncode}): {err}")
        else:
            logger.debug("Apple Notes updated.")
    except subprocess.TimeoutExpired:
        logger.warning("Apple Notes update timed out (10s).")
    except Exception as e:
        logger.warning(f"Apple Notes update error: {e}")

# ── Alerts ────────────────────────────────────────────────────────────────────

def fire_alert(result: dict) -> None:
    sig     = result.get("signal", "?")
    ticker  = result.get("ticker", "?")
    urg     = result.get("urgency", 0.0)
    unc     = result.get("uncertainty", 0.0)
    day     = result.get("day_type", "?")
    summary = result.get("summary", "")
    warn    = result.get("day_type_warning", "")

    color = _COLORS.get(sig, "")
    print(f"\n{_BOLD}{'─'*60}{_RST}")
    print(f"{_BOLD}ALERT: {color}{sig}{_RST}{_BOLD} → {ticker}{_RST}")
    print(f"  Urgency: {urg}  |  Uncertainty: {unc}  |  Day type: {day}")
    print(f"  {summary}")
    if warn and warn.lower() not in ("", "none"):
        print(f"  ⚠  {warn}")
    print(f"{_BOLD}{'─'*60}{_RST}\n")

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return
    try:
        emoji = "🟢" if "BUY" in sig else "🔴"
        payload = {
            "content": (
                f"{emoji} **{sig}** → **{ticker}**\n"
                f"Urgency: `{urg}` | Uncertainty: `{unc}` | Day type: `{day}`\n"
                f"{summary}"
                + (f"\n⚠️ {warn}" if warn and warn.lower() not in ("", "none") else "")
            )
        }
        resp = requests.post(webhook, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            logger.warning(f"Discord webhook returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Discord alert failed: {e}")

# ── Scan Scheduler Thread ─────────────────────────────────────────────────────

def scan_thread_func(args, test_mode: bool = False) -> None:
    """Background thread: schedule and run scan cycles."""
    if test_mode:
        logger.info("TEST MODE: accumulating bars for 2 minutes…")
        for _ in range(120):
            if stop_event.is_set():
                return
            time.sleep(1)
        if not stop_event.is_set():
            logger.info("TEST MODE: running scan now…")
            run_scan(args)
        stop_event.set()
        return

    today_et = datetime.now(ET)
    shutdown_dt = today_et.replace(
        hour=SHUTDOWN_HOUR, minute=SHUTDOWN_MIN, second=0, microsecond=0
    )

    while not stop_event.is_set():
        now = datetime.now(ET)
        if now >= shutdown_dt:
            logger.info("Shutdown time reached — stopping scan scheduler.")
            stop_event.set()
            break

        next_scan = _next_scan_after(now, args.scan_interval)
        if next_scan >= shutdown_dt:
            logger.info("No further scans before 4:05 PM ET shutdown.")
            break

        wait_s = max(0.0, (next_scan - now).total_seconds())
        logger.info(f"Next scan at {next_scan.strftime('%H:%M:%S')} ET (in {wait_s:.0f}s)")

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))

        if not stop_event.is_set():
            # Skip scan if outside market hours (9:25 AM - 4:05 PM ET)
            now_et = datetime.now(ET)
            if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 25):
                logger.info("Before market open — sleeping 60s…")
                time.sleep(60)
                continue
            elif now_et.hour >= 16 and (now_et.hour > 16 or now_et.minute >= 5):
                logger.info("After market close — skipping scan…")
                time.sleep(60)
                continue
            run_scan(args)

# ── Live Stream Thread ────────────────────────────────────────────────────────

def stream_thread_func(client: db.Live) -> None:
    """Background thread: pull records from the Databento Live feed."""
    try:
        logger.info("Stream thread started — waiting for bars…")
        for record in client:
            if stop_event.is_set():
                break
            process_record(record)
    except Exception as e:
        if not stop_event.is_set():
            logger.error(f"Live stream error: {e}", exc_info=True)
        stop_event.set()
    finally:
        logger.info("Stream thread exiting.")

# ── Shutdown / Save ───────────────────────────────────────────────────────────

def save_final_results() -> None:
    out_path = LOG_DIR / f"{today_str}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(scan_results_history, f, indent=2, default=str)
        logger.info(f"Final scan results saved → {out_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def save_session_data(bars_dict: dict, last_results: list) -> None:
    """Pickle all accumulated 1-min bars + last scan results for offline replay."""
    pkl_path = LOG_DIR / f"{today_str}_session.pkl"
    try:
        data = {
            "bars": bars_dict,
            "last_results": last_results,
            "prior_closes": prior_closes,
            "daily_atrs": daily_atrs,
            "intraday_levels": intraday_levels,
            "timestamp": datetime.now(ET).isoformat(),
            "date": today_str,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = pkl_path.stat().st_size / 1_048_576
        logger.info(f"Session bars saved → {pkl_path} ({size_mb:.1f} MB, {len(bars_dict):,} symbols)")
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")


# ── Replay Mode ───────────────────────────────────────────────────────────────

def _replay_session(date_str: str, top_n: int = 20) -> None:
    """Load a saved session pickle, score all symbols, render charts, open dashboard."""
    pkl_path = LOG_DIR / f"{date_str}_session.pkl"
    if not pkl_path.exists():
        print(f"No session data found: {pkl_path}")
        sys.exit(1)

    print(f"Loading session data from {pkl_path}…")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    saved_bars: dict    = data["bars"]
    saved_closes: dict  = data.get("prior_closes", {})
    saved_atrs: dict    = data.get("daily_atrs", {})
    saved_levels: dict  = data.get("intraday_levels", {})
    saved_ts: str       = data.get("timestamp", date_str)

    print(f"  {len(saved_bars):,} symbols | recorded at {saved_ts}")

    # Reconstruct caches from saved data (fall back to Historical if missing)
    global prior_closes, daily_atrs, intraday_levels
    if saved_closes:
        prior_closes = saved_closes
    if saved_atrs:
        daily_atrs = saved_atrs
    if saved_levels:
        intraday_levels = saved_levels
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not saved_closes or not saved_atrs:
        if api_key:
            fetched_closes, fetched_atrs = fetch_prior_closes(api_key)
            if not saved_closes:
                prior_closes = fetched_closes
            if not saved_atrs:
                daily_atrs = fetched_atrs
    if not saved_levels and api_key:
        try:
            intraday_levels = fetch_intraday_key_levels(api_key)
        except Exception as e:
            print(f"  Intraday key-level fetch failed (non-fatal): {e}")
            intraday_levels = {}

    # Score all symbols (same logic as run_scan but synchronous, no live feed)
    results = []
    df5m_cache: dict[str, pd.DataFrame] = {}
    skipped = 0
    t0 = time.monotonic()

    for sym, bar_list in saved_bars.items():
        try:
            if len(bar_list) < MIN_BARS:
                continue
            df = pd.DataFrame(bar_list)
            # Skip price filter only — dollar-volume filter is intentionally omitted
            # in replay mode because stocks were already liquid-filtered during the
            # live session, and EQUS.MINI records only a fraction of real tape volume.
            avg_price = df["close"].mean()
            if avg_price < MIN_PRICE:
                skipped += 1
                continue
            day_range = df["high"].max() - df["low"].min()
            if avg_price > 0 and (day_range / avg_price) < MIN_RANGE_PCT:
                skipped += 1
                continue
            if "datetime" in df.columns and len(df) >= 2:
                session_minutes = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 60
                expected_bars = max(session_minutes, 1)
                if len(df) / expected_bars < MIN_BAR_COMPLETENESS:
                    skipped += 1
                    continue
            pc = prior_closes.get(sym)
            if pc is None:
                continue
            df5 = resample_to_5min(df)
            if len(df5) < 2:
                continue
            gap_dir = "up" if df.iloc[0]["open"] > pc else "down"
            score = score_gap(df5, prior_close=pc, gap_direction=gap_dir, ticker=sym,
                              daily_atr=daily_atrs.get(sym))
            score["_prior_close"] = pc
            annotate_adr_multiple(score, df, sym)
            results.append(score)
            df5m_cache[sym] = df5
        except Exception as e:
            pass

    results.sort(key=lambda x: -x.get("urgency", 0))
    results = _dedup_etf_families(results)
    results = _compute_movement(results, {})   # first scan — all "—"
    elapsed = time.monotonic() - t0

    print(f"  Scored {len(results)} symbols in {elapsed:.1f}s (skipped {skipped})")

    # Print leaderboard to console
    try:
        replay_dt = datetime.fromisoformat(saved_ts)
    except Exception:
        replay_dt = datetime.now(ET)

    print_leaderboard(results[:top_n], replay_dt, len(saved_bars), len(results), elapsed, 5)

    # Generate dashboard with charts (synchronous — we're not in a hurry)
    chart_candidates = results[:DASHBOARD_CHARTS]
    df5m_for_dash = {
        r["ticker"]: df5m_cache[r["ticker"]]
        for r in chart_candidates
        if r["ticker"] in df5m_cache
    }
    _generate_dashboard(
        chart_candidates, df5m_for_dash,
        replay_dt, len(saved_bars), len(results), elapsed, 5
    )

    print(f"\nDashboard → {DASHBOARD_PATH}")
    subprocess.run(["open", str(DASHBOARD_PATH)], check=False)

# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live US equities scanner — Brooks price action scoring"
    )
    parser.add_argument("--test", action="store_true",
                        help="Dry run: connect, accumulate 2 min, scan once, exit")
    parser.add_argument("--replay", metavar="DATE",
                        help="Replay saved session (e.g. --replay 2026-04-14), generate dashboard and open")
    parser.add_argument("--scan-interval", type=int, default=DEFAULT_SCAN_INTERVAL,
                        metavar="MIN", help=f"Minutes between scans (default: {DEFAULT_SCAN_INTERVAL})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        metavar="N", help=f"Leaderboard rows (default: {DEFAULT_TOP_N})")
    parser.add_argument("--min-urgency", type=float, default=DEFAULT_MIN_URGENCY,
                        metavar="SCORE", help=f"Alert threshold urgency (default: {DEFAULT_MIN_URGENCY})")
    args = parser.parse_args()

    # ── Replay mode (no live feed needed) ────────────────────────────────────
    if args.replay:
        _replay_session(args.replay, top_n=args.top)
        return

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set — check credentials/.env")
        sys.exit(1)

    now_et = datetime.now(ET)
    logger.info(
        f"Live Scanner v1.3 | {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET"
        + (" | TEST MODE" if args.test else "")
    )

    global prior_closes, daily_atrs, intraday_levels
    try:
        prior_closes, daily_atrs = fetch_prior_closes(api_key)
    except Exception as e:
        logger.error(f"Historical fetch error: {e}")
        prior_closes, daily_atrs = {}, {}
        if not prior_closes:
            logger.warning("No prior closes loaded — gap scoring will be degraded.")

    # Intraday key levels (PDH/PDL/ONH/ONL/PMH/PML) — cached once, used in charts
    try:
        intraday_levels = fetch_intraday_key_levels(api_key)
    except Exception as e:
        logger.error(f"Intraday key-level fetch error (non-fatal): {e}", exc_info=True)
        intraday_levels = {}

    # Mid-session bootstrap: prime `bars` with today's intraday 1-min history
    # so the first post-restart scan has enough bars for real Brooks scoring.
    try:
        _backfill_intraday_bars_pure(
            api_key,
            bars=bars,
            bars_lock=bars_lock,
            instrument_map=instrument_map,
            instrument_map_lock=instrument_map_lock,
        )
    except Exception as e:
        logger.error(f"Intraday backfill error (non-fatal): {e}", exc_info=True)

    def _shutdown(signum, frame):
        logger.info(f"Signal {signum} received — shutting down…")
        stop_event.set()

    def _diag_dump(signum, frame):
        """SIGUSR1 handler: dump urgency histogram, bar-count histogram, top-10
        scored with component breakdown. Does NOT affect state or stop scanner.
        """
        try:
            diag_path = LOG_DIR / f"{today_str}_diag.json"
            with bars_lock:
                bar_counts = {sym: len(bl) for sym, bl in bars.items()}
            total_syms = len(bar_counts)
            counts = list(bar_counts.values())
            counts.sort()
            def _pct(p):
                if not counts: return 0
                return counts[min(len(counts)-1, int(len(counts)*p))]
            bar_hist_buckets = {
                "0-1":   sum(1 for c in counts if c <= 1),
                "2-5":   sum(1 for c in counts if 2 <= c <= 5),
                "6-10":  sum(1 for c in counts if 6 <= c <= 10),
                "11-20": sum(1 for c in counts if 11 <= c <= 20),
                "21-50": sum(1 for c in counts if 21 <= c <= 50),
                "50+":   sum(1 for c in counts if c > 50),
            }

            # Re-score all symbols without filters so we can see the urgency
            # distribution across the full pool (not just the filter-passers).
            all_scored = []
            with bars_lock:
                snap = {sym: list(bl) for sym, bl in bars.items()}
            for sym, bar_list in snap.items():
                if len(bar_list) < MIN_BARS:
                    continue
                try:
                    df = pd.DataFrame(bar_list)
                    pc = prior_closes.get(sym)
                    if pc is None:
                        continue
                    df5 = resample_to_5min(df)
                    if len(df5) < 2:
                        continue
                    gap_dir = "up" if df.iloc[0]["open"] > pc else "down"
                    sc = score_gap(df5, prior_close=pc, gap_direction=gap_dir,
                                   ticker=sym, daily_atr=daily_atrs.get(sym))
                    sc["_bars_1m"] = len(bar_list)
                    sc["_bars_5m"] = len(df5)
                    all_scored.append(sc)
                except Exception:
                    pass
            urgencies = [s.get("urgency", 0) for s in all_scored]
            urg_hist = {
                "0":     sum(1 for u in urgencies if u < 1),
                "1-2":   sum(1 for u in urgencies if 1 <= u < 3),
                "3-4":   sum(1 for u in urgencies if 3 <= u < 5),
                "5-6":   sum(1 for u in urgencies if 5 <= u < 7),
                "7+":    sum(1 for u in urgencies if u >= 7),
            }
            all_scored.sort(key=lambda s: -s.get("urgency", 0))
            top10 = []
            for s in all_scored[:10]:
                top10.append({
                    "ticker":       s.get("ticker"),
                    "urgency":      round(s.get("urgency", 0), 2),
                    "uncertainty":  round(s.get("uncertainty", 0), 2),
                    "signal":       s.get("signal"),
                    "bars_1m":      s.get("_bars_1m"),
                    "bars_5m":      s.get("_bars_5m"),
                    "day_type":     s.get("day_type"),
                    "warn":         s.get("day_type_warning"),
                    "components":   s.get("components") or s.get("component_scores")
                                    or {k: v for k, v in s.items()
                                        if k in ("spike","gap","pullback","follow_through","tail","trend")},
                })
            payload = {
                "timestamp_et":       datetime.now(ET).isoformat(),
                "total_symbols_bars": total_syms,
                "bar_count_hist":     bar_hist_buckets,
                "bar_count_pcts":     {"p10": _pct(0.10), "p50": _pct(0.50), "p90": _pct(0.90)},
                "scored_pool_size":   len(all_scored),
                "urgency_hist":       urg_hist,
                "top10":              top10,
                "mapped_iids":        len(instrument_map),
            }
            with open(diag_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Diagnostic dump written → {diag_path}")
        except Exception as e:
            logger.error(f"Diagnostic dump failed: {e}", exc_info=True)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGUSR1, _diag_dump)

    logger.info(f"Connecting to Databento Live ({DATASET} / {SCHEMA})…")
    client = db.Live(key=api_key, reconnect_policy="reconnect")
    client.subscribe(dataset=DATASET, schema=SCHEMA, stype_in="raw_symbol", symbols="ALL_SYMBOLS")
    logger.info("Subscribed to ALL_SYMBOLS — streaming 1-min bars…")

    stream_th = threading.Thread(target=stream_thread_func, args=(client,), daemon=True, name="stream")
    scan_th   = threading.Thread(target=scan_thread_func, args=(args, args.test), daemon=True, name="scanner")
    stream_th.start()
    scan_th.start()

    today_et = datetime.now(ET)
    shutdown_dt = today_et.replace(hour=SHUTDOWN_HOUR, minute=SHUTDOWN_MIN, second=0, microsecond=0)

    try:
        while not stop_event.is_set():
            now = datetime.now(ET)
            if now >= shutdown_dt and not args.test:
                logger.info("4:05 PM ET — initiating clean shutdown.")
                stop_event.set()
                break
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
        stop_event.set()

    stop_event.set()
    try:
        client.stop()
    except Exception:
        pass

    scan_th.join(timeout=15)
    stream_th.join(timeout=15)

    n_syms = len(bars)
    n_bars = sum(len(v) for v in bars.values())
    logger.info(f"Shutdown complete. Tracked {n_syms:,} symbols, {n_bars:,} total 1-min bars.")
    save_final_results()
    save_session_data(bars, scan_results_history[-1]["results"] if scan_results_history else [])

    # Pattern Lab: expire any pending detections that didn't get full outcome tracking
    try:
        from shared.pattern_lab import expire_pending
        expired = expire_pending(today_str)
        if expired:
            logger.info(f"Pattern Lab: expired {expired} pending detections at shutdown")
    except Exception as e:
        logger.debug(f"Pattern Lab expire error (non-fatal): {e}")

    # Pattern Lab: push final stats to aiedge.trade dashboard
    try:
        from pattern_lab_api import push_to_dashboard
        push_to_dashboard("https://www.aiedge.trade/api/patterns")
        logger.info("Pattern Lab: pushed stats to aiedge.trade")
    except Exception as e:
        logger.debug(f"Pattern Lab push error (non-fatal): {e}")

    logger.info("Exiting.")


if __name__ == "__main__":
    main()
