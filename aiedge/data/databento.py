"""Databento Historical API fetchers for the live scanner.

All functions that hit the Historical REST API live here. They are
pure from-scratch queries — they never touch the live stream or any
scanner runtime state except `backfill_intraday_bars`, which
explicitly takes the two shared dicts + their locks as parameters.

Extracted from live_scanner.py (Phase 4a).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import date, datetime, timedelta
from functools import wraps

import databento as db
import pandas as pd
import pytz

from aiedge.data.resample import SCAN_BAR_SCHEMA

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
DATASET = "EQUS.MINI"
SCHEMA = SCAN_BAR_SCHEMA       # "ohlcv-1m"


# ── Timeout decorator (Unix-signal based) ────────────────────────────

def _timeout_handler(signum, frame):
    """SIGALRM handler — raises TimeoutError when a fetcher hangs."""
    raise TimeoutError("Historical API fetch timeout (30s exceeded)")


def with_timeout(seconds: int):
    """Decorator to enforce a time limit on a function (Unix signals)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


# ── Trading-calendar helper ──────────────────────────────────────────

def _prev_trading_days(n: int) -> date:
    """Return the date N trading days ago (Mon-Fri, no holiday adjustment)."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


# ── Prior-close + 20-day ADR fetch ───────────────────────────────────

def fetch_prior_closes(api_key: str) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch prior-day closes AND 20-day ADR for all symbols.

    Returns (closes, daily_atrs) — both keyed by raw symbol string.

    Pulls 30 calendar days (~20 trading days) of ohlcv-1d in a single query:
      - prior close  = last bar's close price
      - daily_atr    = 20-day average of (high - low)
    Symbology resolved via a separate symbology.resolve() call.
    """
    logger.info("Fetching prior closes + 20-day ADR from Databento Historical API…")
    hist = db.Historical(key=api_key)

    yesterday = _prev_trading_days(1)
    thirty_ago = yesterday - timedelta(days=30)   # ~20 trading days
    end_date = yesterday + timedelta(days=1)      # exclusive end

    try:
        store = hist.timeseries.get_range(
            dataset=DATASET,
            schema="ohlcv-1d",
            symbols="ALL_SYMBOLS",
            stype_in="raw_symbol",
            start=thirty_ago.isoformat(),
            end=end_date.isoformat(),
        )
        df = store.to_df()

        if df.empty:
            logger.warning("Daily-bar query returned no rows.")
            return {}, {}

        df = df[df["close"] > 0]

        # Prior close = last close per instrument_id
        id_to_close: dict[int, float] = (
            df.groupby("instrument_id")["close"].last().to_dict()
        )

        # Daily ATR = mean(high - low) over last 20 rows per instrument_id
        df["_range"] = df["high"] - df["low"]
        id_to_adr: dict[int, float] = (
            df.groupby("instrument_id")["_range"]
            .apply(lambda s: float(s.tail(20).mean()))
            .to_dict()
        )

        logger.info(
            f"Got closes for {len(id_to_close):,} instrument_ids | "
            f"ADR for {len(id_to_adr):,} instrument_ids."
        )

        # Resolve instrument_id → raw_symbol
        sym_result = hist.symbology.resolve(
            dataset=DATASET,
            symbols="ALL_SYMBOLS",
            stype_in="instrument_id",
            stype_out="raw_symbol",
            start_date=thirty_ago,
            end_date=end_date,
        )

        id_to_sym: dict[int, str] = {}
        for iid_str, entries in sym_result.get("result", {}).items():
            if entries:
                sym = entries[-1].get("s")
                if sym:
                    id_to_sym[int(iid_str)] = sym

        logger.info(f"Resolved symbols for {len(id_to_sym):,} instrument_ids.")

        closes: dict[str, float] = {}
        adr_by_sym: dict[str, float] = {}
        for iid, close_px in id_to_close.items():
            sym = id_to_sym.get(iid)
            if sym:
                closes[sym] = close_px
                adr_by_sym[sym] = id_to_adr.get(iid, 0.0)

        logger.info(
            f"Prior closes: {len(closes):,} symbols | "
            f"ADR: {sum(1 for v in adr_by_sym.values() if v > 0):,} symbols"
        )
        return closes, adr_by_sym

    except Exception as e:
        logger.error(f"Failed to fetch daily data: {e}", exc_info=True)
        return {}, {}


# ── 1-min range fetcher (used by backfill + levels) ──────────────────

def _fetch_ohlcv1m_range(api_key: str, start_et: datetime, end_et: datetime,
                         label: str) -> pd.DataFrame | None:
    """Fetch ALL_SYMBOLS ohlcv-1m over [start_et, end_et).

    Returns df with instrument_id + datetime(ET) + open/high/low/close, or
    None on failure. Retries with deeper lag if Historical hasn't
    materialized the window yet.
    """
    hist = db.Historical(key=api_key)
    start_utc = start_et.astimezone(pytz.UTC)

    df = None
    last_err = None
    for extra_lag in (0, 5, 10, 15):
        attempt_end = end_et - timedelta(minutes=extra_lag)
        if attempt_end <= start_et:
            break
        try:
            store = hist.timeseries.get_range(
                dataset=DATASET,
                schema=SCHEMA,
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                start=start_utc.isoformat(),
                end=attempt_end.astimezone(pytz.UTC).isoformat(),
            )
            df = store.to_df()
            break
        except Exception as e:
            last_err = e
            if "data_end_after_available_end" in str(e) or "422" in str(e):
                logger.warning(f"{label}: not yet materialized (+{extra_lag}m) — retrying.")
                continue
            logger.error(f"{label} query failed: {e}")
            return None

    if df is None or df.empty:
        logger.warning(f"{label}: no rows returned (last err: {last_err}).")
        return None

    df = df[df["close"] > 0]
    if df.index.name in (None, "ts_event"):
        df = df.reset_index().rename(columns={df.index.name or "index": "ts_event"})
    if "ts_event" not in df.columns:
        df["ts_event"] = df.index
    df["datetime"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(ET)
    return df


# ── Intraday backfill (mutates caller-provided state) ────────────────

def backfill_intraday_bars(
    api_key: str,
    bars: dict[str, list[dict]],
    bars_lock: threading.Lock,
    instrument_map: dict[int, str],
    instrument_map_lock: threading.Lock,
) -> int:
    """Fetch today's 1-min bars (09:30 ET → now) and prime `bars` so
    scoring has history immediately on the first scan after a mid-
    session restart.

    One Historical call covers the entire ALL_SYMBOLS universe — Databento
    returns a single record set that we groupby(instrument_id) to rebuild
    per-ticker bar lists in the exact dict-shape the live handler uses.

    The `bars`, `bars_lock`, `instrument_map`, `instrument_map_lock`
    arguments are the caller's shared mutable state; this function
    mutates them in place under the supplied locks and returns the
    count of symbols primed with ≥1 bar.
    """
    now_et = datetime.now(ET)
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    # Nothing to backfill before the open or after close
    if now_et <= session_open:
        logger.info("Before regular-session open — skipping intraday backfill.")
        return 0
    end_cap = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    query_end = min(now_et, end_cap)

    # Historical ohlcv-1m publishes with a materialization lag (~minutes)
    # behind the Live feed. Querying up to "now" triggers 422
    # data_end_after_available_end. Pull up to ~10 min ago to stay inside the
    # Historical window; the Live stream will backfill the gap as bars close.
    HIST_LAG = timedelta(minutes=10)
    query_end = (query_end - HIST_LAG).replace(second=0, microsecond=0)
    if query_end <= session_open:
        logger.info(
            f"Session too young for intraday backfill "
            f"({now_et.strftime('%H:%M')} ET, need ≥ 09:40 ET) — skipping."
        )
        return 0

    start_utc = session_open.astimezone(pytz.UTC)

    logger.info(
        f"Backfilling intraday 1-min bars {session_open.strftime('%H:%M')} → "
        f"{query_end.strftime('%H:%M')} ET via Historical API…"
    )
    t0 = time.monotonic()
    hist = db.Historical(key=api_key)

    # Retry with progressively deeper lag if Historical claims the window
    # isn't materialized yet (422 data_end_after_available_end).
    df = None
    last_err = None
    for extra_lag in (0, 5, 10, 15):
        attempt_end_et = query_end - timedelta(minutes=extra_lag)
        if attempt_end_et <= session_open:
            break
        attempt_end_utc = attempt_end_et.astimezone(pytz.UTC)
        try:
            store = hist.timeseries.get_range(
                dataset=DATASET,
                schema=SCHEMA,
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                start=start_utc.isoformat(),
                end=attempt_end_utc.isoformat(),
            )
            df = store.to_df()
            query_end = attempt_end_et
            if extra_lag:
                logger.info(f"Backfill succeeded after extra {extra_lag}-min lag.")
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            if "data_end_after_available_end" in msg or "422" in msg:
                logger.warning(
                    f"Historical window not yet materialized (+{extra_lag}m) — retrying with more lag."
                )
                continue
            logger.error(f"Intraday backfill query failed: {e}", exc_info=True)
            return 0

    if df is None:
        logger.error(f"Intraday backfill gave up after retries. Last error: {last_err}")
        return 0

    if df.empty:
        logger.warning("Intraday backfill returned no rows.")
        return 0

    # Resolve instrument_id → raw_symbol for exactly this session
    try:
        sym_result = hist.symbology.resolve(
            dataset=DATASET,
            symbols="ALL_SYMBOLS",
            stype_in="instrument_id",
            stype_out="raw_symbol",
            start_date=session_open.date(),
            end_date=(session_open + timedelta(days=1)).date(),
        )
    except Exception as e:
        logger.error(f"Symbology resolve for backfill failed: {e}", exc_info=True)
        return 0

    id_to_sym: dict[int, str] = {}
    for iid_str, entries in sym_result.get("result", {}).items():
        if entries:
            sym = entries[-1].get("s")
            if sym:
                id_to_sym[int(iid_str)] = sym

    # Drop zero-close rows (dead tickers or pre-open phantoms)
    df = df[df["close"] > 0]

    # df.index is ts_event (UTC). Build the datetime column in ET to match live.
    if df.index.name in (None, "ts_event"):
        df = df.reset_index().rename(columns={df.index.name or "index": "ts_event"})
    if "ts_event" not in df.columns:
        df["ts_event"] = df.index

    df["datetime"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(ET)
    # Seed instrument_map AND bars together under their respective locks
    primed = 0
    total_bars = 0
    with instrument_map_lock, bars_lock:
        # Seed the instrument_map so any live bars that arrive during/after
        # the backfill can attribute correctly even if the Live SymbolMappingMsg
        # hasn't been processed yet (race with the stream thread starting).
        for iid, sym in id_to_sym.items():
            instrument_map[iid] = sym

        # Build bars[sym] in chronological order per iid
        df_sorted = df.sort_values(["instrument_id", "datetime"])
        cols = ["instrument_id", "datetime", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df_sorted.columns]
        if missing:
            logger.error(f"Backfill df missing columns: {missing} — have {list(df_sorted.columns)}")
            return 0

        for iid, grp in df_sorted.groupby("instrument_id", sort=False):
            sym = id_to_sym.get(int(iid))
            if not sym:
                continue
            bar_list = bars.setdefault(sym, [])
            for _, row in grp.iterrows():
                bar_list.append({
                    "datetime": row["datetime"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                })
            if bar_list:
                primed += 1
                total_bars += len(grp)

    elapsed = time.monotonic() - t0
    logger.info(
        f"Intraday backfill done in {elapsed:.1f}s — "
        f"primed {primed:,} symbols with {total_bars:,} one-min bars "
        f"(instrument_map seeded with {len(id_to_sym):,} iids)."
    )
    return primed
