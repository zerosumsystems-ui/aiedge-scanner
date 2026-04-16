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

import databento as db

# ── Constants ─────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
DATASET = "EQUS.MINI"
SCHEMA = "ohlcv-1m"

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

# ── ETF Family Dedup ──────────────────────────────────────────────────────────
# Many index ETFs move as a single "story": on a sharp NDX up-move, QQQ, TQQQ,
# QLD, and SQQQ (inverse) all rank high together and clutter the top of the
# dashboard with three or four tickers telling the same tape. This map lets us
# collapse each family down to a single "family leader" (the highest-urgency
# member surfaced this scan) and suppress the rest.
#
# Extend: add a new family name (key) whose value is the list of tickers that
# trade the same underlying. Non-ETF equities are *not* in any family and pass
# through the dedup untouched. Case-sensitive — use upper-case tickers.
ETF_FAMILIES: dict[str, list[str]] = {
    "SPX":       ["SPY", "IVV", "VOO", "SPLG", "SSO", "UPRO", "SPUU",
                  "SPXL", "SPXU", "SPXS", "SPDN", "SDS", "SH"],
    "NDX":       ["QQQ", "QQQM", "TQQQ", "SQQQ", "QLD", "QID", "PSQ"],
    "DJIA":      ["DIA", "UDOW", "SDOW", "DDM", "DXD", "DOG"],
    "RUSSELL":   ["IWM", "VTWO", "UWM", "TNA", "TZA", "TWM", "SRTY", "RWM"],
    "TECH":      ["XLK", "VGT", "TECL", "TECS", "ROM", "REW"],
    "GOLD":      ["GLD", "IAU", "GLDM", "BAR", "OUNZ", "UGL", "DGP", "GLL", "DGLD"],
    "SILVER":    ["SLV", "SIVR", "AGQ", "ZSL"],
    "OIL":       ["USO", "BNO", "UCO", "SCO", "OILU", "OILD"],
    "NATGAS":    ["UNG", "BOIL", "KOLD"],
    "SEMIS":     ["SMH", "SOXX", "SOXL", "SOXS", "USD", "SSG"],
    "BONDS20":   ["TLT", "TMF", "TMV", "TBT", "TBF"],
    "VIX":       ["VXX", "VIXY", "VIXM", "UVXY", "SVXY", "UVIX", "SVIX"],
    "CHINA":     ["FXI", "MCHI", "KWEB", "YINN", "YANG", "FXP", "CHAU", "CHAD"],
    "EMERGING":  ["EEM", "IEMG", "VWO", "EDC", "EDZ"],
    "FIN":       ["XLF", "VFH", "FAS", "FAZ", "UYG", "SKF"],
    "ENERGY":    ["XLE", "VDE", "ERX", "ERY", "GUSH", "DRIP"],
    "BIOTECH":   ["XBI", "IBB", "LABU", "LABD"],
    "REIT":      ["IYR", "VNQ", "XLRE", "URE", "DRN", "DRV", "SRS"],
    "HEALTHCARE":["XLV", "VHT", "CURE", "RXD"],
    "STAPLES":   ["XLP", "VDC"],
    "DISCR":     ["XLY", "VCR"],
    "UTILS":     ["XLU", "VPU"],
    "INDUSTR":   ["XLI", "VIS"],
    "DEFENSE":   ["ITA", "DFEN"],
    "HOMEBUILD": ["ITB", "XHB", "NAIL", "CLAW"],
    "REGBANK":   ["KRE", "KBWB", "DPST", "WDRW"],
    "CLEANENRG": ["ICLN", "TAN", "FAN", "PBD"],
    "RETAIL":    ["XRT", "RETL"],
    "TRANSPORT": ["IYT", "XTN"],
    "BITCOIN":   ["BITO", "BTF", "BITI", "BITX", "IBIT", "FBTC", "ARKB",
                  "HODL", "BRRR", "GBTC", "BITB"],
    "ETHER":     ["ETHA", "ETHE", "FETH", "ETHV", "ETH", "ETHU", "ETHD"],
}
# Reverse lookup built once at import: ticker -> family name
_TICKER_TO_FAMILY: dict[str, str] = {
    t: fam for fam, tickers in ETF_FAMILIES.items() for t in tickers
}

# Dual-class / same-company pairs. Treated identically to ETF families in dedup:
# same family key → higher-urgency wins, siblings suppressed with badge tooltip.
# The two share classes of the same underlying business trade as one story.
SAME_COMPANY: dict[str, list[str]] = {
    "GOOG_FAMILY":  ["GOOG", "GOOGL"],
    "BRK_FAMILY":   ["BRK.A", "BRK.B", "BRK-A", "BRK-B"],   # include both Databento and canonical forms
    "FOX_FAMILY":   ["FOX", "FOXA"],
    "NWS_FAMILY":   ["NWS", "NWSA"],
    "HEI_FAMILY":   ["HEI", "HEI.A", "HEI-A"],
    "MKC_FAMILY":   ["MKC", "MKC.V", "MKC-V"],
    "LEN_FAMILY":   ["LEN", "LEN.B", "LEN-B"],
    "PRA_FAMILY":   ["PRA", "PRAA"],
}
# Merge into the main family index so _dedup_etf_families() handles these too.
for _fam, _tickers in SAME_COMPANY.items():
    ETF_FAMILIES[_fam] = _tickers
    for _t in _tickers:
        _TICKER_TO_FAMILY[_t] = _fam

# ── Prior Closes ──────────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Historical API fetch timeout (30s exceeded)")

def with_timeout(seconds):
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

def _prev_trading_days(n: int) -> date:
    """Return the date N trading days ago (Mon–Fri, no holiday adjustment)."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


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

    yesterday  = _prev_trading_days(1)
    thirty_ago = yesterday - timedelta(days=30)   # ~20 trading days
    end_date   = yesterday + timedelta(days=1)    # exclusive end

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

        closes:     dict[str, float] = {}
        adr_by_sym: dict[str, float] = {}
        for iid, close_px in id_to_close.items():
            sym = id_to_sym.get(iid)
            if sym:
                closes[sym]     = close_px
                adr_by_sym[sym] = id_to_adr.get(iid, 0.0)

        logger.info(
            f"Prior closes: {len(closes):,} symbols | "
            f"ADR: {sum(1 for v in adr_by_sym.values() if v > 0):,} symbols"
        )
        return closes, adr_by_sym

    except Exception as e:
        logger.error(f"Failed to fetch daily data: {e}", exc_info=True)
        return {}, {}


def backfill_intraday_bars(api_key: str) -> int:
    """Fetch today's 1-min bars (09:30 ET → now) and prime the in-memory
    `bars` dict so scoring has history immediately on the first scan after
    a mid-session restart.

    One Historical call covers the entire ALL_SYMBOLS universe — Databento
    returns a single record set that we groupby(instrument_id) to rebuild
    per-ticker bar lists in the exact dict-shape the live handler uses.

    Returns the count of symbols primed with ≥1 bar.
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
    end_utc   = query_end.astimezone(pytz.UTC)

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
            end_utc = attempt_end_utc
            query_end = attempt_end_et
            if extra_lag:
                logger.info(f"Backfill succeeded after extra {extra_lag}-min lag.")
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            if "data_end_after_available_end" in msg or "422" in msg:
                logger.warning(f"Historical window not yet materialized (+{extra_lag}m) — retrying with more lag.")
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
        # to_df() sometimes uses the index directly
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
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
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

def annotate_adr_multiple(score: dict, df_1m: pd.DataFrame, sym: str) -> None:
    """Compute Minervini-style ADR multiple = today's range ÷ 20-day ADR.

    Uses the full-resolution 1-min bars so intraday wick extremes are preserved
    (a 5-min resample would round wicks that print between 5-min anchors).
    The 20-day ADR baseline is fetched once at scanner startup from Databento
    daily bars (see fetch_prior_closes) and cached in daily_atrs — it does NOT
    re-fetch intraday, per the user spec.
    """
    today_high = float(df_1m["high"].max())
    today_low  = float(df_1m["low"].min())
    today_range = max(today_high - today_low, 0.0)
    adr_20 = daily_atrs.get(sym, 0.0) or 0.0
    adr_mult = (today_range / adr_20) if adr_20 > 0 else 0.0
    score["today_range"]  = round(today_range, 4)
    score["adr_20"]       = round(adr_20, 4)
    score["adr_multiple"] = round(adr_mult, 2)


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 1-min OHLCV DataFrame to 5-min bars."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.set_index("datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)

    df5 = (
        df.resample("5min", label="left", closed="left")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    # Forward-fill NaN rows that result from gaps in intraday 1-min feed
    # (e.g., gaps during very low volume periods like 12:25-12:40).
    # ffill preserves the last known OHLC for gaps < ~1hr, visual continuity on charts.
    df5["open"] = df5["open"].ffill()
    df5["high"] = df5["high"].ffill()
    df5["low"] = df5["low"].ffill()
    df5["close"] = df5["close"].ffill()
    df5["volume"] = df5["volume"].fillna(0)  # NaN volume → 0 (no trades)
    df5 = df5.dropna(subset=["open", "close"])
    df5 = df5[df5["open"] > 0]
    return df5.reset_index().rename(columns={"datetime": "datetime"})

# ── Chart Rendering ───────────────────────────────────────────────────────────

def render_chart_base64(df_5m: pd.DataFrame, ticker: str, prior_close: float,
                        adr_multiple: float | None = None,
                        levels: dict | None = None) -> str | None:
    """Render a 5-min candlestick chart and return base64-encoded PNG.

    `levels` is an optional per-ticker dict of intraday reference levels:
        {"pdh":..., "pdl":..., "onh":..., "onl":..., "pmh":..., "pml":...}
    Any combination may be missing. The renderer handles in-range vs
    off-screen (corner badges) placement itself — axis is NEVER stretched.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        path = tmp.name

        # Build key_levels dict in the shape chart_renderer expects.
        # Periods: prior_day (PDO/PDH/PDL), overnight (ONH/ONL), premarket (PMH/PML).
        key_levels: dict = {}
        if prior_close and prior_close > 0:
            key_levels.setdefault("prior_day", {})["open"] = prior_close
        if levels:
            pd_e = key_levels.setdefault("prior_day", {})
            if levels.get("pdh") is not None: pd_e["high"] = levels["pdh"]
            if levels.get("pdl") is not None: pd_e["low"]  = levels["pdl"]
            on_e = {}
            if levels.get("onh") is not None: on_e["high"] = levels["onh"]
            if levels.get("onl") is not None: on_e["low"]  = levels["onl"]
            if on_e: key_levels["overnight"] = on_e
            pm_e = {}
            if levels.get("pmh") is not None: pm_e["high"] = levels["pmh"]
            if levels.get("pml") is not None: pm_e["low"]  = levels["pml"]
            if pm_e: key_levels["premarket"] = pm_e

        render_chart(
            ticker=ticker,
            timeframe="5min",
            df=df_5m,
            output_path=path,
            key_levels=key_levels,
            theme="dark_color",
            show_volume=True,
            adr_multiple=adr_multiple,
        )

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(path)
        return b64
    except Exception as e:
        logger.debug(f"Chart render failed for {ticker}: {e}")
        try:
            os.unlink(path)
        except Exception:
            pass
        return None

# ── Intraday Key Levels (PDH/PDL/ONH/ONL/PMH/PML) ─────────────────────────────

def _fetch_ohlcv1m_range(api_key: str, start_et: datetime, end_et: datetime,
                         label: str) -> pd.DataFrame | None:
    """Fetch ALL_SYMBOLS ohlcv-1m over [start_et, end_et). Returns df with
    instrument_id + datetime(ET) + open/high/low/close, or None on failure.
    Retries with deeper lag if Historical hasn't materialized the window yet.
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


def fetch_intraday_key_levels(api_key: str) -> dict[str, dict[str, float]]:
    """Fetch per-symbol intraday reference levels from Databento Historical:

      PDH / PDL  — prior trading day regular session (09:30–16:00 ET) high/low
      ONH / ONL  — today pre-market extended (04:00–09:30 ET) high/low
      PMH / PML  — today narrow pre-market (08:00–09:30 ET) high/low

    PDO (prior day open) is already fetched via `fetch_prior_closes` as the
    prior daily close; kept separate so each cache has a single responsibility.

    Runs once at scanner startup. Any missing level for a symbol is simply
    absent from its dict — callers must use dict.get(...).
    """
    logger.info("Fetching intraday key levels (PDH/PDL/ONH/ONL/PMH/PML)…")
    t0 = time.monotonic()
    hist = db.Historical(key=api_key)

    yesterday = _prev_trading_days(1)
    today     = date.today()

    # Prior-day regular session window
    pd_start = ET.localize(datetime.combine(yesterday, datetime.min.time())
                           ).replace(hour=9, minute=30)
    pd_end   = pd_start.replace(hour=16, minute=0)

    # Today's extended pre-market window (ON = 04:00–09:30, PM = 08:00–09:30)
    on_start = ET.localize(datetime.combine(today, datetime.min.time())
                           ).replace(hour=4, minute=0)
    on_end   = on_start.replace(hour=9, minute=30)

    # Fetch prior-day bars
    df_pd = _fetch_ohlcv1m_range(api_key, pd_start, pd_end, "PriorDay")

    # Fetch pre-market bars — cap end at now if we're mid-session-start
    now_et = datetime.now(ET)
    eff_on_end = on_end if now_et >= on_end else (now_et - timedelta(minutes=10))
    if eff_on_end <= on_start:
        logger.info("Before pre-market materialization — ON/PM levels skipped.")
        df_pm = None
    else:
        df_pm = _fetch_ohlcv1m_range(api_key, on_start, eff_on_end, "PreMarket")

    if df_pd is None and df_pm is None:
        logger.warning("No intraday level data available — charts will have PDO only.")
        return {}

    # Resolve instrument_id → raw_symbol across both windows
    try:
        sym_result = hist.symbology.resolve(
            dataset=DATASET,
            symbols="ALL_SYMBOLS",
            stype_in="instrument_id",
            stype_out="raw_symbol",
            start_date=yesterday,
            end_date=today + timedelta(days=1),
        )
    except Exception as e:
        logger.error(f"Symbology resolve failed for key levels: {e}")
        return {}

    id_to_sym: dict[int, str] = {}
    for iid_str, entries in sym_result.get("result", {}).items():
        if entries:
            sym = entries[-1].get("s")
            if sym:
                id_to_sym[int(iid_str)] = sym

    levels: dict[str, dict[str, float]] = {}

    if df_pd is not None:
        g = df_pd.groupby("instrument_id").agg(pdh=("high", "max"), pdl=("low", "min"))
        for iid, row in g.iterrows():
            sym = id_to_sym.get(iid)
            if sym:
                d = levels.setdefault(sym, {})
                d["pdh"] = float(row["pdh"])
                d["pdl"] = float(row["pdl"])

    if df_pm is not None:
        # ON window = full 04:00–09:30
        g_on = df_pm.groupby("instrument_id").agg(onh=("high", "max"), onl=("low", "min"))
        for iid, row in g_on.iterrows():
            sym = id_to_sym.get(iid)
            if sym:
                d = levels.setdefault(sym, {})
                d["onh"] = float(row["onh"])
                d["onl"] = float(row["onl"])

        # PM narrow window = 08:00–09:30 subset
        from datetime import time as _dtime
        pm_mask = df_pm["datetime"].dt.time >= _dtime(8, 0)
        df_pm_narrow = df_pm[pm_mask]
        if not df_pm_narrow.empty:
            g_pm = df_pm_narrow.groupby("instrument_id").agg(pmh=("high", "max"), pml=("low", "min"))
            for iid, row in g_pm.iterrows():
                sym = id_to_sym.get(iid)
                if sym:
                    d = levels.setdefault(sym, {})
                    d["pmh"] = float(row["pmh"])
                    d["pml"] = float(row["pml"])

    logger.info(f"Intraday key levels: {len(levels):,} symbols in {time.monotonic()-t0:.1f}s")
    return levels


# ── ETF Family Dedup ──────────────────────────────────────────────────────────

def _dedup_etf_families(results: list[dict]) -> list[dict]:
    """Collapse ETF-family repeats in a ranked list.

    `results` must already be sorted by urgency desc. For each family that
    appears, keep the single highest-urgency ticker (the "family leader") and
    drop the rest. The leader is annotated with:

        r["family"]               family name (e.g. "NDX")
        r["family_leader"]        True
        r["family_siblings"]      list of suppressed tickers (in urgency order)
        r["family_sibling_count"] len(siblings)

    Non-ETF tickers (not in ETF_FAMILIES) pass through unchanged — this is
    intentionally a pure map-based filter, no asset-class lookup needed.
    """
    seen: set[str] = set()
    out: list[dict] = []
    siblings_by_fam: dict[str, list[str]] = {}

    for r in results:
        tkr = r.get("ticker", "")
        fam = _TICKER_TO_FAMILY.get(tkr)
        if fam is None:
            out.append(r)
            continue
        if fam not in seen:
            seen.add(fam)
            r["family"] = fam
            r["family_leader"] = True
            out.append(r)
        else:
            siblings_by_fam.setdefault(fam, []).append(tkr)

    for r in out:
        fam = r.get("family")
        if fam and r.get("family_leader"):
            sibs = siblings_by_fam.get(fam, [])
            r["family_siblings"]      = sibs
            r["family_sibling_count"] = len(sibs)
    return out


# ── Rank Movement ─────────────────────────────────────────────────────────────

def _compute_movement(results: list[dict], prior: dict) -> list[dict]:
    """Annotate each result with rank, rank_change, and urgency_delta vs prior scan."""
    first_scan = len(prior) == 0
    for i, r in enumerate(results):
        rank = i + 1
        ticker = r["ticker"]
        r["rank"] = rank
        if first_scan:
            r["prev_rank"] = None
            r["rank_change"] = None
            r["urgency_delta"] = None
            r["_first_scan"] = True
        elif ticker in prior:
            prev_rank = prior[ticker]["rank"]
            r["prev_rank"] = prev_rank
            r["rank_change"] = prev_rank - rank
            r["urgency_delta"] = round(r["urgency"] - prior[ticker]["urgency"], 1)
            r["_first_scan"] = False
        else:
            r["prev_rank"] = None
            r["rank_change"] = None
            r["urgency_delta"] = None
            r["_first_scan"] = False
    return results


def _fmt_movement(r: dict) -> str:
    if r.get("_first_scan"):
        return "—"
    rc = r.get("rank_change")
    if rc is None:
        return "NEW"
    pr = r["prev_rank"]
    if rc == 0:
        return f"was #{pr}  (=)"
    sign = "+" if rc > 0 else ""
    return f"was #{pr}  ({sign}{rc})"


def _fmt_delta(r: dict) -> str:
    if r.get("_first_scan") or r.get("urgency_delta") is None:
        return "—"
    d = r["urgency_delta"]
    sign = "+" if d >= 0 else ""
    return f"U {sign}{d:.1f}"

# ── Pattern Lab Helpers ────────────────────────────────────────────────────────

_PATTERN_LAB_OK = True  # flipped to False on first import failure

def _log_pattern_lab_detections(
    ticker: str, score: dict, bpa_setups: list[dict],
    df5: "pd.DataFrame", scan_time: "datetime",
) -> None:
    """Log BPA detections to the Pattern Lab database."""
    global _PATTERN_LAB_OK
    if not _PATTERN_LAB_OK:
        return
    try:
        from shared.pattern_lab import log_detection
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
        elif setup_type in ("L1", "L2"):
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
        )


def _update_pattern_lab_outcomes(scan_time: "datetime") -> None:
    """Revisit pending detections and fill in outcome data from current bars."""
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
    from collections import defaultdict
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
                _log_pattern_lab_detections(sym, score, _bpa_setups, df5, now_et)

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

# ── Console Output ────────────────────────────────────────────────────────────

_COLORS = {
    "BUY_PULLBACK":  "\033[92m",
    "BUY_SPIKE":     "\033[32m",
    "SELL_PULLBACK": "\033[91m",
    "SELL_SPIKE":    "\033[31m",
    "WAIT":          "\033[93m",
    "FOG":           "\033[33m",
    "AVOID":         "\033[90m",
    "PASS":          "\033[90m",
}
_RST  = "\033[0m"
_BOLD = "\033[1m"
_W    = 100
_UP   = "\033[92m▲\033[0m"
_DOWN = "\033[91m▼\033[0m"


def _movement_arrow(r: dict) -> str:
    if r.get("_first_scan") or r.get("rank_change") is None:
        return " "
    rc = r["rank_change"]
    if rc > 0:
        return _UP
    if rc < 0:
        return _DOWN
    return "="


def _next_scan_after(now: datetime, interval_min: int) -> datetime:
    first = now.replace(
        hour=FIRST_SCAN_HOUR, minute=FIRST_SCAN_MIN, second=0, microsecond=0
    )
    if now < first:
        return first
    elapsed_s = (now - first).total_seconds()
    periods_done = int(elapsed_s / (interval_min * 60))
    return first + timedelta(minutes=interval_min * (periods_done + 1))


def _next_scan_time_str(now_et: datetime, interval_min: int) -> str:
    nxt = _next_scan_after(now_et, interval_min)
    return nxt.strftime("%I:%M %p").lstrip("0")


def print_leaderboard(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> None:
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    title = f" LIVE SCAN — {time_str} ET | {now_et.strftime('%Y-%m-%d')} | {total_symbols:,} symbols "
    print()
    print(_BOLD + "━" * _W + _RST)
    print(_BOLD + title.center(_W) + _RST)
    print(_BOLD + "━" * _W + _RST)

    if not results:
        print("  (no liquid stocks passed filters this cycle)")
    else:
        hdr = (
            f"{'':1}{'#':>3}  {'TICKER':<8}  {'URGENCY':>7}  "
            f"{'ADR':>7}  {'MOVE':>5}  {'UNCERT':>6}  "
            f"{'SIGNAL':<16}  {'MOVEMENT':<18}  DELTA"
        )
        print(_BOLD + hdr + _RST)
        print("─" * _W)

        for r in results:
            sig    = r.get("signal", "?")
            color  = _COLORS.get(sig, "")
            urg    = r.get("urgency", 0.0)
            unc    = r.get("uncertainty", 0.0)
            ticker = r.get("ticker", "?")
            warn   = r.get("day_type_warning", "")
            arrow  = _movement_arrow(r)
            mov    = _fmt_movement(r)
            dlt    = _fmt_delta(r)
            adr_v  = r.get("daily_atr") or 0.0
            rat_v  = r.get("move_ratio") or 0.0
            adr_col = f"${adr_v:.2f}" if adr_v > 0 else "  —  "
            rat_col = f"{rat_v:.1f}x"  if adr_v > 0 else "  — "

            print(
                f"{arrow:1} {r['rank']:>3}  {ticker:<8}  {urg:>7.1f}  "
                f"{adr_col:>7}  {rat_col:>5}  {unc:>6.1f}  "
                f"{color}{sig:<16}{_RST}  {mov:<18}  {dlt}"
            )
            if warn and warn.lower() not in ("", "none") and r["rank"] <= 10:
                print(f"  {'':>3}  {'':8}  {'':7}  {'':7}  {'':5}  {'':6}  {'':16}  ⚠  {warn[:50]}")

    next_str = _next_scan_time_str(now_et, interval_min)
    print("─" * _W)
    print(f"  {passed} stocks passed | scan took {elapsed:.2f}s | next scan {next_str}")
    print()


def _format_note_text(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> str:
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    lines = [
        f"LIVE SCAN — {time_str} ET",
        f"Last updated: {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET",
        f"{total_symbols:,} symbols streaming",
        "━" * 72,
        "",
    ]

    if not results:
        lines.append("(no liquid stocks passed filters this cycle)")
    else:
        hdr = (
            f"{'#':>3}  {'TICKER':<8}  {'URGENCY':>7}  {'UNCERT':>6}  "
            f"{'SIGNAL':<16}  {'MOVEMENT':<18}  DELTA"
        )
        lines.append(hdr)
        lines.append("─" * 72)
        for r in results:
            sig    = r.get("signal", "?")
            urg    = r.get("urgency", 0.0)
            unc    = r.get("uncertainty", 0.0)
            ticker = r.get("ticker", "?")
            warn   = r.get("day_type_warning", "")
            mov    = _fmt_movement(r)
            dlt    = _fmt_delta(r)
            lines.append(
                f"{r['rank']:>3}  {ticker:<8}  {urg:>7.1f}  {unc:>6.1f}  "
                f"{sig:<16}  {mov:<18}  {dlt}"
            )
            if warn and warn.lower() not in ("", "none") and r["rank"] <= 10:
                lines.append(f"     {'':8}  {'':7}  {'':6}  {'':16}  ⚠ {warn[:58]}")

    next_str = _next_scan_time_str(now_et, interval_min)
    lines += [
        "─" * 72,
        f"{passed} stocks passed | scan took {elapsed:.2f}s | next scan {next_str}",
    ]
    return "\n".join(lines)

# ── HTML Dashboard ────────────────────────────────────────────────────────────

# Signal badge colors (CSS)
_SIG_CSS = {
    "BUY_PULLBACK":  ("bg-buy",   "BUY PULLBACK"),
    "BUY_SPIKE":     ("bg-buy",   "BUY SPIKE"),
    "SELL_PULLBACK": ("bg-sell",  "SELL PULLBACK"),
    "SELL_SPIKE":    ("bg-sell",  "SELL SPIKE"),
    "SELL_PULLBACK_INTRADAY": ("bg-sell", "SELL FLIP"),
    "BUY_PULLBACK_INTRADAY":  ("bg-buy",  "BUY FLIP"),
    "WAIT":          ("bg-wait",  "WAIT"),
    "FOG":           ("bg-fog",   "FOG"),
    "AVOID":         ("bg-avoid", "AVOID"),
    "PASS":          ("bg-avoid", "PASS"),
}

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta http-equiv="refresh" content="300">
<title>Live Scanner</title>
<style>
  :root {
    --bg:       #1a1a1a;
    --surface:  #242424;
    --border:   #333;
    --text:     #e0e0e0;
    --sub:      #888;
    --teal:     #00c896;
    --red:      #e05555;
    --yellow:   #f5c842;
    --orange:   #f5a623;
    --gray:     #555;
    --radius:   8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 14px;
    padding: 12px;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
  header .meta { font-size: 12px; color: var(--sub); text-align: right; }
  .stats {
    display: flex; gap: 16px;
    font-size: 12px; color: var(--sub);
    margin-bottom: 12px;
  }
  .stats span { white-space: nowrap; }

  /* Stock card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 8px;
    overflow: hidden;
  }
  .card summary {
    list-style: none;
    cursor: pointer;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: 28px 62px 1fr 86px auto;
    align-items: center;
    gap: 8px;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  .card summary::-webkit-details-marker { display: none; }
  .card[open] summary { border-bottom: 1px solid var(--border); }

  .rank {
    font-size: 12px; color: var(--sub);
    text-align: center; line-height: 1;
  }
  .rank .arrow { font-size: 10px; display: block; }
  .ticker-block { min-width: 0; }
  .ticker { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
  .day-type { font-size: 11px; color: var(--sub); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cycle-badge {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 9.5px;
    letter-spacing: 0.3px;
    color: var(--teal);
    background: rgba(0,200,150,.06);
  }

  /* ADR block (fills the 1fr spacer column in the card grid) */
  .adr-block { min-width: 0; padding-left: 4px; margin-right: 12px; }
  .adr-val { font-size: 12px; color: var(--sub); white-space: nowrap; }
  .adr-ratio { font-size: 11px; color: var(--sub); opacity: 0.7; white-space: nowrap; }

  /* ADR-multiple column — Minervini-style range-expansion tile */
  .adr-mult {
    text-align: center;
    padding: 6px 8px;
    border-radius: 6px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    line-height: 1.15;
    min-width: 50px;
  }
  .adr-mult .mult-val { font-size: 15px; font-weight: 700; letter-spacing: -0.2px; }
  .adr-mult .mult-lbl { font-size: 9px; opacity: .65; letter-spacing: 0.6px; }
  .adr-mult.tier-cold { color: var(--sub); background: transparent; }
  .adr-mult.tier-warm { color: var(--teal); background: rgba(0,200,150,.10); border: 1px solid rgba(0,200,150,.25); }
  .adr-mult.tier-hot  {
    color: #fff; background: rgba(0,200,150,.35);
    border: 1px solid var(--teal);
    box-shadow: 0 0 0 1px rgba(0,200,150,.20);
  }
  .adr-mult.tier-extreme {
    color: #fff; background: var(--teal);
    border: 1px solid var(--teal);
    box-shadow: 0 0 12px rgba(0,200,150,.40);
  }

  /* Sort toolbar */
  .sort-bar {
    display: flex; gap: 6px; align-items: center;
    margin: 4px 0 10px;
    font-size: 11px; color: var(--sub);
  }
  .sort-bar button {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 4px 10px; border-radius: 4px;
    font: inherit; cursor: pointer;
  }
  .sort-bar button.active { background: rgba(0,200,150,.18); border-color: var(--teal); color: var(--teal); }
  .sort-bar button:hover { border-color: var(--teal); }

  /* Scoring legend */
  .legend { margin: 0 0 12px; font-size: 12px; color: var(--sub); }
  .legend summary {
    cursor: pointer; color: var(--teal); font-size: 11px; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase; padding: 4px 0;
  }
  .legend summary:hover { text-decoration: underline; }
  .legend-body {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; padding: 12px 0 8px;
  }
  .legend-section h3 {
    font-size: 11px; font-weight: 700; color: var(--text);
    text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 6px;
    border-bottom: 1px solid var(--border); padding-bottom: 4px;
  }
  .legend-section dl { margin: 0; }
  .legend-section dt {
    font-weight: 600; color: var(--text); font-size: 11px; margin-top: 6px;
    font-family: 'SF Mono', 'Menlo', monospace;
  }
  .legend-section dd { margin: 2px 0 0 0; font-size: 11px; line-height: 1.4; color: var(--sub); }

  .scores {
    display: flex; flex-direction: column; align-items: flex-end; gap: 8px;
  }
  .score-row { display: flex; gap: 6px; align-items: center; }
  .score-label { font-size: 11px; color: var(--sub); width: 24px; text-align: right; }
  .bar-track {
    width: 80px; height: 5px;
    background: var(--border); border-radius: 3px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-urg  { background: var(--teal); }
  .bar-unc  { background: var(--red); }
  .score-val { font-size: 12px; font-weight: 600; width: 28px; text-align: right; }

  /* Component subscore strip (under URG/UNC bars) */
  .comp-strip {
    display: flex; flex-wrap: wrap; gap: 4px;
    padding: 4px 12px 8px; font-variant-numeric: tabular-nums;
  }
  .comp-tile {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 2px 6px; min-width: 36px; border-radius: 4px;
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    color: var(--text); line-height: 1.1;
  }
  .comp-tile .comp-lbl { font-size: 9px; color: var(--sub); letter-spacing: 0.3px; }
  .comp-tile .comp-val { font-size: 11px; font-weight: 600; }
  .comp-tile.dim { opacity: 0.35; }
  .comp-tile.spt { border-color: rgba(0,200,150,.45); background: rgba(0,200,150,.08); }
  .comp-tile.spt .comp-lbl { color: var(--teal); }
  .comp-tile.bpa { border-color: rgba(220,170,50,.45); background: rgba(220,170,50,.08); }
  .comp-tile.bpa .comp-lbl { color: #dcaa32; }
  .fill-badge { display:inline-block; font-size:10px; padding:1px 5px; border-radius:3px; margin-left:4px; }
  .fill-held { background:rgba(0,200,150,.15); color:var(--teal); }
  .fill-partial { background:rgba(255,200,50,.15); color:#cca000; }
  .fill-recovered { background:rgba(100,180,255,.15); color:#5ba8e6; }
  .fill-failed { background:rgba(255,80,80,.15); color:#e05050; }

  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }
  .bg-buy   { background: rgba(0,200,150,.18); color: #00c896; border: 1px solid rgba(0,200,150,.35); }
  .bg-sell  { background: rgba(224,85,85,.18);  color: #e05555; border: 1px solid rgba(224,85,85,.35); }
  .bg-wait  { background: rgba(245,200,66,.15); color: #f5c842; border: 1px solid rgba(245,200,66,.30); }
  .bg-fog   { background: rgba(245,166,35,.13); color: #f5a623; border: 1px solid rgba(245,166,35,.28); }
  .bg-avoid { background: rgba(85,85,85,.25);   color: #888;    border: 1px solid rgba(85,85,85,.45); }

  /* Movement */
  .movement { font-size: 11px; color: var(--sub); margin-top: 2px; }
  .up   { color: var(--teal); }
  .down { color: var(--red); }
  .delta-up   { color: var(--teal); font-weight: 600; }
  .delta-down { color: var(--red);  font-weight: 600; }

  /* Expanded chart area */
  .chart-wrap {
    position: relative;
    padding: 10px 12px 12px;
    background: #111;
  }
  .chart-wrap img {
    width: 100%; height: auto;
    display: block; border-radius: 4px;
  }
  .chart-overlay {
    position: absolute;
    top: 16px; right: 18px;
    background: rgba(0,0,0,.55);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
    line-height: 1.6;
    color: rgba(255,255,255,.75);
    pointer-events: none;
  }
  .no-chart {
    padding: 18px 12px;
    font-size: 12px; color: var(--sub);
    text-align: center;
  }
  .warn {
    margin: 0 12px 8px;
    padding: 6px 10px;
    background: rgba(245,166,35,.1);
    border-left: 3px solid var(--orange);
    border-radius: 3px;
    font-size: 12px; color: var(--orange);
  }
  .summary-text {
    padding: 8px 12px 10px;
    font-size: 12px; color: var(--sub); line-height: 1.5;
  }
  footer {
    margin-top: 14px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--sub);
    text-align: center;
  }

  /* Mobile layout: compact 3-row grid (not fully stacked)
     Row 1: rank | ticker (+family tag) | signal badge (right)
     Row 2: $ADR  move×  | ADR× pill (right)
     Row 3: URG bar+val  |  UNC bar+val (right) */
  @media (max-width: 500px) {
    .card { margin-bottom: 6px; }
    .card summary {
      padding: 8px 10px;
      grid-template-columns: auto minmax(0,1fr) auto;
      grid-template-areas:
        "rank  ticker  badge"
        "adr   adr     adrmult"
        "urg   urg     unc";
      row-gap: 6px;
      column-gap: 8px;
      align-items: center;
    }
    .rank { grid-area: rank; font-size: 11px; margin: 0; }
    .ticker-block { grid-area: ticker; min-width: 0; margin: 0; }
    .ticker { font-size: 15px; }
    .day-type { font-size: 10px; }

    .adr-block {
      grid-area: adr; margin: 0; padding: 0;
      display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;
    }
    .adr-val { font-size: 11px; }
    .adr-ratio { font-size: 11px; }

    .adr-mult {
      grid-area: adrmult; padding: 3px 8px; min-width: 0; margin: 0;
      justify-self: end;
    }
    .adr-mult .mult-val { font-size: 12px; }
    .adr-mult .mult-lbl { display: none; }

    /* Break .scores into its 3 children and place each in its own grid cell */
    .scores { display: contents; }
    .scores > :nth-child(1) { grid-area: badge; justify-self: end; }
    .scores > :nth-child(2) { grid-area: urg; }
    .scores > :nth-child(3) { grid-area: unc; justify-self: end; }

    .score-row { gap: 4px; }
    .score-label { width: auto; font-size: 10px; }
    .bar-track { width: 60px; }
    .score-val { width: auto; min-width: 22px; font-size: 11px; }
    .badge { font-size: 10px; padding: 2px 6px; }
  }
</style>
</head>
<body>
"""

_HTML_FOOT = """\
<footer>Auto-refreshes every 5 min &nbsp;·&nbsp; Brooks Price Action Scanner</footer>
<script>
(function(){
  var container = document.getElementById('cards');
  if (!container) return;
  var buttons = document.querySelectorAll('.sort-bar button[data-sort]');
  var originals = Array.prototype.slice.call(container.querySelectorAll('details.card'));

  var KEY_MAP = {
    'rank': 'rank',
    'urgency': 'urgency',
    'uncertainty': 'uncertainty',
    'adr-mult': 'adrMult'
  };
  function sortBy(key){
    var dkey = KEY_MAP[key] || key;
    var cards = Array.prototype.slice.call(container.querySelectorAll('details.card'));
    if (key === 'rank'){
      cards.sort(function(a,b){
        return (parseFloat(a.dataset.rank)||0) - (parseFloat(b.dataset.rank)||0);
      });
    } else {
      cards.sort(function(a,b){
        return (parseFloat(b.dataset[dkey])||0) - (parseFloat(a.dataset[dkey])||0);
      });
    }
    cards.forEach(function(c){ container.appendChild(c); });
  }

  buttons.forEach(function(btn){
    btn.addEventListener('click', function(){
      buttons.forEach(function(b){ b.classList.remove('active'); });
      btn.classList.add('active');
      sortBy(btn.dataset.sort);
    });
  });
})();
</script>
</body>
</html>
"""


def _signal_badge(sig: str) -> str:
    css_cls, label = _SIG_CSS.get(sig, ("bg-avoid", sig))
    return f'<span class="badge {css_cls}">{label}</span>'


def _rank_arrow_html(r: dict) -> str:
    rc = r.get("rank_change")
    if r.get("_first_scan") or rc is None:
        return ""
    if rc > 0:
        return f'<span class="arrow up">▲</span>'
    if rc < 0:
        return f'<span class="arrow down">▼</span>'
    return '<span class="arrow" style="color:#555">—</span>'


def _movement_html(r: dict) -> str:
    if r.get("_first_scan"):
        return '<span class="movement">first scan</span>'
    rc = r.get("rank_change")
    ud = r.get("urgency_delta")
    parts = []
    if rc is None:
        parts.append('<span class="movement up">NEW</span>')
    else:
        pr = r["prev_rank"]
        sign = "+" if rc > 0 else ""
        css = "up" if rc > 0 else ("down" if rc < 0 else "")
        parts.append(f'<span class="movement {css}">was #{pr} ({sign}{rc})</span>')
    if ud is not None:
        sign = "+" if ud >= 0 else ""
        css = "delta-up" if ud >= 0 else "delta-down"
        parts.append(f'<span class="{css}" style="font-size:11px;margin-left:6px">{sign}{ud:.1f} U</span>')
    return " ".join(parts)


def _bar_html(value: float, cls: str) -> str:
    pct = min(100, max(0, value * 10))
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill {cls}" style="width:{pct:.0f}%"></div>'
        f'</div>'
    )


def _adr_mult_tier(mult: float) -> str:
    """Return CSS tier class for an ADR multiple (matches chart header thresholds)."""
    if mult <= 0:
        return "tier-cold"
    if mult < 1.0:
        return "tier-cold"
    if mult < 1.5:
        return "tier-warm"
    if mult < 2.0:
        return "tier-hot"
    return "tier-extreme"


def _build_component_strip(details: dict) -> str:
    """Mini per-component subscore strip for the card.

    Order: spike · gap · pull · FT · MA · vol · tail · SPT (new)
    Each tile shows label + rounded value. Dimmed when value == 0.
    SPT tile gets a teal accent so it reads as the newest signal.
    """
    if not details:
        return ""
    _TILE_TIPS = {
        "spike": "Spike Quality (0–4): Strength of opening move — strong body ratio, closes near extreme, small tails",
        "gap":   "Gap Integrity (-2–+2): Did the gap hold? +2=intact, +1=partial fill, 0=filled but recovered, -2=filled and failed",
        "pull":  "Pullback Quality (-1–+2): Depth and character of first pullback — shallow + tight = bullish",
        "FT":    "Follow Through (-1.5–+2): Do bars after the spike continue in direction? Consecutive trend bars = strong",
        "MA":    "MA Separation (0–1): Price distance from 20-EMA — wider spread = stronger trend",
        "vol":   "Volume Confirmation (0–1): Volume expanding on trend bars, contracting on pullback bars",
        "tail":  "Tail Quality (-0.5–+1): Wicks on the right side (rejection tails) — tails favoring direction = bullish",
        "SPT":   "Small Pullback Trend (0–3): Calm drifting trend with tiny pullbacks — the strongest continuation pattern",
        "BPA":   "BPA Alignment (-1–+2): Brooks pattern confirmation — H2/L2=+2, H1/L1=+1.5, opposing=-1. Overlays the signal, not the urgency score",
    }
    tiles = [
        ("spike", details.get("spike_quality", 0.0), ""),
        ("gap",   details.get("gap_integrity", 0.0), ""),
        ("pull",  details.get("pullback_quality", 0.0), ""),
        ("FT",    details.get("follow_through", 0.0), ""),
        ("MA",    details.get("ma_separation", 0.0), ""),
        ("vol",   details.get("volume_conf", 0.0), ""),
        ("tail",  details.get("tail_quality", 0.0), ""),
        ("SPT",   details.get("small_pullback_trend", 0.0), "spt"),
        ("BPA",   details.get("bpa_alignment", 0.0), "bpa"),
    ]
    parts = ['<div class="comp-strip">']
    for label, val, extra_cls in tiles:
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 0.0
        dim = " dim" if abs(v) < 0.005 else ""
        cls = f"comp-tile {extra_cls}{dim}".strip()
        sign = "" if v >= 0 else "−"
        av = abs(v)
        val_str = f"{sign}{av:.1f}" if av < 10 else f"{sign}{av:.0f}"
        tip = _TILE_TIPS.get(label, label)
        parts.append(
            f'<span class="{cls}" title="{tip}">'
            f'<span class="comp-lbl">{label}</span>'
            f'<span class="comp-val">{val_str}</span>'
            f'</span>'
        )
    parts.append('</div>')
    return "".join(parts)


def _build_card_html(r: dict, chart_b64: str | None) -> str:
    ticker   = r.get("ticker", "?")
    sig      = r.get("signal", "PASS")
    urg      = r.get("urgency", 0.0)
    unc      = r.get("uncertainty", 0.0)
    day_type = r.get("day_type", "")
    summary  = r.get("summary", "")
    warn     = r.get("day_type_warning", "")
    pc       = r.get("_prior_close", 0.0)
    rank     = r.get("rank", 0)

    # ADR / move_ratio
    adr_val   = r.get("daily_atr", 0.0) or 0.0
    move_rat  = r.get("move_ratio", 0.0) or 0.0
    adr_mult  = r.get("adr_multiple", 0.0) or 0.0
    adr_str   = f"${adr_val:.2f} ADR" if adr_val > 0 else ""
    ratio_str = f"{move_rat:.1f}× move" if adr_val > 0 else ""

    # Rank block
    rank_html = (
        f'<div class="rank">'
        f'{_rank_arrow_html(r)}'
        f'<span style="font-size:13px;font-weight:600;color:{"#e0e0e0"}">{rank}</span>'
        f'</div>'
    )

    # Ticker + day type (+ family-dedup badge when this row represents a family)
    fam      = r.get("family")
    fam_sibs = r.get("family_siblings") or []
    fam_n    = len(fam_sibs)
    if fam and fam_n:
        sib_tip   = ", ".join(fam_sibs)
        fam_badge = (
            f'<span class="fam-badge" '
            f'title="represents {fam} family ({fam_n + 1} tickers): {ticker}, {sib_tip}" '
            f'style="display:inline-block;margin-left:6px;padding:1px 6px;'
            f'font-size:10px;font-weight:600;letter-spacing:0.3px;'
            f'background:#2b3b52;color:#8ab4f8;border:1px solid #3a5272;'
            f'border-radius:10px;vertical-align:middle;">'
            f'{fam}&nbsp;+{fam_n}'
            f'</span>'
        )
    else:
        fam_badge = ""

    # Cycle-phase badge (Layer 1 classifier) — shown inline with day_type
    details_r = r.get("details") or {}
    cp = details_r.get("cycle_phase") or {}
    cp_label_map = {
        "bull_spike": "↑ SPIKE",
        "bear_spike": "↓ SPIKE",
        "bull_channel": "↑ channel",
        "bear_channel": "↓ channel",
        "trading_range": "↔ range",
    }
    cp_top = cp.get("top") if isinstance(cp, dict) else None
    cp_conf = cp.get("confidence", 0.0) if isinstance(cp, dict) else 0.0
    if cp_top and cp_conf >= 0.30:
        cp_html = (
            f'<span class="cycle-badge" '
            f'title="Cycle phase (Layer 1) — top-1 with confidence. Higher is cleaner.">'
            f'{cp_label_map.get(cp_top, cp_top)} {cp_conf:.2f}'
            f'</span>'
        )
    else:
        cp_html = ""

    # Gap fill-status badge
    gfs = details_r.get("gap_fill_status", "")
    _fill_cls = {"held": "fill-held", "partial_fill": "fill-partial",
                 "filled_recovered": "fill-recovered", "filled_failed": "fill-failed"}
    _fill_label = {"held": "gap held", "partial_fill": "partial fill",
                   "filled_recovered": "fill recovered", "filled_failed": "fill failed"}
    if gfs and gfs != "held":
        fill_badge = f'<span class="fill-badge {_fill_cls.get(gfs, "")}">{_fill_label.get(gfs, gfs)}</span>'
    else:
        fill_badge = ""

    ticker_html = (
        f'<div class="ticker-block">'
        f'<div class="ticker">{ticker}{fam_badge}</div>'
        f'<div class="day-type">{day_type} {cp_html} {fill_badge}</div>'
        f'</div>'
    )

    # ADR block (fills the 1fr spacer column)
    adr_html = (
        f'<div class="adr-block">'
        f'<div class="adr-val">{adr_str}</div>'
        f'<div class="adr-ratio">{ratio_str}</div>'
        f'</div>'
    )

    # ADR-multiple tile (Minervini-style expansion gauge)
    tier = _adr_mult_tier(adr_mult)
    if adr_mult > 0:
        mult_tile = (
            f'<div class="adr-mult {tier}">'
            f'<div class="mult-val">{adr_mult:.2f}\u00d7</div>'
            f'<div class="mult-lbl">ADR</div>'
            f'</div>'
        )
    else:
        mult_tile = (
            f'<div class="adr-mult tier-cold">'
            f'<div class="mult-val" style="font-size:12px">—</div>'
            f'<div class="mult-lbl">ADR</div>'
            f'</div>'
        )

    # Scores + badge
    scores_html = (
        f'<div class="scores">'
        f'<div>'
        f'{_signal_badge(sig)}'
        f'</div>'
        f'<div class="score-row">'
        f'<span class="score-label">URG</span>'
        f'{_bar_html(urg, "bar-urg")}'
        f'<span class="score-val" style="color:#00c896">{urg:.1f}</span>'
        f'</div>'
        f'<div class="score-row">'
        f'<span class="score-label">UNC</span>'
        f'{_bar_html(unc, "bar-unc")}'
        f'<span class="score-val" style="color:#e05555">{unc:.1f}</span>'
        f'</div>'
        f'</div>'
    )

    # Movement
    movement_html = f'<div style="padding:0 12px 8px;display:flex;gap:8px;align-items:center">{_movement_html(r)}</div>'

    # Per-component subscore strip (spike · gap · pull · FT · MA · vol · tail · SPT)
    comp_strip_html = _build_component_strip(r.get("details") or {})

    # Chart area
    if chart_b64:
        overlay = f'URG {urg:.1f} / UNC {unc:.1f}'
        if adr_mult > 0:
            overlay += f' · {adr_mult:.2f}× ADR'
        elif adr_val > 0:
            overlay += f' · {ratio_str}'
        chart_html = (
            f'<div class="chart-wrap">'
            f'<img src="data:image/png;base64,{chart_b64}" alt="{ticker} 5-min chart">'
            f'<div class="chart-overlay">{overlay}</div>'
            f'</div>'
        )
    else:
        chart_html = '<div class="no-chart">Chart not available</div>'

    # Warning
    warn_html = ""
    if warn and warn.lower() not in ("", "none"):
        warn_html = f'<div class="warn">⚠ {warn}</div>'

    # Summary
    summary_html = f'<div class="summary-text">{summary}</div>' if summary else ""

    # Sortable data attrs — consumed by the dashboard sort toolbar JS
    data_attrs = (
        f'data-urgency="{urg:.3f}" '
        f'data-uncertainty="{unc:.3f}" '
        f'data-adr-mult="{adr_mult:.3f}" '
        f'data-rank="{rank}" '
        f'data-ticker="{ticker}"'
    )

    return (
        f'<details class="card" {data_attrs}>'
        f'<summary>'
        f'{rank_html}'
        f'{ticker_html}'
        f'{adr_html}'
        f'{mult_tile}'
        f'{scores_html}'
        f'</summary>'
        f'{movement_html}'
        f'{comp_strip_html}'
        f'{warn_html}'
        f'{summary_html}'
        f'{chart_html}'
        f'</details>\n'
    )


# ── aiedge.trade Integration ──────────────────────────────────────────────────

AIEDGE_SCAN_URL = "https://www.aiedge.trade/api/scan"  # apex redirects 307 → www and urllib drops POST body on redirect

def _serialize_bars(df_5m: pd.DataFrame, last_n: int = 80) -> list[dict]:
    """5-min OHLCV DataFrame → compact bars list for ScanResult.chart.bars."""
    if df_5m is None or len(df_5m) == 0:
        return []
    df = df_5m.copy()
    # chart_renderer expects a "datetime" column, but some callers pass an index.
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    df = df.sort_index().tail(last_n)
    bars: list[dict] = []
    for ts, row in df.iterrows():
        try:
            bar = {
                "t": int(pd.Timestamp(ts).timestamp()),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        vol = row.get("volume") if hasattr(row, "get") else None
        if vol is not None:
            try:
                bar["v"] = float(vol)
            except (TypeError, ValueError):
                pass
        bars.append(bar)
    return bars


def _serialize_key_levels(
    prior_close: float | None,
    levels: dict | None,
) -> dict | None:
    """Map internal levels dict → KeyLevels shape the site expects, or None."""
    out: dict[str, float] = {}
    if prior_close and prior_close > 0:
        out["priorClose"] = float(prior_close)
    if levels:
        def _maybe(src_key: str, dst_key: str) -> None:
            v = levels.get(src_key)
            if v is not None:
                try:
                    out[dst_key] = float(v)
                except (TypeError, ValueError):
                    pass
        _maybe("pdh", "priorDayHigh")
        _maybe("pdl", "priorDayLow")
        _maybe("onh", "overnightHigh")
        _maybe("onl", "overnightLow")
        _maybe("pmh", "premarketHigh")
        _maybe("pml", "premarketLow")
    return out or None


def _serialize_scan_payload(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    df5m_map: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Convert internal result dicts to the ScanPayload JSON format for aiedge.trade."""
    time_str = now_et.strftime("%I:%M %p").lstrip("0") + " ET"
    date_str = now_et.strftime("%Y-%m-%d")
    next_str = _next_scan_time_str(now_et, interval_min)

    def _adr_tier(mult: float) -> str:
        if mult >= 2.0: return "extreme"
        if mult >= 1.5: return "hot"
        if mult >= 1.0: return "warm"
        return "cold"

    def _map_fill_status(gfs: str) -> str | None:
        m = {"held": "held", "partial_fill": "partial", "filled_recovered": "recovered", "filled_failed": "failed"}
        return m.get(gfs)

    def _map_signal(sig: str) -> str:
        sig_up = sig.upper()
        if "BUY" in sig_up: return "BUY"
        if "SELL" in sig_up: return "SELL"
        if sig_up == "WAIT": return "WAIT"
        if sig_up == "FOG": return "FOG"
        if sig_up in ("AVOID", "PASS"): return "AVOID"
        return "AVOID"

    serialized = []
    for r in results:
        details = r.get("details") or {}
        cp = details.get("cycle_phase") or {}
        cp_top = cp.get("top") if isinstance(cp, dict) else None
        cp_conf = cp.get("confidence", 0.0) if isinstance(cp, dict) else 0.0
        cp_label_map = {
            "bull_spike": "↑ SPIKE",
            "bear_spike": "↓ SPIKE",
            "bull_channel": "↑ channel",
            "bear_channel": "↓ channel",
            "trading_range": "↔ range",
        }
        cycle_phase = None
        if cp_top and cp_conf >= 0.30:
            cycle_phase = f"{cp_label_map.get(cp_top, cp_top)} {cp_conf:.2f}"

        gfs = details.get("gap_fill_status", "")
        fill_status = _map_fill_status(gfs) if gfs and gfs != "held" else None

        adr_mult = r.get("adr_multiple", 0.0) or 0.0

        ticker = r.get("ticker", "?")
        chart_obj: dict | None = None
        if df5m_map is not None:
            df5 = df5m_map.get(ticker)
            bars = _serialize_bars(df5) if df5 is not None else []
            if bars:
                chart_obj = {"bars": bars, "timeframe": "5min"}
                kl = _serialize_key_levels(
                    r.get("_prior_close"),
                    intraday_levels.get(ticker),
                )
                if kl:
                    chart_obj["keyLevels"] = kl

        entry = {
            "ticker": ticker,
            "rank": r.get("rank", 0),
            "urgency": round(r.get("urgency", 0.0), 1),
            "uncertainty": round(r.get("uncertainty", 0.0), 1),
            "signal": _map_signal(r.get("signal", "PASS")),
            "dayType": (r.get("day_type", "") or "").replace(" ", "_"),
            "adr": round(r.get("daily_atr", 0.0) or 0.0, 2),
            "adrRatio": round(r.get("move_ratio", 0.0) or 0.0, 1),
            "adrMult": round(adr_mult, 2),
            "adrTier": _adr_tier(adr_mult),
            "movement": _fmt_movement(r),
            "components": {
                "spike": round(float(details.get("spike_quality", 0.0)), 1),
                "gap": round(float(details.get("gap_integrity", 0.0)), 1),
                "pull": round(float(details.get("pullback_quality", 0.0)), 1),
                "ft": round(float(details.get("follow_through", 0.0)), 1),
                "ma": round(float(details.get("ma_separation", 0.0)), 1),
                "vol": round(float(details.get("volume_conf", 0.0)), 1),
                "tail": round(float(details.get("tail_quality", 0.0)), 1),
                "spt": round(float(details.get("small_pullback_trend", 0.0)), 1),
                "bpa": round(float(details.get("bpa_alignment", 0.0)), 1),
            },
            "summary": r.get("summary", ""),
        }
        if cycle_phase:
            entry["cyclePhase"] = cycle_phase
        if fill_status:
            entry["fillStatus"] = fill_status
        if r.get("day_type_warning"):
            entry["warning"] = r["day_type_warning"]
        if chart_obj:
            entry["chart"] = chart_obj

        serialized.append(entry)

    return {
        "timestamp": time_str,
        "date": date_str,
        "symbolsScanned": total_symbols,
        "passedFilters": passed,
        "scanTime": f"{elapsed:.2f}s",
        "nextScan": next_str,
        "results": serialized,
    }


def _post_to_aiedge(
    results: list[dict],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
    df5m_map: dict[str, pd.DataFrame] | None = None,
) -> None:
    """POST scan results to aiedge.trade/api/scan. Fire-and-forget in bg thread."""
    try:
        import os
        import urllib.request
        sync_secret = os.environ.get("SYNC_SECRET")
        if not sync_secret:
            logger.warning(
                "aiedge.trade POST skipped: SYNC_SECRET env var not set. "
                "Export it so the scanner can authenticate with /api/scan."
            )
            return
        payload = _serialize_scan_payload(
            results, now_et, total_symbols, passed, elapsed, interval_min, df5m_map
        )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            AIEDGE_SCAN_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {sync_secret}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info(f"aiedge.trade POST → {resp.status} ({len(data)//1024}KB)")
    except Exception as e:
        logger.warning(f"aiedge.trade POST failed: {e}")


def _generate_dashboard(
    results: list[dict],
    df5m_map: dict[str, pd.DataFrame],
    now_et: datetime,
    total_symbols: int,
    passed: int,
    elapsed: float,
    interval_min: int,
) -> None:
    """Render charts + build HTML dashboard. Runs in background thread."""
    t0 = time.monotonic()
    logger.info(f"Dashboard: rendering charts for {len(results)} stocks…")

    # Render charts
    chart_b64_map: dict[str, str | None] = {}
    for r in results:
        ticker = r["ticker"]
        df5 = df5m_map.get(ticker)
        if df5 is not None and len(df5) >= 2:
            chart_b64_map[ticker] = render_chart_base64(
                df5, ticker, r.get("_prior_close", 0),
                adr_multiple=r.get("adr_multiple"),
                levels=intraday_levels.get(ticker),
            )
        else:
            chart_b64_map[ticker] = None

    charts_ok = sum(1 for v in chart_b64_map.values() if v is not None)
    logger.info(f"Dashboard: {charts_ok}/{len(results)} charts rendered in {time.monotonic()-t0:.1f}s")

    # Build HTML
    time_str = now_et.strftime("%I:%M %p").lstrip("0")
    next_str = _next_scan_time_str(now_et, interval_min)

    body_parts = [_HTML_HEAD]
    body_parts.append(
        f'<header>'
        f'<h1>Live Scanner</h1>'
        f'<div class="meta">{time_str} ET &nbsp;·&nbsp; {now_et.strftime("%Y-%m-%d")}</div>'
        f'</header>\n'
    )
    body_parts.append(
        f'<div class="stats">'
        f'<span>📡 {total_symbols:,} symbols</span>'
        f'<span>✅ {passed} passed filters</span>'
        f'<span>⏱ {elapsed:.2f}s</span>'
        f'<span>Next: {next_str}</span>'
        f'</div>\n'
    )

    # Sort toolbar — JS in footer wires the clicks to reorder cards in place.
    body_parts.append(
        '<div class="sort-bar">'
        '<span>Sort:</span>'
        '<button data-sort="rank" class="active">Rank</button>'
        '<button data-sort="urgency">Urgency</button>'
        '<button data-sort="adr-mult">ADR ×</button>'
        '<button data-sort="uncertainty">Uncertainty</button>'
        '</div>\n'
    )

    # Scoring legend — collapsible
    body_parts.append(
        '<details class="legend">'
        '<summary>How Scoring Works</summary>'
        '<div class="legend-body">'

        '<div class="legend-section">'
        '<h3>Main Scores</h3>'
        '<dl>'
        '<dt>URG (Urgency 0–10)</dt>'
        '<dd>How strongly the chart is pulling in one direction. Weighted sum of 16 components below, normalized by day type. Higher = clearer trend.</dd>'
        '<dt>UNC (Uncertainty 0–10)</dt>'
        '<dd>How confused or two-sided the chart is. Overlapping bars, dojis, alternating colors. Higher = harder to read.</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Signals</h3>'
        '<dl>'
        '<dt>BUY PULLBACK / SELL PULLBACK</dt>'
        '<dd>High urgency + low uncertainty + good R:R. The highest-confidence entry — trend is strong, pullback gives you a stop.</dd>'
        '<dt>BUY SPIKE / SELL SPIKE</dt>'
        '<dd>Strong directional move with no pullback yet. Market is leaving — consider market order with wide stop.</dd>'
        '<dt>SELL FLIP / BUY FLIP</dt>'
        '<dd>Intraday direction reversal. Always-in direction flipped against the gap, confirmed by BPA pattern (L2/H2).</dd>'
        '<dt>WAIT</dt>'
        '<dd>Promising direction but needs more bars. Urgency decent but not actionable yet.</dd>'
        '<dt>FOG</dt>'
        '<dd>Can\'t read the chart. High uncertainty regardless of urgency. Sit on hands.</dd>'
        '<dt>AVOID</dt>'
        '<dd>Gap failed (price through prior close) or trap state (high urgency + high uncertainty — both sides showing strength).</dd>'
        '<dt>PASS</dt>'
        '<dd>Readable but weak. No urgency, no edge.</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Urgency Components (hover tiles for details)</h3>'
        '<dl>'
        '<dt>spike (0–4)</dt><dd>Opening spike strength — body ratio, close location, tail size on first 1–4 trend bars</dd>'
        '<dt>gap (-2–+2)</dt><dd>Gap integrity — did price hold the gap? +2 intact, -2 filled. Post-fill recovery analysis: filled_recovered avoids the -2 penalty</dd>'
        '<dt>pull (-1–+2)</dt><dd>First pullback quality — shallow (good) vs deep (bad), tight vs loose</dd>'
        '<dt>FT (-1.5–+2)</dt><dd>Follow through — are bars after the spike continuing in direction?</dd>'
        '<dt>MA (0–1)</dt><dd>Moving average separation — price distance from 20-EMA</dd>'
        '<dt>vol (0–1)</dt><dd>Volume confirmation — volume expanding on trend bars, contracting on pullbacks</dd>'
        '<dt>tail (-0.5–+1)</dt><dd>Tail quality — wicks rejecting the wrong direction</dd>'
        '<dt>SPT (0–3)</dt><dd>Small pullback trend — calm drifting trend with tiny pullbacks, strongest continuation</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>BPA Overlay (not part of urgency sum)</h3>'
        '<dl>'
        '<dt>BPA (-1–+2)</dt><dd>Brooks pattern alignment — runs 8 pattern detectors (H1, H2, L1, L2, FL1, FL2, spike&amp;channel, failed breakout). '
        'Modifies the signal after urgency/uncertainty are computed: strong in-direction pattern can upgrade WAIT→BUY, opposing pattern can downgrade BUY→WAIT</dd>'
        '</dl>'
        '</div>'

        '<div class="legend-section">'
        '<h3>Other Indicators</h3>'
        '<dl>'
        '<dt>ADR ×</dt><dd>Today\'s range as a multiple of the 20-day average daily range. &gt;1.0 = expanding beyond normal, &lt;0.5 = quiet day</dd>'
        '<dt>Fill Status</dt><dd>Gap fill analysis: held (gap intact), partial fill (&lt;50% filled), fill recovered (filled but price recovered), fill failed (filled, no recovery)</dd>'
        '</dl>'
        '</div>'

        '</div>'
        '</details>\n'
    )

    body_parts.append('<div id="cards">\n')
    if not results:
        body_parts.append('<p style="color:var(--sub);padding:20px 0">No liquid stocks passed filters this cycle.</p>\n')
    else:
        for r in results:
            ticker = r["ticker"]
            body_parts.append(_build_card_html(r, chart_b64_map.get(ticker)))
    body_parts.append('</div>\n')

    body_parts.append(_HTML_FOOT)
    html = "".join(body_parts)

    try:
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Dashboard saved → {DASHBOARD_PATH} ({len(html)//1024}KB)")
    except Exception as e:
        logger.warning(f"Dashboard write failed: {e}")

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
        backfill_intraday_bars(api_key)
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
