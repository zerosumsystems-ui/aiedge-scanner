"""
Chart generation stage: renders charts for each script segment.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

from shared.chart_renderer import render_chart
from shared.databento_client import DatabentClient

logger = logging.getLogger(__name__)


def run(config: dict, script: dict, run_id: str, run_dir: str) -> list[str]:
    """
    Generate charts for each segment in the script.

    Returns list of chart PNG paths.
    """
    pipeline_name = config["pipeline_name"]
    charts_dir = Path(run_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    branding = config.get("assembly", {}).get("branding", {})

    client = DatabentClient()
    chart_paths = []

    # Load screener output to get key_levels per ticker
    screener_key_levels = {}
    screener_output_path = Path(run_dir) / "screener_output.json"
    if screener_output_path.exists():
        try:
            with open(screener_output_path) as f:
                screener_out = json.load(f)
            for sym, sym_data in screener_out.get("data", {}).get("futures", {}).items():
                if "key_levels" in sym_data:
                    screener_key_levels[sym.upper()] = sym_data["key_levels"]
            if screener_key_levels:
                logger.info(f"Loaded key_levels for: {list(screener_key_levels.keys())}")
        except Exception as e:
            logger.warning(f"Could not load key_levels from screener_output.json: {e}")

    for i, segment in enumerate(script["segments"]):
        chart_spec = segment.get("chart_spec")
        if not chart_spec:
            logger.warning(f"Segment {i} has no chart_spec, generating placeholder")
            chart_paths.append("")
            continue

        ticker = chart_spec.get("ticker", "UNKNOWN")
        timeframe = chart_spec.get("timeframe", "daily")
        lookback = chart_spec.get("lookback", 60)
        annotations = chart_spec.get("annotations", {})
        overlays = chart_spec.get("overlays", [])

        output_path = str(charts_dir / f"chart_{i}.png")

        try:
            # Fetch data for the chart
            df = _fetch_chart_data(client, ticker, timeframe, lookback)

            if df.empty:
                logger.error(f"No data for {ticker} ({timeframe})")
                chart_paths.append("")
                continue

            # Resolve display name for futures
            company = _CME_COMPANY_NAMES.get(ticker.split(".")[0].upper())

            # Key levels from screener (futures only)
            base_ticker = ticker.split(".")[0].upper()
            key_levels = screener_key_levels.get(base_ticker)

            render_chart(
                ticker,
                timeframe,
                lookback,
                annotations,
                overlays,
                segment.get("topic", ""),
                output_path,
                df=df,
                company=company,
                key_levels=key_levels,
            )

            chart_paths.append(output_path)
            logger.info(f"Chart {i} rendered: {ticker} ({timeframe})")

        except Exception as e:
            logger.error(f"Chart {i} generation failed for {ticker}: {e}")
            chart_paths.append("")

    logger.info(f"Chart generation complete: {sum(1 for p in chart_paths if p)}/{len(chart_paths)} charts")
    return chart_paths


import time as _time

_CME_TICKERS = frozenset({
    "ES", "NQ", "YM", "RTY", "GC", "CL", "ZN", "ZB", "ZF", "ZT",
    "MES", "MNQ", "MYM", "M2K", "MGC",
})

_CME_COMPANY_NAMES = {
    "ES": "E-Mini S&P 500",
    "NQ": "E-Mini Nasdaq-100",
    "YM": "E-Mini Dow Jones",
    "RTY": "E-Mini Russell 2000",
    "GC": "Gold Futures",
    "CL": "Crude Oil WTI",
    "ZN": "10-Yr T-Note",
    "ZB": "30-Yr T-Bond",
    "MES": "Micro E-Mini S&P 500",
    "MNQ": "Micro E-Mini Nasdaq-100",
}


def _is_cme(ticker: str) -> bool:
    base = ticker.split(".")[0].upper()
    return base in _CME_TICKERS or ".c." in ticker or ".n." in ticker


def _safe_cme_end() -> datetime:
    """
    Return a safe end datetime for GLBX.MDP3 queries.

    GLBX.MDP3 ohlcv data lags real-time by ~1-2 hours beyond the licensed window.
    Using now-2h or the prior market open (whichever is earlier) ensures the start
    of our query window is well within the indexed range. The dataset_unavailable_range
    self-heal in query_ohlcv will cap the end further if needed.
    """
    now_utc = datetime.now(timezone.utc)
    is_dst = bool(_time.localtime().tm_isdst)
    et_offset_hours = 4 if is_dst else 5
    # Prior NY market open: today's 9:30 AM ET in UTC
    market_open_utc = now_utc.replace(
        hour=13 + (0 if et_offset_hours == 4 else 1),
        minute=30, second=0, microsecond=0,
    )
    return min(now_utc - timedelta(hours=2), market_open_utc)


def _fetch_chart_data(
    client: DatabentClient, ticker: str, timeframe: str, lookback: int
) -> pd.DataFrame:
    """Fetch OHLCV data appropriate for the chart timeframe, with 5-min resampling."""
    base = ticker.split(".")[0].upper()
    is_cme = _is_cme(ticker)

    if is_cme:
        dataset = "GLBX.MDP3"
        symbol = f"{base}.c.0"
    else:
        dataset = "EQUS.MINI"
        symbol = ticker

    # Databento valid bar schemas: ohlcv-1m, ohlcv-1h, ohlcv-1d
    # No ohlcv-5m or ohlcv-15m — fetch ohlcv-1m and resample.
    resample_to_5min  = timeframe in ("5min",)
    resample_to_15min = timeframe in ("15min", "15m")

    # For CME symbols anchor backwards from a safe historical end so that the
    # start itself is never past the available data window.
    # For equities, anchor from now (equity data lags only ~30 min).
    if is_cme:
        anchor = _safe_cme_end()  # <= prior market open or now-2h
    else:
        anchor = datetime.now(timezone.utc)

    if timeframe == "daily":
        schema = "ohlcv-1d"
        start = anchor - timedelta(days=max(lookback + 10, 30))
        end   = anchor if not is_cme else None  # None → defaults to _safe_end in client
    elif timeframe in ("60min", "1h"):
        schema = "ohlcv-1h"
        start = anchor - timedelta(hours=max(lookback + 5, 24))
        end   = anchor if is_cme else None
    elif resample_to_15min:
        schema = "ohlcv-1m"
        if is_cme:
            # Globex session: 18:00 ET prior calendar day → anchor
            is_dst = bool(_time.localtime().tm_isdst)
            et_offset_hours = 4 if is_dst else 5
            prior_cal = anchor.date() - timedelta(days=1)
            start = datetime(
                prior_cal.year, prior_cal.month, prior_cal.day,
                18 + et_offset_hours, 0, 0, tzinfo=timezone.utc,
            )
        else:
            start = anchor - timedelta(hours=max(lookback * 15 // 60 + 2, 8))
        end = anchor if is_cme else None
    elif resample_to_5min:
        schema = "ohlcv-1m"
        # lookback 5-min bars → lookback*5 minutes of 1-min data, plus buffer
        start = anchor - timedelta(hours=max(lookback * 5 // 60 + 4, 8))
        end   = anchor if is_cme else None
    elif timeframe == "1min":
        schema = "ohlcv-1m"
        start = anchor - timedelta(hours=max(lookback // 60 + 2, 4))
        end   = anchor if is_cme else None
    else:
        schema = "ohlcv-1d"
        start = anchor - timedelta(days=max(lookback + 10, 30))
        end   = None

    query_kwargs = dict(dataset=dataset, symbols=[symbol], schema=schema, start=start)
    if end is not None:
        query_kwargs["end"] = end

    df = client.query_ohlcv(**query_kwargs)

    if df.empty:
        return df

    # Normalise columns and index
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    # Drop non-OHLCV columns before resampling
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[ohlcv_cols]

    # Resample 1-min → 5-min if needed
    if resample_to_5min:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in ohlcv_cols:
            agg["volume"] = "sum"
        df = df.resample("5min").agg(agg).dropna(subset=["open", "close"])

    # Resample 1-min → 15-min if needed
    if resample_to_15min:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in ohlcv_cols:
            agg["volume"] = "sum"
        df = df.resample("15min").agg(agg).dropna(subset=["open", "close"])

    return df
