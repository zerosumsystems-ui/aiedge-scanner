"""Intraday reference levels (PDH/PDL/ONH/ONL/PMH/PML).

Runs once at scanner startup. Pulls yesterday's regular-session bars +
today's pre-market bars from Databento Historical, then reduces each
per-instrument window to high/low extremes.

Extracted from live_scanner.py (Phase 4b). PDO (prior-day open) is
not computed here — it comes from `fetch_prior_closes` in
`aiedge.data.databento`, which treats it as the last daily close.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as _dtime, timedelta

import databento as db

from aiedge.data.databento import (
    DATASET,
    ET,
    _fetch_ohlcv1m_range,
    _prev_trading_days,
)

logger = logging.getLogger(__name__)


def fetch_intraday_key_levels(api_key: str) -> dict[str, dict[str, float]]:
    """Fetch per-symbol intraday reference levels from Databento Historical.

    Returns {symbol: {"pdh":.., "pdl":.., "onh":.., "onl":.., "pmh":.., "pml":..}}
    — any missing level is simply absent from its dict, so callers must
    use dict.get(...).

    Windows (all ET):
      PDH / PDL — prior trading day regular session (09:30-16:00) high/low
      ONH / ONL — today extended pre-market (04:00-09:30) high/low
      PMH / PML — today narrow pre-market (08:00-09:30) high/low
    """
    logger.info("Fetching intraday key levels (PDH/PDL/ONH/ONL/PMH/PML)…")
    t0 = time.monotonic()
    hist = db.Historical(key=api_key)

    yesterday = _prev_trading_days(1)
    today = date.today()

    # Prior-day regular session window
    pd_start = ET.localize(datetime.combine(yesterday, datetime.min.time())
                           ).replace(hour=9, minute=30)
    pd_end = pd_start.replace(hour=16, minute=0)

    # Today's extended pre-market window (ON = 04:00-09:30, PM = 08:00-09:30)
    on_start = ET.localize(datetime.combine(today, datetime.min.time())
                           ).replace(hour=4, minute=0)
    on_end = on_start.replace(hour=9, minute=30)

    df_pd = _fetch_ohlcv1m_range(api_key, pd_start, pd_end, "PriorDay")

    # Pre-market bars — cap end at now if we're mid-session-start
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
        # ON window = full 04:00-09:30
        g_on = df_pm.groupby("instrument_id").agg(onh=("high", "max"), onl=("low", "min"))
        for iid, row in g_on.iterrows():
            sym = id_to_sym.get(iid)
            if sym:
                d = levels.setdefault(sym, {})
                d["onh"] = float(row["onh"])
                d["onl"] = float(row["onl"])

        # PM narrow window = 08:00-09:30 subset
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
