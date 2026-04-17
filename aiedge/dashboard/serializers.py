"""JSON serializers for the aiedge.trade /api/scan payload.

Turns the live scanner's internal result dicts into the `ScanPayload`
shape that the site expects. Also exposes `_serialize_bars` and
`_serialize_key_levels` — the per-ticker sub-serializers — in case
other consumers want to emit the same compact chart format.

Extracted from live_scanner.py (Phase 4h). `_serialize_scan_payload`
takes `intraday_levels`, `first_scan_hour`, `first_scan_min` as
explicit parameters instead of reading live_scanner globals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from aiedge.dashboard.console import _next_scan_time_str
from aiedge.signals.postprocess import _fmt_movement


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
    prior_close: Optional[float],
    levels: Optional[dict],
) -> Optional[dict]:
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
    intraday_levels: dict,
    first_scan_hour: int,
    first_scan_min: int,
    df5m_map: Optional[dict[str, pd.DataFrame]] = None,
) -> dict:
    """Convert internal result dicts to the ScanPayload JSON format for aiedge.trade."""
    time_str = now_et.strftime("%I:%M %p").lstrip("0") + " ET"
    date_str = now_et.strftime("%Y-%m-%d")
    next_str = _next_scan_time_str(now_et, interval_min, first_scan_hour, first_scan_min)

    def _adr_tier(mult: float) -> str:
        if mult >= 2.0:
            return "extreme"
        if mult >= 1.5:
            return "hot"
        if mult >= 1.0:
            return "warm"
        return "cold"

    def _map_fill_status(gfs: str) -> Optional[str]:
        m = {"held": "held", "partial_fill": "partial",
             "filled_recovered": "recovered", "filled_failed": "failed"}
        return m.get(gfs)

    def _map_signal(sig: str) -> str:
        sig_up = sig.upper()
        if "BUY" in sig_up:
            return "BUY"
        if "SELL" in sig_up:
            return "SELL"
        if sig_up == "WAIT":
            return "WAIT"
        if sig_up == "FOG":
            return "FOG"
        if sig_up in ("AVOID", "PASS"):
            return "AVOID"
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
        chart_obj: Optional[dict] = None
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
        htf = r.get("htf_alignment")
        if htf and htf != "no_data":
            entry["htfAlignment"] = htf
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
