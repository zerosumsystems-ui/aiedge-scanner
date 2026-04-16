"""Render two SPT-validation example charts:
  - SPY 2026-02-12 (SPT day, peak SPT 3.00, urg 3.9 — clean drift with shallow pullbacks)
  - SPY 2026-03-16 (chop day, peak SPT 0.00, urg 0.9 — no SPT)

Saves to logs/live_scanner/screenshots/SPT_*.png  (dark_color theme).
"""
from __future__ import annotations
import os, sys, datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import databento as db

from shared.brooks_score import _score_small_pullback_trend, score_gap
from shared.chart_renderer import render_chart

KEY_ENV_FILE = Path.home() / "keys" / "databento.env"
DB_KEY = ""
for line in KEY_ENV_FILE.read_text().splitlines():
    if line.startswith("DATABENTO_API_KEY="):
        DB_KEY = line.split("=", 1)[1].strip()
        break

OUT_DIR = Path.home() / "video-pipeline" / "logs" / "live_scanner" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    ("SPT_FIRE",  "2026-02-12", "SPY"),   # peak SPT 3.00, urg 3.9 — clean small-pullback trend
    ("SPT_CHOP",  "2026-01-15", "SPY"),   # peak SPT 0.62, urg 0.7 — chop, no SPT signal
]


def fetch_day(sym: str, day: str) -> pd.DataFrame:
    hist = db.Historical(key=DB_KEY)
    start = dt.datetime.fromisoformat(day + "T14:30:00+00:00")
    end   = dt.datetime.fromisoformat(day + "T20:00:00+00:00")
    store = hist.timeseries.get_range(
        dataset="XNAS.ITCH", schema="ohlcv-1m",
        symbols=[sym], stype_in="raw_symbol",
        start=start.isoformat(), end=end.isoformat(),
    )
    df = store.to_df()
    if df.empty: return df
    df = df[["open","high","low","close","volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_convert("America/New_York")
    df5 = df.resample("5min").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()
    return df5


for label, day, sym in CASES:
    df5 = fetch_day(sym, day)
    if df5.empty:
        print(f"!! no data for {label} {day}"); continue

    direction = "up" if df5["close"].iloc[-1] >= df5["open"].iloc[0] else "down"
    # Rolling SPT peak
    rolling = []
    for end in range(15, len(df5) + 1):
        rolling.append(_score_small_pullback_trend(df5.iloc[end-15:end], direction))
    spt_peak = max(rolling) if rolling else 0.0
    res = score_gap(df5, prior_close=float(df5.iloc[0]["open"]),
                    gap_direction=direction, ticker=sym)

    out = OUT_DIR / f"{label}_{sym}_{day}.png"
    title = f"{sym} {day} — {label}  ·  SPT peak {spt_peak:.2f}  ·  URG {res['urgency']:.1f}  ·  day_type {res['day_type']}"
    render_chart(
        ticker=sym, timeframe="5m", lookback=len(df5),
        annotations=[], overlays=[],
        title=title, output_path=str(out),
        df=df5, company=sym, theme="dark_color",
    )
    print(f"wrote {out}  spt_peak={spt_peak:.2f}  urgency={res['urgency']}  day_type={res['day_type']}")

print("done")
