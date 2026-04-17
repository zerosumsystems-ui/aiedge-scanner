"""1-minute → 5-minute bar resampling.

Two variants:
  - `_resample_to_5min(df)` — simple private helper used by the
    historical scan pipeline (scan_universe). Input is already
    DatetimeIndex-ed and RTH-filtered; no partial-bar handling.
  - `resample_to_5min(df, now=None)` — public live-scanner variant
    that accepts a `datetime` column, localizes to ET, forward-fills
    OHLC, and drops the most recent 5-min bar if its
    [start, start+5m) window hasn't closed yet. Passing `now` lets
    tests fix the wall-clock.

The local `_resample_to_5min` in `backfill_historical_databento.py`
is a separate implementation (ETH-vs-RTH aware) and is not affected
by either of these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import pytz


# Default 5-min resample from 1-min bars for scoring
SCAN_BAR_SCHEMA = "ohlcv-1m"
SCAN_RESAMPLE = "5min"

# Eastern timezone — canonical reference for NYSE session timing.
# Also exported from aiedge.data.databento; defining here avoids a
# circular import (databento → resample for SCAN_BAR_SCHEMA).
ET = pytz.timezone("America/New_York")


def _resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 5-min bars for scoring.

    Input must have columns: open, high, low, close, volume.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df_1m.columns:
        agg["volume"] = "sum"

    df_5m = df_1m.resample(SCAN_RESAMPLE).agg(agg).dropna(subset=["open", "close"])
    return df_5m


def resample_to_5min(df: pd.DataFrame, now: Optional[datetime] = None) -> pd.DataFrame:
    """Resample a 1-min OHLCV DataFrame to 5-min bars (live-scanner variant).

    Input has a `datetime` *column* (not index), optionally tz-naive
    — this function localizes to ET if naive. Forward-fills OHLC
    across any gap bars and zero-fills volume. Drops the most recent
    5-min bar if its [start, start+5m) window hasn't closed yet,
    preventing partial-bar leakage into scoring.

    Pass `now` to test at a specific wall-clock time; defaults to
    real now in ET.
    """
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
    df5["open"] = df5["open"].ffill()
    df5["high"] = df5["high"].ffill()
    df5["low"] = df5["low"].ffill()
    df5["close"] = df5["close"].ffill()
    df5["volume"] = df5["volume"].fillna(0)
    df5 = df5.dropna(subset=["open", "close"])
    df5 = df5[df5["open"] > 0]

    # Defensive: if the last bar's 5-min window hasn't closed yet, drop it.
    # A bar labeled 09:40 covers [09:40, 09:45). At any wall-clock time before
    # 09:45 that bar is still forming and must not feed scoring.
    if len(df5) > 0:
        wall_raw = now if now is not None else datetime.now(ET)
        wall = pd.Timestamp(wall_raw)
        if wall.tz is None:
            wall = wall.tz_localize(ET)
        else:
            wall = wall.tz_convert(ET)
        last_bar_end = df5.index[-1] + pd.Timedelta(minutes=5)
        if wall < last_bar_end:
            df5 = df5.iloc[:-1]

    return df5.reset_index().rename(columns={"datetime": "datetime"})
