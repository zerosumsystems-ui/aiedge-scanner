"""1-minute → 5-minute bar resampling for the scanner pipeline.

Extracted from shared/brooks_score.py (Phase 3i). The local
`_resample_to_5min` in `backfill_historical_databento.py` is a
separate implementation (ETH-vs-RTH aware) and is not affected.
"""

import pandas as pd


# Default 5-min resample from 1-min bars for scoring
SCAN_BAR_SCHEMA = "ohlcv-1m"
SCAN_RESAMPLE = "5min"


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
