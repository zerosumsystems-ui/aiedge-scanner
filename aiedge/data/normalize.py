"""Databento DataFrame normalization.

Lowercase columns, coerce the timestamp index to UTC. Extracted from
shared/brooks_score.py (Phase 3i).
"""

import pandas as pd


def _normalize_databento_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Databento DataFrame column names to lowercase.

    Also handles the `ts_event` index and `symbol` column that
    Databento returns, and ensures the index is UTC-localized.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    return df
