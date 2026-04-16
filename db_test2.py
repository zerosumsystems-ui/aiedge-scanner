"""TEST 2: ohlcv-1m for filtered ~1700 liquid symbols (Option A phase 2)"""
import databento as db
import time
import os
import json
import pandas as pd

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

# Load daily data to get filtered symbol list
df_d = pd.read_parquet("/tmp/equs_daily_apr10.parquet")
# Filter: volume > 50k AND close > $5
mask = (df_d["volume"] > 50000) & (df_d["close"] > 5.0)
filtered_syms = df_d.loc[mask, "symbol"].unique().tolist()
print(f"Filtered symbols: {len(filtered_syms)}")

# Pull ohlcv-1m for filtered symbols
t0 = time.time()
data = client.timeseries.get_range(
    dataset="EQUS.MINI",
    schema="ohlcv-1m",
    symbols=filtered_syms,
    start="2026-04-10",
    end="2026-04-11",
)
df = data.to_df()
t1 = time.time()

r = {
    "filter_count": len(filtered_syms),
    "time_sec": round(t1 - t0, 1),
    "rows": len(df),
    "mem_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1),
    "unique_symbols": int(df["symbol"].nunique()),
    "bars_per_symbol_avg": round(len(df) / max(df["symbol"].nunique(), 1), 0),
}

print(json.dumps(r, indent=2))
