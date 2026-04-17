"""TEST 1: ohlcv-1d ALL_SYMBOLS — lightweight, should be fast"""
import databento as db
import time
import os
import json

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

t0 = time.time()
data = client.timeseries.get_range(
    dataset="EQUS.MINI",
    schema="ohlcv-1d",
    symbols="ALL_SYMBOLS",
    start="2026-04-10",
    end="2026-04-11",
)
df = data.to_df()
t1 = time.time()

r = {
    "time_sec": round(t1 - t0, 1),
    "rows": len(df),
    "symbols": int(df["symbol"].nunique()),
    "columns": list(df.columns),
}

if "volume" in df.columns and "close" in df.columns:
    r["active_vol_gt0"] = int((df["volume"] > 0).sum())
    mask_liq = (df["volume"] > 100000) & (df["close"] > 5.0)
    r["liquid_vol100k_price5"] = int(mask_liq.sum())
    mask_liq2 = (df["volume"] > 50000) & (df["close"] > 5.0)
    r["liquid_vol50k_price5"] = int(mask_liq2.sum())
    mask_liq3 = (df["volume"] > 500000) & (df["close"] > 5.0)
    r["liquid_vol500k_price5"] = int(mask_liq3.sum())

print(json.dumps(r, indent=2))

# Save for later tests
df.to_parquet("/tmp/equs_daily_apr10.parquet")
print("Saved daily data to /tmp/equs_daily_apr10.parquet")
