"""TEST 2b: Check symbol column format in daily data"""
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

print(f"Time: {t1-t0:.1f}s, Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"dtypes:\n{df.dtypes}")
print(f"\nSymbol column sample:")
print(df["symbol"].head(20).tolist())
print(f"\nSymbol nunique: {df['symbol'].nunique()}")
print(f"Symbol value_counts head:")
print(df["symbol"].value_counts().head(10))
print(f"\nNull symbols: {df['symbol'].isna().sum()}")
print(f"Empty symbols: {(df['symbol'] == '').sum()}")

# Filter liquid
mask = (df["volume"] > 50000) & (df["close"] > 5.0)
filtered = df.loc[mask]
print(f"\nFiltered (vol>50k, close>$5): {len(filtered)} symbols")
print(f"Sample filtered symbols: {filtered['symbol'].head(20).tolist()}")
