import databento as db
import time
import os
import json

# Load key from env file
with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()
results = {}

# TEST 1: ohlcv-1d ALL_SYMBOLS for Apr 10 (complete session)
print("=" * 60)
print("TEST 1: ohlcv-1d ALL_SYMBOLS for April 10")
print("=" * 60, flush=True)
try:
    t0 = time.time()
    data = client.timeseries.get_range(
        dataset="EQUS.MINI",
        schema="ohlcv-1d",
        symbols="ALL_SYMBOLS",
        start="2026-04-10",
        end="2026-04-11",
    )
    df_d = data.to_df()
    t1 = time.time()

    results["t1_time"] = round(t1 - t0, 1)
    results["t1_rows"] = len(df_d)
    results["t1_symbols"] = int(df_d["symbol"].nunique())

    if "volume" in df_d.columns:
        active = int((df_d["volume"] > 0).sum())
        results["t1_active"] = active

        # Filter: volume > 100k shares AND close > $5
        if "close" in df_d.columns:
            mask = (df_d["volume"] > 100000) & (df_d["close"] > 5.0)
            results["t1_liquid_gt5"] = int(mask.sum())
            mask2 = (df_d["volume"] > 50000) & (df_d["close"] > 5.0)
            results["t1_liquid50k_gt5"] = int(mask2.sum())

    print(json.dumps(results, indent=2))
    print(f"Columns: {list(df_d.columns)}")
    print(df_d.head(5).to_string())

except Exception as e:
    results["t1_error"] = str(e)
    print(f"ERROR: {e}")

# TEST 2: ohlcv-1m for ALL_SYMBOLS for Apr 10 (the big one)
print("\n" + "=" * 60)
print("TEST 2: ohlcv-1m ALL_SYMBOLS for April 10")
print("=" * 60, flush=True)
try:
    t0 = time.time()
    data = client.timeseries.get_range(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols="ALL_SYMBOLS",
        start="2026-04-10",
        end="2026-04-11",
    )
    df = data.to_df()
    t1 = time.time()

    results["t2_time"] = round(t1 - t0, 1)
    results["t2_rows"] = len(df)
    results["t2_symbols"] = int(df["symbol"].nunique())
    results["t2_mb"] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1)

    if "volume" in df.columns:
        vol_by_sym = df.groupby("symbol")["volume"].sum()
        results["t2_active"] = int((vol_by_sym > 0).sum())
        results["t2_liquid_100k"] = int((vol_by_sym > 100000).sum())
        results["t2_liquid_1m"] = int((vol_by_sym > 1000000).sum())

        if "close" in df.columns:
            last_prices = df.groupby("symbol")["close"].last()
            results["t2_price_gt5"] = int((last_prices > 5.0).sum())
            results["t2_price_gt10"] = int((last_prices > 10.0).sum())

    if len(df) > 0:
        results["t2_first_ts"] = str(df.index[0])
        results["t2_last_ts"] = str(df.index[-1])

    print(json.dumps(results, indent=2))

except Exception as e:
    results["t2_error"] = str(e)
    print(f"ERROR: {e}")

# TEST 3: ohlcv-1m for top 2000 by volume (Option D comparison)
print("\n" + "=" * 60)
print("TEST 3: ohlcv-1m for top 2000 liquid symbols")
print("=" * 60, flush=True)
try:
    if "vol_by_sym" in dir():
        top2000 = vol_by_sym.nlargest(2000).index.tolist()
        t0 = time.time()
        data2k = client.timeseries.get_range(
            dataset="EQUS.MINI",
            schema="ohlcv-1m",
            symbols=top2000,
            start="2026-04-10",
            end="2026-04-11",
        )
        df2k = data2k.to_df()
        t1 = time.time()
        results["t3_time"] = round(t1 - t0, 1)
        results["t3_rows"] = len(df2k)
        results["t3_symbols"] = int(df2k["symbol"].nunique())
        print(f"Top 2000 pull: {t1-t0:.1f}s, {len(df2k):,} rows")
except Exception as e:
    results["t3_error"] = str(e)
    print(f"ERROR: {e}")

# TEST 4: Check today's data availability (intraday lag)
print("\n" + "=" * 60)
print("TEST 4: Today's data availability check")
print("=" * 60, flush=True)
try:
    t0 = time.time()
    data_today = client.timeseries.get_range(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=["AAPL", "MSFT", "SPY"],
        start="2026-04-13",
    )
    df_today = data_today.to_df()
    t1 = time.time()
    results["t4_today_rows"] = len(df_today)
    if len(df_today) > 0:
        results["t4_first"] = str(df_today.index[0])
        results["t4_last"] = str(df_today.index[-1])
    print(f"Today rows: {len(df_today)}, time: {t1-t0:.1f}s")
    if len(df_today) > 0:
        print(f"First: {df_today.index[0]}, Last: {df_today.index[-1]}")
except Exception as e:
    results["t4_error"] = str(e)
    print(f"ERROR: {e}")

# Save full results
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(json.dumps(results, indent=2, default=str))

with open(os.path.expanduser("~/video-pipeline/benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved to ~/video-pipeline/benchmark_results.json")
print("DONE")
