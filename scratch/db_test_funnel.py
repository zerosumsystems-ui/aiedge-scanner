"""Full funnel test: daily filter -> minute pull using instrument_ids"""
import databento as db
import time
import os
import json

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

# ---- PHASE 1: Daily pull for all symbols ----
print("PHASE 1: ohlcv-1d ALL_SYMBOLS", flush=True)
t0 = time.time()
data_d = client.timeseries.get_range(
    dataset="EQUS.MINI",
    schema="ohlcv-1d",
    symbols="ALL_SYMBOLS",
    start="2026-04-10",
    end="2026-04-11",
)
df_d = data_d.to_df()
t1 = time.time()
p1_time = round(t1 - t0, 1)

# Filter by volume and price
mask = (df_d["volume"] > 50000) & (df_d["close"] > 5.0)
liquid_ids = df_d.loc[mask, "instrument_id"].unique().tolist()
print(f"  Time: {p1_time}s | Total: {len(df_d)} | Filtered: {len(liquid_ids)} instrument_ids")

# ---- PHASE 2: Minute data for filtered instruments ----
print(f"\nPHASE 2: ohlcv-1m for {len(liquid_ids)} instruments", flush=True)
t0 = time.time()
data_m = client.timeseries.get_range(
    dataset="EQUS.MINI",
    schema="ohlcv-1m",
    symbols=liquid_ids,
    stype_in="instrument_id",
    start="2026-04-10",
    end="2026-04-11",
)
df_m = data_m.to_df()
t1 = time.time()
p2_time = round(t1 - t0, 1)

mem_mb = round(df_m.memory_usage(deep=True).sum() / 1024 / 1024, 1)
n_instruments = df_m["instrument_id"].nunique()
bars_per = round(len(df_m) / max(n_instruments, 1), 0)

print(f"  Time: {p2_time}s | Rows: {len(df_m):,} | Instruments: {n_instruments} | MB: {mem_mb}")
print(f"  Avg bars/instrument: {bars_per}")

# ---- PHASE 2b: Try to resolve symbols ----
print(f"\nPHASE 2b: Resolve symbology", flush=True)
try:
    # Try symbology resolution
    sym_map = data_m.symbology
    print(f"  Symbology type: {type(sym_map)}, len: {len(sym_map) if hasattr(sym_map, '__len__') else 'N/A'}")
    if isinstance(sym_map, dict) and len(sym_map) > 0:
        sample = dict(list(sym_map.items())[:5])
        print(f"  Sample: {sample}")
except Exception as e:
    print(f"  Symbology error: {e}")

# Try request_symbology
try:
    t0 = time.time()
    sym_res = client.symbology.resolve(
        dataset="EQUS.MINI",
        symbols=liquid_ids[:50],  # test with 50
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date="2026-04-10",
        end_date="2026-04-10",
    )
    t1 = time.time()
    print(f"  Symbology resolve time: {t1-t0:.1f}s")
    if hasattr(sym_res, 'mappings'):
        print(f"  Mappings count: {len(sym_res.mappings)}")
        # Show first few
        for i, (k, v) in enumerate(sym_res.mappings.items()):
            if i < 5:
                print(f"    {k} -> {v}")
    else:
        print(f"  Result type: {type(sym_res)}")
        print(f"  Result: {str(sym_res)[:500]}")
except Exception as e:
    print(f"  Symbology resolve error: {e}")

# ---- SUMMARY ----
total_time = p1_time + p2_time
print(f"\n{'='*60}")
print(f"FUNNEL SUMMARY (Option A)")
print(f"  Phase 1 (daily filter): {p1_time}s -> {len(liquid_ids)} symbols")
print(f"  Phase 2 (minute pull):  {p2_time}s -> {len(df_m):,} rows, {mem_mb} MB")
print(f"  Total time: {total_time}s")
print(f"  Instruments with data: {n_instruments}")

# Save results
r = {
    "p1_time": p1_time, "p1_total": len(df_d), "p1_filtered": len(liquid_ids),
    "p2_time": p2_time, "p2_rows": len(df_m), "p2_mb": mem_mb,
    "p2_instruments": n_instruments, "bars_per_instrument": bars_per,
    "total_time": total_time,
}
print(f"\n{json.dumps(r, indent=2)}")
with open("/tmp/funnel_results.json", "w") as f:
    json.dump(r, f, indent=2)
