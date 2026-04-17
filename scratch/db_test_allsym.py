"""TEST: ALL_SYMBOLS ohlcv-1m brute force (Option B) + symbology fix"""
import databento as db
import time
import os
import json

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

# ---- TEST: Symbology resolve (fix date range) ----
print("SYMBOLOGY TEST:", flush=True)
try:
    t0 = time.time()
    sym_res = client.symbology.resolve(
        dataset="EQUS.MINI",
        symbols=[1, 2, 35, 37, 38, 54, 74, 86, 122, 142],  # sample instrument_ids
        stype_in="instrument_id",
        stype_out="raw_symbol",
        start_date="2026-04-10",
        end_date="2026-04-11",
    )
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s")
    print(f"  Type: {type(sym_res)}")
    if hasattr(sym_res, 'mappings'):
        for k, v in sym_res.mappings.items():
            print(f"  {k} -> {v}")
    else:
        result_str = str(sym_res)
        print(f"  Result: {result_str[:500]}")
except Exception as e:
    print(f"  Error: {e}")

# ---- ALL_SYMBOLS ohlcv-1m (the big pull) ----
print("\nALL_SYMBOLS ohlcv-1m brute force:", flush=True)
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

r = {
    "time_sec": round(t1 - t0, 1),
    "rows": len(df),
    "mem_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1),
    "instruments": int(df["instrument_id"].nunique()),
}

# Volume distribution
if "volume" in df.columns:
    vol_by_inst = df.groupby("instrument_id")["volume"].sum()
    r["vol_gt0"] = int((vol_by_inst > 0).sum())
    r["vol_gt50k"] = int((vol_by_inst > 50000).sum())
    r["vol_gt100k"] = int((vol_by_inst > 100000).sum())
    r["vol_gt1m"] = int((vol_by_inst > 1000000).sum())

    if "close" in df.columns:
        last_close = df.groupby("instrument_id")["close"].last()
        r["price_gt5"] = int((last_close > 5.0).sum())
        r["price_gt10"] = int((last_close > 10.0).sum())

        # Combined filter: vol > 50k AND close > $5
        valid_ids = set(vol_by_inst[vol_by_inst > 50000].index) & set(last_close[last_close > 5.0].index)
        r["combined_vol50k_price5"] = len(valid_ids)

print(json.dumps(r, indent=2))
with open("/tmp/allsym_results.json", "w") as f:
    json.dump(r, f, indent=2)
print("DONE")
