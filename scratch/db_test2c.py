"""TEST 2c: Get symbology map and do proper filtering"""
import databento as db
import time
import os
import json

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

# Pull daily data with symbology
t0 = time.time()
data = client.timeseries.get_range(
    dataset="EQUS.MINI",
    schema="ohlcv-1d",
    symbols="ALL_SYMBOLS",
    start="2026-04-10",
    end="2026-04-11",
    stype_in="instrument_id",
    stype_out="raw_symbol",
)
df = data.to_df()
t1 = time.time()

# Check for symbology mapping
print(f"Time: {t1-t0:.1f}s, Rows: {len(df)}")

# Try the symbology map from DBNStore
print(f"\nType of data: {type(data)}")
print(f"Has symbology: {hasattr(data, 'symbology')}")

# Check what methods are available
attrs = [a for a in dir(data) if not a.startswith('_')]
print(f"Data attrs: {attrs[:30]}")

# Try to get symbol map
if hasattr(data, 'symbology'):
    sym = data.symbology
    print(f"Symbology type: {type(sym)}")
    if hasattr(sym, 'mappings'):
        print(f"Mappings count: {len(sym.mappings) if hasattr(sym.mappings, '__len__') else 'N/A'}")

# Check the to_df with map_symbols
try:
    df2 = data.to_df(map_symbols=True)
    print(f"\nWith map_symbols=True:")
    print(f"Symbol sample: {df2['symbol'].head(10).tolist()}")
    print(f"Symbol nunique: {df2['symbol'].nunique()}")
except Exception as e:
    print(f"map_symbols error: {e}")

# Alternative: try pretty_ts
try:
    df3 = data.to_df(pretty_ts=True)
    print(f"\nWith pretty_ts:")
    print(df3.head(3).to_string())
except Exception as e:
    print(f"pretty_ts error: {e}")

# Check metadata
if hasattr(data, 'metadata'):
    print(f"\nMetadata: {data.metadata}")

# instrument_id mapping
print(f"\ninstrument_id sample: {df['instrument_id'].head(10).tolist()}")
print(f"instrument_id nunique: {df['instrument_id'].nunique()}")
