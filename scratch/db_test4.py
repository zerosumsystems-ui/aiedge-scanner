"""TEST 4: Check today's intraday data availability (lag test)"""
import databento as db
import time
import os
import json

with open(os.path.expanduser("~/video-pipeline/credentials/.env")) as f:
    for line in f:
        if "DATABENTO_API_KEY" in line:
            os.environ["DATABENTO_API_KEY"] = line.strip().split("=", 1)[1]

client = db.Historical()

# Try to get today's data for a few symbols
t0 = time.time()
try:
    data = client.timeseries.get_range(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=["AAPL", "MSFT", "SPY", "NVDA", "TSLA"],
        start="2026-04-13",
    )
    df = data.to_df()
    t1 = time.time()

    r = {
        "time_sec": round(t1 - t0, 1),
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
    }
    if len(df) > 0:
        r["first_ts"] = str(df.index[0])
        r["last_ts"] = str(df.index[-1])
        r["bars_count"] = int(len(df))
        # Show last few bars
        print("Last 5 bars:")
        print(df.tail(5).to_string())
    else:
        r["note"] = "No data returned for today"

    print(json.dumps(r, indent=2, default=str))

except Exception as e:
    print(f"Error: {e}")
    # Check what the latest available timestamp is
    print("Checking latest available data...")
    try:
        data2 = client.timeseries.get_range(
            dataset="EQUS.MINI",
            schema="ohlcv-1m",
            symbols=["AAPL"],
            start="2026-04-11",
        )
        df2 = data2.to_df()
        if len(df2) > 0:
            print(f"Latest available: {df2.index[-1]}")
            print(f"Rows: {len(df2)}")
    except Exception as e2:
        print(f"Error2: {e2}")
