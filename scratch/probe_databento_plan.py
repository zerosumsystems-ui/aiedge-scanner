#!/usr/bin/env python3
"""
Databento plan probe — run on Will's real machine (not sandbox).

Will: run this script with `python3 probe_databento_plan.py`. It loads the
Databento key from credentials/.env and hits Databento's metadata +
timeseries endpoints to discover what datasets and schemas your current
(post-cancellation) plan can actually access.

Goals:
  1. List all datasets the account can see.
  2. Probe the candidate datasets (XNAS.ITCH, XNAS.BASIC, DBEQ.BASIC,
     EQUS.MINI, EQUS.SUMMARY, EQUS.BASIC) for schema availability and
     dataset date range.
  3. Run tiny SPY trial queries against each (dataset, schema) pair:
     - ohlcv-1d for last 5 days, ending YESTERDAY midnight UTC → T-1
       historical (should work on pay-per-GB for any dataset we have
       historical access to)
     - ohlcv-1d for last 2 days, ending TODAY 4 AM UTC → "live" boundary
       (this is the one that started failing on XNAS.ITCH after the cancel)
     - ohlcv-1m for today's session, 13:30 → 14:30 UTC → pure live data
       (the exact query the gap pipeline makes)

Output: a clean PASS/FAIL grid for each combination, with the error class
when a probe fails. This tells us exactly what you can and can't do on
your current plan without burning hundreds of GB on trial.

Cost: each probe is a single symbol (SPY) over a tiny window — well under
1 MB of data per call. Total cost < $0.05 at pay-per-GB rates.

Safety: all queries are READ-ONLY via the historical client. No uploads.
No account mutation. No side effects.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Load credentials ─────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
env_path = REPO / "credentials" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    import databento as db
except ImportError:
    print("ERROR: databento not installed. Run: pip install databento")
    sys.exit(1)

key = os.environ.get("DATABENTO_API_KEY")
if not key:
    print("ERROR: DATABENTO_API_KEY not set (check credentials/.env)")
    sys.exit(1)

client = db.Historical(key)

# ── 1. List datasets ─────────────────────────────────────────────────────
print("=" * 72)
print(" 1. Datasets this account can see")
print("=" * 72)
try:
    datasets = client.metadata.list_datasets()
    for d in datasets:
        print(f"  {d}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

CANDIDATES = [
    "XNAS.ITCH",    # the one the gap pipeline currently uses
    "XNAS.BASIC",   # cheaper NASDAQ feed
    "DBEQ.BASIC",   # deprecated Jan 2025 → folded into EQUS.*
    "EQUS.MINI",    # consolidated, lightest
    "EQUS.BASIC",   # consolidated, standard
    "EQUS.SUMMARY", # consolidated EOD OHLCV, zero license fees
]

# ── 2. Per-dataset schema availability ───────────────────────────────────
print()
print("=" * 72)
print(" 2. Per-dataset schema list (metadata.list_schemas)")
print("=" * 72)
for ds in CANDIDATES:
    print(f"\n  {ds}")
    try:
        schemas = client.metadata.list_schemas(dataset=ds)
        print(f"    schemas: {schemas}")
    except Exception as e:
        msg = str(e)[:160]
        print(f"    ERROR: {type(e).__name__}: {msg}")

# ── 3. Per-dataset date range ────────────────────────────────────────────
print()
print("=" * 72)
print(" 3. Per-dataset range (metadata.get_dataset_range)")
print("=" * 72)
for ds in CANDIDATES:
    print(f"\n  {ds}")
    try:
        rng = client.metadata.get_dataset_range(dataset=ds)
        start = rng.get("start_date") or rng.get("start") or "?"
        end = rng.get("end_date") or rng.get("end") or "?"
        print(f"    overall: {start} → {end}")
        sch_info = rng.get("schema") or {}
        for name in ("ohlcv-1d", "ohlcv-1h", "ohlcv-1m", "trades", "tbbo"):
            if name in sch_info:
                s_end = sch_info[name].get("end") or sch_info[name].get("end_date") or "?"
                s_start = sch_info[name].get("start") or sch_info[name].get("start_date") or "?"
                print(f"    {name:<10} {s_start} → {s_end}")
    except Exception as e:
        msg = str(e)[:160]
        print(f"    ERROR: {type(e).__name__}: {msg}")

# ── 4. Trial queries ─────────────────────────────────────────────────────
print()
print("=" * 72)
print(" 4. Trial queries — can we actually pull SPY data?")
print("=" * 72)
now_utc = datetime.now(timezone.utc)
today_mid = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
yday_mid = today_mid - timedelta(days=1)

# End timestamps for the different "periods":
#   T-1 historical:  ends at yday_mid (yesterday midnight UTC)
#   T-0 boundary:    ends at today_mid + 4 hours (04:00 UTC is the exact
#                    cutoff mentioned in the error message)
#   live intraday:   13:30 UTC (NYSE open) to now - 30 min

probes = []
for ds in CANDIDATES:
    # T-1 historical, 5 days of daily bars
    probes.append((ds, "ohlcv-1d", yday_mid - timedelta(days=5), yday_mid, "T-1 hist"))
    # T-0 boundary, daily bars ending 04:00 UTC today
    probes.append((ds, "ohlcv-1d", yday_mid, today_mid + timedelta(hours=4), "T-0 boundary"))
    # Live intraday — 1-min bars from 13:30 UTC onwards
    live_start = today_mid + timedelta(hours=13, minutes=30)
    live_end = min(now_utc - timedelta(minutes=30), live_start + timedelta(hours=1))
    if live_end > live_start:
        probes.append((ds, "ohlcv-1m", live_start, live_end, "live intraday"))

results = []
for dataset, schema, start, end, label in probes:
    tag = f"{dataset:<13} {schema:<9} {label:<14}"
    try:
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=["SPY"],
            schema=schema,
            start=start,
            end=end,
            stype_in="raw_symbol",
        )
        df = data.to_df() if hasattr(data, "to_df") else data
        n = len(df) if hasattr(df, "__len__") else 0
        status = "OK" if n > 0 else "EMPTY"
        results.append((dataset, schema, label, status, f"{n} rows"))
        print(f"  [{status:<14}] {tag} → {n} rows")
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "live data license" in low:
            cls = "NO_LIVE_LICENSE"
        elif "not authorized" in low or "unauthorized" in low or "forbidden" in low:
            cls = "NOT_AUTHORIZED"
        elif "dataset_unavailable_range" in low:
            cls = "OUT_OF_RANGE"
        elif "not fully available" in low or "data_schema_not_fully_available" in low:
            cls = "PARTIAL_AVAIL"
        elif "not_found" in low or "invalid" in low:
            cls = "UNKNOWN_DATASET"
        else:
            cls = "ERROR"
        short = msg if len(msg) < 120 else msg[:117] + "..."
        results.append((dataset, schema, label, cls, short))
        print(f"  [{cls:<14}] {tag} → {short}")

# ── 5. Summary grid ──────────────────────────────────────────────────────
print()
print("=" * 72)
print(" 5. Summary — gap pipeline viability on current plan")
print("=" * 72)
print()
print(f"  {'Dataset':<13} {'Schema':<9} {'Period':<14} Status")
print(f"  {'-'*13} {'-'*9} {'-'*14} {'-'*14}")
for r in results:
    print(f"  {r[0]:<13} {r[1]:<9} {r[2]:<14} {r[3]}")
print()
print("Interpretation:")
print("  - T-1 hist   PASS → cheap historical queries work (yesterday and earlier).")
print("  - T-0 bdry   PASS → ohlcv-1d for today's session available at 04:00 UTC.")
print("  - live intraday PASS → live data license is active for this dataset+schema.")
print()
print("If ALL T-1 hist rows PASS but ALL live intraday rows FAIL with")
print("NO_LIVE_LICENSE, the conclusion is: you need a live subscription to run")
print("the gap pipelines today. See report for the exact plan options.")
