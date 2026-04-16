"""Re-render dashboard.html from today's saved session bars to confirm SPT
appears on cards. Reproduces the live-scanner scoring loop on stored data,
then calls the same dashboard generator.
"""
from __future__ import annotations
import sys, pickle, datetime as dt
from pathlib import Path

ROOT = Path.home() / "video-pipeline"
sys.path.insert(0, str(ROOT))

import pandas as pd

from shared.brooks_score import score_gap
import live_scanner as ls

SESSION_FILE = ROOT / "logs" / "live_scanner" / "2026-04-14_session.pkl"
DAYFILE      = ROOT / "logs" / "live_scanner" / "2026-04-14.json"

print(f"Loading session bars: {SESSION_FILE}")
with SESSION_FILE.open("rb") as f:
    bars = pickle.load(f)

print(f"Got {len(bars):,} symbols")

# Score
results: list[dict] = []
df5_cache: dict[str, pd.DataFrame] = {}
for sym, bar_list in bars.items():
    if len(bar_list) < ls.MIN_BARS:
        continue
    df = pd.DataFrame(bar_list)
    if df["close"].mean() < ls.MIN_PRICE:
        continue
    if (df["close"] * df["volume"]).mean() < ls.MIN_DOLLAR_VOL:
        continue
    pc = ls.prior_closes.get(sym) if hasattr(ls, "prior_closes") else None
    # We can't fetch fresh prior closes here; synthesize from first bar
    pc = float(df.iloc[0]["open"])
    df5 = ls.resample_to_5min(df)
    if len(df5) < 2:
        continue
    direction = "up" if df5["close"].iloc[-1] >= df5["open"].iloc[0] else "down"
    score = score_gap(df5, prior_close=pc, gap_direction=direction, ticker=sym)
    score["_prior_close"] = pc
    ls.annotate_adr_multiple(score, df, sym)
    results.append(score)
    df5_cache[sym] = df5

results.sort(key=lambda x: -x.get("urgency", 0))
results = ls._dedup_etf_families(results)
results = ls._compute_movement(results, {})
top = results[:25]

print(f"Scored: {len(results)} | top urgency: {top[0]['urgency'] if top else 0}")
for r in top[:5]:
    d = r.get("details", {})
    print(f"  {r['ticker']:<6} URG {r['urgency']:.1f}  spike {d.get('spike_quality',0):.1f}  pull {d.get('pullback_quality',0):.1f}  SPT {d.get('small_pullback_trend',0):.2f}  ({r.get('day_type')})")

# Render dashboard
ls._generate_dashboard(top, df5_cache,
                       now_et=dt.datetime.now(ls.ET),
                       total_symbols=len(bars), passed=len(results),
                       elapsed=0.0, interval_min=5)
print("\nDashboard regenerated → ~/video-pipeline/logs/live_scanner/dashboard.html")
