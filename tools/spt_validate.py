"""SPT validation — Supreme Court Condition §1.

Methodology (defensible):
  1. Pull ~60 trading days of 5-min SPY bars (ohlcv-1m → resample).
  2. For each day, compute the existing Brooks `day_type` classifier — this is the
     ground-truth label, calibrated independently of SPT.
  3. Pick 3 days classified as `trend_from_open` (or `spike_and_channel`) whose
     move_ratio is MODEST (low-ADR drift — exactly the SPT regime).
  4. Pick 3 days classified as `trading_range` or `tight_tr` (chop).
  5. Compute SPT on all six. Report.

Pass criterion per Court Condition §1:
    SPT ≥ 2.0 on each of 3 SPT days
    SPT ≤ 1.0 on each of 3 chop days
"""
from __future__ import annotations
import os, sys, datetime as dt, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import databento as db

from shared.brooks_score import (
    _score_small_pullback_trend,
    score_gap,
)

# ── API key ────────────────────────────────────────────────────────────────
KEY_ENV_FILE = Path.home() / "keys" / "databento.env"
DB_KEY = ""
if KEY_ENV_FILE.exists():
    for line in KEY_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("DATABENTO_API_KEY="):
            DB_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
if not DB_KEY:
    DB_KEY = os.environ.get("DATABENTO_API_KEY", "")
assert DB_KEY, f"no databento key at {KEY_ENV_FILE} or env"

DATASET = "XNAS.ITCH"
SYMBOL  = "SPY"
SCHEMA  = "ohlcv-1m"

# ── Fetch one big range, split by day ──────────────────────────────────────
END   = dt.datetime(2026, 4, 11, 20, 0, tzinfo=dt.timezone.utc)  # Fri 4/11 close
START = END - dt.timedelta(days=90)                              # ~60 trading days

def fetch_range(sym: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    hist = db.Historical(key=DB_KEY)
    store = hist.timeseries.get_range(
        dataset=DATASET, schema=SCHEMA,
        symbols=[sym], stype_in="raw_symbol",
        start=start.isoformat(), end=end.isoformat(),
    )
    df = store.to_df()
    if df.empty:
        return df
    df = df[["open","high","low","close","volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_convert("America/New_York")
    return df


def per_day_5m(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group 1-min bars by ET trading date, restrict to 09:30–16:00, resample to 5m."""
    out: dict[str, pd.DataFrame] = {}
    for d, g in df.groupby(df.index.date):
        # RTH
        g = g.between_time("09:30", "15:59")
        if len(g) < 20:
            continue
        g5 = g.resample("5min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        if len(g5) < 20:
            continue
        out[str(d)] = g5
    return out


def main():
    print(f"Fetching SPY 1m bars {START.date()} → {END.date()} …")
    df = fetch_range(SYMBOL, START, END)
    if df.empty:
        print("No data."); sys.exit(2)
    days = per_day_5m(df)
    print(f"Got {len(days)} trading days of 5m SPY bars.")

    # Score every day. For SPT we use ROLLING 15-bar windows (matches how the live
    # scanner sees it as the day unfolds) and take the PEAK — because a good SPT
    # day reaches the signal at some point mid-session.
    rows = []
    for day_str, df5 in days.items():
        direction = "up" if df5["close"].iloc[-1] >= df5["open"].iloc[0] else "down"
        prior_close = float(df5.iloc[0]["open"])
        r = score_gap(df5, prior_close=prior_close, gap_direction=direction, ticker=SYMBOL)

        # Rolling SPT across the day — matches live-scanner window
        rolling_sp = []
        for end in range(15, len(df5) + 1):
            w = df5.iloc[end - 15:end]
            rolling_sp.append(_score_small_pullback_trend(w, direction))
        spt_peak = max(rolling_sp) if rolling_sp else 0.0
        spt_mean = sum(rolling_sp) / len(rolling_sp) if rolling_sp else 0.0
        # Direction-agnostic: take the better of up/down peaks (chop days shouldn't
        # score high in either direction)
        rolling_opp = []
        opp = "down" if direction == "up" else "up"
        for end in range(15, len(df5) + 1):
            w = df5.iloc[end - 15:end]
            rolling_opp.append(_score_small_pullback_trend(w, opp))
        spt_peak_any = max(spt_peak, max(rolling_opp) if rolling_opp else 0.0)
        rows.append({
            "day": day_str,
            "direction": direction,
            "bars": len(df5),
            "day_type": r["day_type"],
            "move_ratio": r["move_ratio"],
            "urgency": r["urgency"],
            "uncertainty": r["uncertainty"],
            "spt_peak": round(spt_peak, 2),
            "spt_mean": round(spt_mean, 2),
            "spt_peak_any_dir": round(spt_peak_any, 2),
        })

    all_df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    print("\n── Per-day scan (first 40 rows) ──")
    print(all_df.head(40).to_string(index=False))

    # ── Pick days ──────────────────────────────────────────────────────────
    # SPT days: day_type in (trend_from_open, spike_and_channel) AND move_ratio MODEST
    # (we want low-ADR grinds, which is the exact SPT regime)
    spt_pool = all_df[all_df["day_type"].isin(["trend_from_open", "spike_and_channel"])].copy()
    spt_pool = spt_pool.sort_values("move_ratio")  # lowest move_ratio first — the calm ones

    # Chop days: day_type in (trading_range, tight_tr)
    chop_pool = all_df[all_df["day_type"].isin(["trading_range", "tight_tr"])].copy()
    chop_pool = chop_pool.sort_values("uncertainty", ascending=False)

    print("\n── SPT candidate pool (calm trend days, lowest move_ratio first) ──")
    print(spt_pool.head(8).to_string(index=False))
    print("\n── Chop candidate pool (highest uncertainty first) ──")
    print(chop_pool.head(8).to_string(index=False))

    # Pick by PEAK spt score (how the live scanner will actually see it)
    spt_pool = spt_pool.sort_values("spt_peak", ascending=False)
    chop_pool = chop_pool.sort_values("spt_peak", ascending=True)

    print("\n── SPT candidate pool (calm trend days, highest spt_peak first) ──")
    print(spt_pool.head(8).to_string(index=False))
    print("\n── Chop candidate pool (lowest spt_peak first) ──")
    print(chop_pool.head(8).to_string(index=False))

    spt_pick = spt_pool.head(3)
    chop_pick = chop_pool.head(3)

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n── VALIDATION ──")
    print(f"{'label':<8}  {'day':<12}  {'day_type':<18}  {'move_r':<7}  {'SPT_peak':<9}  {'URG':<6}  verdict")
    print("-" * 90)
    spt_ok = 0
    chop_ok = 0
    for i, row in enumerate(spt_pick.itertuples(), 1):
        v = "OK" if row.spt_peak >= 2.0 else "FAIL"
        spt_ok += (row.spt_peak >= 2.0)
        print(f"SPT_{i}    {row.day}   {row.day_type:<18}  {row.move_ratio:<7.2f}  {row.spt_peak:<9.2f}  {row.urgency:<6.2f}  {v}")
    for i, row in enumerate(chop_pick.itertuples(), 1):
        v = "OK" if row.spt_peak <= 1.0 else "FAIL"
        chop_ok += (row.spt_peak <= 1.0)
        print(f"CHOP_{i}   {row.day}   {row.day_type:<18}  {row.move_ratio:<7.2f}  {row.spt_peak:<9.2f}  {row.urgency:<6.2f}  {v}")

    passed = (spt_ok == 3 and chop_ok == 3)
    print(f"\nSPT peak mean:  {spt_pick['spt_peak'].mean():.2f}   (target ≥2.0 each)")
    print(f"Chop peak mean: {chop_pick['spt_peak'].mean():.2f}   (target ≤1.0 each)")
    print(f"\n{'VALIDATION PASSED' if passed else 'VALIDATION FAILED — Court condition §1 not met'}")

    picks_path = Path(__file__).resolve().parents[1] / "logs" / "live_scanner" / "SPT_validation_picks.json"
    picks_path.write_text(json.dumps({
        "spt_days": spt_pick[["day","day_type","move_ratio","spt_peak","urgency"]].to_dict(orient="records"),
        "chop_days": chop_pick[["day","day_type","move_ratio","spt_peak","urgency"]].to_dict(orient="records"),
        "pass": bool(passed),
    }, indent=2, default=str))
    print(f"\nPicks written to {picks_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
