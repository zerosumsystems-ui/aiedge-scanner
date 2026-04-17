#!/usr/bin/env python3
"""export_calibration.py — build time-split calibration diagram for /review.

Loads WIN/LOSS detections from Pattern Lab, sorts by detection_date, splits
60/40:
  - Train 60%  → build an in-memory PriorsStore
  - Test  40%  → for each, compute predicted p_win via priors.p_win(...) with
                 min_samples=30, pair with realized outcome

Writes ~/code/aiedge/site/public/calibration.json with:
  - bins: reliability_table() rows
  - brier, ece, overall_hit_rate
  - n_test, coverage stats

Avoids leakage because the test detection's outcome was never written to
the in-memory store. Reuses Databento fetch logic from backfill_priors.py.

Usage:
    python3 scripts/export_calibration.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import databento as db
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiedge.analysis.reliability import (
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from aiedge.context.htf import classify_htf_alignment
from aiedge.features.regime import realized_vol_tercile
from aiedge.risk.priors import p_win
from aiedge.storage.priors_store import PriorsStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_calibration")

DATASET = "EQUS.MINI"
DAILY_LOOKBACK_DAYS = 90
MIN_DAILY_BARS_FOR_REGIME = 10
MIN_DAILY_BARS_FOR_HTF = 25
MIN_SAMPLES = 30


def _load_databento_key() -> str:
    if os.environ.get("DATABENTO_API_KEY"):
        return os.environ["DATABENTO_API_KEY"]
    key_path = Path.home() / "keys" / "databento.env"
    if key_path.exists():
        for line in key_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
    if not os.environ.get("DATABENTO_API_KEY"):
        raise RuntimeError("DATABENTO_API_KEY not found")
    return os.environ["DATABENTO_API_KEY"]


def _load_detections(sqlite_path: str) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    q = """
        SELECT setup_type, ticker, detection_date, direction, day_type, result
        FROM detections
        WHERE result IN ('WIN', 'LOSS')
          AND detection_date IS NOT NULL
          AND direction IN ('long', 'short')
        ORDER BY detection_date ASC, rowid ASC
    """
    rows = [
        dict(
            setup_type=r[0],
            ticker=r[1],
            detection_date=r[2],
            direction=r[3],
            day_type=r[4] or "undetermined",
            result=r[5],
        )
        for r in conn.execute(q)
    ]
    conn.close()
    return rows


def _fetch_daily_closes_universe(
    api_key: str, tickers: list[str], start_date: date, end_date: date,
) -> dict[str, pd.Series]:
    hist = db.Historical(key=api_key)
    start_with_pad = start_date - timedelta(days=DAILY_LOOKBACK_DAYS)
    logger.info(
        f"Fetching daily closes for {len(tickers)} tickers "
        f"from {start_with_pad} to {end_date}…"
    )
    store = hist.timeseries.get_range(
        dataset=DATASET,
        schema="ohlcv-1d",
        symbols=tickers,
        stype_in="raw_symbol",
        start=start_with_pad.isoformat(),
        end=end_date.isoformat(),
    )
    df = store.to_df()
    if df.empty:
        logger.warning("Databento returned empty DataFrame")
        return {}
    df.columns = [c.lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise RuntimeError(
            f"No 'symbol' column in Databento response (got {df.columns.tolist()})"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts_event", "timestamp", "date"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
    df["_date"] = pd.to_datetime(df.index).date
    closes: dict[str, pd.Series] = {}
    for sym, g in df.groupby("symbol"):
        s = pd.Series(g["close"].values, index=pd.to_datetime(g["_date"]))
        s = s.sort_index()
        closes[sym] = s
    logger.info(
        f"Fetched daily closes for {len(closes)} tickers "
        f"(requested {len(tickers)}; missing {len(tickers) - len(closes)})"
    )
    return closes


def _weekly_from_daily(daily: pd.Series) -> pd.Series:
    if daily.empty:
        return daily
    return daily.resample("W-FRI").last().dropna()


def _regime_for(daily: pd.Series, detection_date: date) -> str | None:
    hist = daily[daily.index.date < detection_date]
    if len(hist) < MIN_DAILY_BARS_FOR_REGIME:
        return None
    return realized_vol_tercile(hist.values.tolist(), lookback_days=20)


def _htf_for(
    daily: pd.Series, detection_date: date, direction: str,
) -> str | None:
    hist = daily[daily.index.date < detection_date]
    if len(hist) < MIN_DAILY_BARS_FOR_HTF:
        return None
    weekly = _weekly_from_daily(hist)
    if len(weekly) < 12:
        return None
    result = classify_htf_alignment(
        hist.values.tolist(),
        weekly.values.tolist(),
        direction,  # type: ignore[arg-type]
    )
    return result["alignment"]


def _compute_stratum(
    det: dict, daily_closes: dict[str, pd.Series],
) -> tuple[str, str, str, str] | None:
    """Return (setup_type, regime, htf_alignment, day_type) or None if unresolvable."""
    ticker = det["ticker"]
    det_date = date.fromisoformat(det["detection_date"])
    daily = daily_closes.get(ticker)
    if daily is None or daily.empty:
        return None
    regime = _regime_for(daily, det_date)
    if regime is None:
        return None
    alignment = _htf_for(daily, det_date, det["direction"])
    if alignment is None:
        return None
    return (det["setup_type"], regime, alignment, det["day_type"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern-lab", default="db/pattern_lab.sqlite")
    ap.add_argument(
        "--out",
        default=str(Path.home() / "code" / "aiedge" / "site" / "public" / "calibration.json"),
    )
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = ap.parse_args()

    scanner_root = Path(__file__).resolve().parent.parent
    os.chdir(scanner_root)

    detections = _load_detections(args.pattern_lab)
    logger.info(f"Loaded {len(detections)} detections from {args.pattern_lab}")
    if not detections:
        logger.error("No detections — nothing to calibrate")
        return 1

    # Time split by detection_date (already sorted asc in query)
    split_idx = int(len(detections) * args.train_frac)
    train = detections[:split_idx]
    test = detections[split_idx:]
    logger.info(
        f"Split: train={len(train)} "
        f"({train[0]['detection_date']}…{train[-1]['detection_date']})  "
        f"test={len(test)} "
        f"({test[0]['detection_date']}…{test[-1]['detection_date']})"
    )

    tickers = sorted({d["ticker"] for d in detections})
    dates = [date.fromisoformat(d["detection_date"]) for d in detections]
    api_key = _load_databento_key()
    daily_closes = _fetch_daily_closes_universe(
        api_key, tickers, min(dates), max(dates),
    )

    # Build scratch priors store from training set
    logger.info("Building in-memory priors from training set…")
    store = PriorsStore(":memory:")
    train_counters = Counter()
    for det in train:
        stratum = _compute_stratum(det, daily_closes)
        if stratum is None:
            train_counters["unresolved"] += 1
            continue
        setup, regime, alignment, day_type = stratum
        won = det["result"] == "WIN"
        store.record_outcome(
            setup_type=setup,
            regime=regime,
            htf_alignment=alignment,
            day_type=day_type,
            won=won,
        )
        train_counters["wrote"] += 1
    logger.info(f"Train priors built: {dict(train_counters)}")

    # Predict on test set
    logger.info("Scoring test detections…")
    predicted: list[float] = []
    outcomes: list[int] = []
    match_levels = Counter()
    test_counters = Counter()
    for det in test:
        stratum = _compute_stratum(det, daily_closes)
        if stratum is None:
            test_counters["unresolved"] += 1
            continue
        setup, regime, alignment, day_type = stratum
        lookup = p_win(
            store,
            setup_type=setup,
            regime=regime,
            htf_alignment=alignment,
            day_type=day_type,
            min_samples=args.min_samples,
        )
        match_levels[lookup.matched_level] += 1
        predicted.append(float(lookup.p_win))
        outcomes.append(1 if det["result"] == "WIN" else 0)
        test_counters["scored"] += 1
    logger.info(f"Test predictions: {dict(test_counters)}")
    logger.info(f"Match levels: {dict(match_levels)}")

    if not predicted:
        logger.error("No test predictions — cannot calibrate")
        return 1

    # Metrics
    table = reliability_table(predicted, outcomes, n_bins=args.n_bins)
    brier = brier_score(predicted, outcomes)
    ece = expected_calibration_error(predicted, outcomes, n_bins=args.n_bins)
    overall_hit = sum(outcomes) / len(outcomes)

    bins = []
    for _, row in table.iterrows():
        bins.append({
            "bin_mid": float(row["bin_midpoint"]),
            "bin_lo": float(row["bin_lo"]),
            "bin_hi": float(row["bin_hi"]),
            "predicted": float(row["mean_predicted"]),
            "empirical": float(row["empirical_hit_rate"]),
            "count": int(row["n"]),
        })

    # Coverage: fraction of scored detections that hit an "exact" match (≥30 samples
    # at fully-specific key) vs fell back vs defaulted to 0.50.
    n_scored = sum(test_counters.values()) if test_counters else 0
    coverage = {
        "exact": match_levels.get("exact", 0),
        "regime_plus_align": match_levels.get("regime+align", 0),
        "regime": match_levels.get("regime", 0),
        "setup": match_levels.get("setup", 0),
        "default": match_levels.get("default", 0),
    }

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pattern_lab_path": args.pattern_lab,
        "train_frac": args.train_frac,
        "min_samples": args.min_samples,
        "n_train": len(train),
        "n_test": len(test),
        "n_test_scored": len(predicted),
        "n_test_unresolved": test_counters.get("unresolved", 0),
        "train_date_range": [train[0]["detection_date"], train[-1]["detection_date"]],
        "test_date_range": [test[0]["detection_date"], test[-1]["detection_date"]],
        "coverage": coverage,
        "bins": bins,
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "overall_hit_rate": round(overall_hit, 4),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    logger.info("")
    logger.info("═══ CALIBRATION ═══")
    logger.info(f"  n_test_scored:    {len(predicted)}")
    logger.info(f"  brier:            {brier:.4f}")
    logger.info(f"  ece:              {ece:.4f}")
    logger.info(f"  overall hit rate: {overall_hit:.3f}")
    logger.info(f"  coverage:         {coverage}")
    logger.info("")
    logger.info(f"Wrote {out_path}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
