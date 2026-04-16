#!/usr/bin/env python3
"""
ADR smoke test — generates a sample dashboard.html and a sample chart tile
using synthetic data so we can visually verify the new ADR multiple UI.

Run:  cd ~/video-pipeline && python tests/adr_smoke_test.py
Opens:
  ~/video-pipeline/logs/live_scanner/dashboard.html
  ~/video-pipeline/tests/out/sample_chart_tile.png
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the live_scanner module so we can reuse its HTML generator.
import live_scanner as ls
from shared.chart_renderer import render_chart

ET = pytz.timezone("America/New_York")
OUT = ROOT / "tests" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def _synthetic_5m(ticker: str, n: int = 40, seed: int = 1, drift: float = 0.5) -> pd.DataFrame:
    """Make a believable 5-min OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    # Anchor to today's regular session
    start = datetime.now(ET).replace(hour=9, minute=30, second=0, microsecond=0)
    idx = pd.date_range(start=start, periods=n, freq="5min", tz=ET)
    price = 50.0
    closes = []
    highs = []
    lows = []
    opens = []
    for i in range(n):
        op = price
        step = rng.normal(drift * 0.05, 0.25)
        cl = op + step
        hi = max(op, cl) + abs(rng.normal(0, 0.15))
        lo = min(op, cl) - abs(rng.normal(0, 0.15))
        opens.append(op); highs.append(hi); lows.append(lo); closes.append(cl)
        price = cl
    vols = rng.integers(50_000, 300_000, size=n)
    return pd.DataFrame({
        "datetime": idx, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


def _mock_score(ticker: str, rank: int, urg: float, unc: float, signal: str,
                adr_mult: float, daily_atr: float, prior_close: float,
                day_type: str = "Trend from Open") -> dict:
    return {
        "ticker": ticker,
        "signal": signal,
        "urgency": urg,
        "uncertainty": unc,
        "day_type": day_type,
        "day_type_warning": "",
        "summary": f"{ticker} mock summary — synthetic data for ADR-multiple UI verification.",
        "_prior_close": prior_close,
        "rank": rank,
        "prev_rank": None,
        "rank_change": None,
        "urgency_delta": None,
        "_first_scan": True,
        "daily_atr": daily_atr,
        "move_ratio": adr_mult * 0.8,   # arbitrary for display
        "adr_multiple": adr_mult,
        "adr_20": daily_atr,
        "today_range": adr_mult * daily_atr,
    }


def main() -> None:
    # Four synthetic tickers showing each ADR tier
    mocks = [
        _mock_score("AAA", 1, 8.6, 2.1, "BUY_PULLBACK", adr_mult=2.35, daily_atr=1.85, prior_close=48.5),
        _mock_score("BBB", 2, 7.9, 3.2, "BUY_SPIKE",    adr_mult=1.72, daily_atr=2.10, prior_close=51.2),
        _mock_score("CCC", 3, 6.1, 4.0, "WAIT",         adr_mult=1.18, daily_atr=1.45, prior_close=42.7),
        _mock_score("DDD", 4, 4.2, 5.3, "FOG",          adr_mult=0.62, daily_atr=0.95, prior_close=33.1,
                    day_type="Tight Trading Range"),
    ]

    df5_map = {m["ticker"]: _synthetic_5m(m["ticker"], seed=i+1, drift=(1.0 if i<2 else 0.2))
               for i, m in enumerate(mocks)}

    # Regenerate the full dashboard via the real generator
    now_et = datetime.now(ET)
    ls._generate_dashboard(mocks, df5_map, now_et,
                           total_symbols=1234, passed=4, elapsed=0.42, interval_min=5)
    print(f"Dashboard → {ls.DASHBOARD_PATH}")

    # Render a single chart tile standalone to a known path for the screenshot
    sample_path = OUT / "sample_chart_tile.png"
    render_chart(
        ticker="AAA",
        timeframe="5min",
        df=df5_map["AAA"],
        output_path=str(sample_path),
        key_levels={"prior_day": {"open": 48.5}},
        theme="dark_color",
        show_volume=True,
        adr_multiple=2.35,
    )
    print(f"Chart tile → {sample_path}")


if __name__ == "__main__":
    main()
