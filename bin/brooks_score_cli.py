"""CLI entry point for the Brooks gap scorer.

Two modes:
  - `demo` : score a handful of synthetic charts (NOW-like, XLE-like,
             MRVL-like, gap-chop, ORCL-like) to sanity-check scorer output
  - `scan` : run the live Databento universe scan and print a leaderboard

Usage:
  python bin/brooks_score_cli.py                          # synthetic demo
  python bin/brooks_score_cli.py --mode scan              # full scan
  python bin/brooks_score_cli.py --mode scan --top 10
  python bin/brooks_score_cli.py --mode scan --tickers "AAPL,NVDA,TSLA"

Extracted from shared/brooks_score.py (Phase 3j). The library now has
no __main__ block or synthetic-data demo helpers; the scorer is pure
library code.
"""

from __future__ import annotations

import argparse
import json
import logging
import pandas as pd

# After `pip install -e .` the package resolves without any sys.path hack.
# Running this file directly (`python bin/brooks_score_cli.py`) also works
# because Python puts the script's directory on sys.path by default.

from aiedge.signals.components import LIQUIDITY_MIN_DOLLAR_VOL
from shared.brooks_score import scan_universe, score_gap, score_multiple


# =============================================================================
# SYNTHETIC BAR GENERATOR + DEMO FIXTURES
# =============================================================================

def _make_bars(specs: list[tuple], start_price: float) -> pd.DataFrame:
    """Build a synthetic DataFrame from bar specs.

    Each spec: (direction, body_pct, range_size)
      direction: "bull" or "bear" or "doji_bull" or "doji_bear"
      body_pct: body as fraction of range (0-1)
      range_size: absolute range of the bar
    Doji variants let us control whether close > open (doji_bull) or close < open
    (doji_bear) so color alternation scoring works correctly on synthetic data.
    """
    rows = []
    price = start_price
    for i, (direction, body_pct, range_size) in enumerate(specs):
        body = range_size * body_pct
        tail_total = range_size - body

        if direction == "bull":
            low = price - tail_total * 0.3
            high = low + range_size
            o = low + tail_total * 0.2
            c = high - tail_total * 0.1
            price = c
        elif direction == "bear":
            high = price + tail_total * 0.3
            low = high - range_size
            o = high - tail_total * 0.2
            c = low + tail_total * 0.1
            price = c
        elif direction == "doji_bull":
            mid = price
            high = mid + range_size / 2
            low = mid - range_size / 2
            o = mid - body / 2
            c = mid + body / 2
            price = mid
        else:  # doji_bear
            mid = price
            high = mid + range_size / 2
            low = mid - range_size / 2
            o = mid + body / 2
            c = mid - body / 2
            price = mid

        rows.append({
            "datetime": pd.Timestamp("2026-04-13 09:30") + pd.Timedelta(minutes=5 * i),
            "open": round(o, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(c, 2),
            "volume": 50000 + i * 1000,
        })

    return pd.DataFrame(rows)


def _demo_now_like():
    """NOW-like: strong bull gap, clean spike, shallow pullback, early follow-through.

    Should score urgency ~7+, uncertainty ~2-3, signal = BUY_PULLBACK.
    """
    prior_close = 870.00
    gap_open = 885.00  # ~1.7% gap

    specs = [
        ("bull", 0.80, 4.00), ("bull", 0.75, 3.50),
        ("bull", 0.70, 3.00), ("bull", 0.65, 2.80),
        ("bear", 0.40, 1.80), ("bear", 0.35, 1.20),
        ("bull", 0.65, 2.50), ("bull", 0.70, 2.80),
    ]
    return _make_bars(specs, gap_open), prior_close


def _demo_xle_like():
    """XLE-like: weak gap, rapid color alternation, dojis, two-sided.

    Should score urgency ~2, uncertainty ~5+, signal = FOG.
    """
    prior_close = 60.00
    gap_open = 60.80

    specs = [
        ("bull", 0.40, 0.50), ("bear", 0.45, 0.55),
        ("bull", 0.30, 0.40), ("bear", 0.50, 0.70),
        ("doji_bull", 0.12, 0.60), ("bear", 0.55, 0.65),
        ("bull", 0.35, 0.50), ("bear", 0.45, 0.60),
        ("doji_bear", 0.10, 0.45), ("bull", 0.30, 0.40),
        ("bear", 0.40, 0.55), ("doji_bull", 0.15, 0.50),
    ]
    return _make_bars(specs, gap_open), prior_close


def _demo_mrvl_like():
    """MRVL-like: decent initial 2-bar move, then stalls and becomes two-sided.

    Should score urgency ~3-4, uncertainty ~5+, signal = FOG.
    """
    prior_close = 65.00
    gap_open = 67.00

    specs = [
        ("bull", 0.65, 1.20), ("bull", 0.55, 0.90),
        ("bear", 0.70, 1.30), ("bear", 0.65, 1.10),
        ("bull", 0.40, 0.70), ("bear", 0.50, 0.80),
        ("bull", 0.35, 0.60), ("doji_bear", 0.10, 0.55),
        ("bear", 0.45, 0.75), ("bull", 0.40, 0.65),
        ("doji_bull", 0.12, 0.50), ("bear", 0.50, 0.70),
    ]
    return _make_bars(specs, gap_open), prior_close


def _demo_gap_chop():
    """Gap-and-chop: big gap from prior close, then sideways in a range.

    Open is in the MIDDLE of the day's range — should classify as trading_range
    despite the large gap from prior close.
    """
    prior_close = 92.00
    gap_open = 100.00

    specs = [
        ("bear", 0.45, 0.60), ("bear", 0.40, 0.50),
        ("bull", 0.35, 0.55), ("bull", 0.40, 0.50),
        ("bear", 0.45, 0.55), ("doji_bull", 0.12, 0.40),
        ("bull", 0.35, 0.45), ("bear", 0.40, 0.50),
        ("bull", 0.30, 0.40), ("bear", 0.35, 0.45),
        ("doji_bear", 0.10, 0.35), ("bull", 0.40, 0.50),
        ("bear", 0.45, 0.55), ("bull", 0.30, 0.40),
        ("doji_bull", 0.15, 0.35), ("bear", 0.35, 0.45),
        ("bull", 0.40, 0.50), ("bear", 0.30, 0.40),
        ("doji_bear", 0.10, 0.35), ("bull", 0.35, 0.45),
    ]
    return _make_bars(specs, gap_open), prior_close


def _demo_orcl_like():
    """ORCL-like: strong trend from open. Open sets the low, 80% trend bars,
    small pullbacks, then some late-session chop (normal on trend days).

    Should classify as trend_from_open or spike_and_channel.
    """
    prior_close = 150.00
    gap_open = 167.72

    specs = [
        ("bull", 0.75, 1.80), ("bull", 0.70, 1.60), ("bull", 0.75, 1.70),
        ("bull", 0.70, 1.50), ("bull", 0.65, 1.40), ("bull", 0.70, 1.60),
        ("bull", 0.65, 1.30), ("bull", 0.70, 1.50),
        ("bear", 0.35, 0.80), ("bear", 0.30, 0.60),
        ("bull", 0.65, 1.30), ("bull", 0.60, 1.20), ("bull", 0.65, 1.10),
        ("bull", 0.55, 1.00), ("bull", 0.60, 1.10), ("bull", 0.55, 0.90),
        ("bear", 0.40, 0.90), ("bull", 0.60, 1.00),
        ("bull", 0.55, 1.00), ("bull", 0.60, 1.10), ("bull", 0.50, 0.90),
        ("bear", 0.50, 1.00), ("bull", 0.40, 0.80),
        ("bear", 0.45, 0.90), ("bull", 0.35, 0.70),
        ("bear", 0.40, 0.80), ("doji_bull", 0.15, 0.60),
    ]
    return _make_bars(specs, gap_open), prior_close


# =============================================================================
# CLI ENTRY POINTS
# =============================================================================

def _run_demo():
    """Score five synthetic charts and print per-ticker detail + ranked leaderboard."""
    print("=" * 80)
    print("BROOKS GAP SCORER — DEMO (synthetic data)")
    print("=" * 80)

    demos = [
        ("NOW-like (strong bull gap, clean spike)", _demo_now_like, "up"),
        ("XLE-like (weak gap, overlapping, foggy)", _demo_xle_like, "up"),
        ("MRVL-like (decent gap, stalls, two-sided)", _demo_mrvl_like, "up"),
        ("GAP-CHOP (big gap, then sideways)", _demo_gap_chop, "up"),
        ("ORCL-like (trend from open, late chop)", _demo_orcl_like, "up"),
    ]

    all_gaps = []
    all_dfs = {}

    for label, demo_fn, direction in demos:
        df, prior_close = demo_fn()
        ticker = label.split("(")[0].strip().replace("-like", "").strip()

        print(f"\n{'─' * 80}")
        print(f"  {label}")
        print(f"  Prior close: ${prior_close:.2f}  |  Gap open: ${df.iloc[0]['open']:.2f}  |  Bars: {len(df)}")
        print(f"{'─' * 80}")

        result = score_gap(df, prior_close, direction, ticker=ticker)

        print(f"  SIGNAL:      {result['signal']}")
        print(f"  Urgency:     {result['urgency']}/10")
        print(f"  Uncertainty: {result['uncertainty']}/10")
        print(f"  Day type:    {result['day_type']} (conf={result['day_type_confidence']:.0%})")
        print(f"  Warning:     {result['day_type_warning']}")
        print(f"  OR %:        {result['opening_range_pct']:.0%}")
        print(f"  Phase:       {result['phase']}")
        print(f"  Always-in:   {result['always_in']}")
        print(f"  Spike bars:  {result['spike_bars']}")
        print(f"  Pullback:    {result['pullback_depth_pct']:.1%} of spike")
        print(f"  Gap held:    {result['gap_held']}")
        print(f"  Risk:        ${result['risk']:.2f}")
        print(f"  Reward:      ${result['reward']:.2f}")
        print(f"  R:R:         {result['rr_ratio']:.1f}")
        print(f"  Summary:     {result['summary']}")
        print(f"  Details:     {json.dumps(result['details'], indent=2)}")

        all_gaps.append({"ticker": ticker, "prior_close": prior_close, "gap_direction": direction})
        all_dfs[ticker] = df

    print(f"\n{'=' * 80}")
    print("RANKED (best setup first):")
    print(f"{'=' * 80}")
    ranked = score_multiple(all_gaps, all_dfs)
    for i, r in enumerate(ranked, 1):
        dt = r.get("day_type", "?")
        print(f"  {i}. {r['ticker']:8s}  {r['signal']:16s}  "
              f"U={r['urgency']:4.1f}  Unc={r['uncertainty']:4.1f}  "
              f"R:R={r['rr_ratio']:4.1f}  {dt:<18}  {r['phase']}")


def _run_scan(args):
    """Run the live universe scan with Databento data."""
    from datetime import datetime
    import pytz

    et = pytz.timezone("America/New_York")
    now_et = datetime.now(et)

    print(f"\n{'=' * 80}")
    print(f"  STRONGEST STOCKS — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"  Filters: urgency >= {args.min_urgency}  |  uncertainty <= {args.max_uncertainty}  |  min $vol/bar >= ${args.min_dollar_vol:,.0f}")
    print(f"{'=' * 80}\n")

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        print(f"  Custom universe: {len(tickers)} symbols\n")

    results = scan_universe(
        tickers=tickers,
        min_urgency=args.min_urgency,
        max_uncertainty=args.max_uncertainty,
        min_dollar_vol=args.min_dollar_vol,
        verbose=args.verbose,
    )

    if not results:
        print("  No stocks passed the filters. Try lowering --min-urgency or raising --max-uncertainty.\n")
        return

    top_n = results[:args.top]

    header = (f"  {'#':>3}  {'Ticker':<7}  {'Urgency':>7}  {'Uncert':>6}  "
              f"{'Signal':<16}  {'DayType':<18}  {'Gap%':>6}  {'R:R':>5}  Warning")
    print(header)
    print(f"  {'─' * len(header)}")

    for i, r in enumerate(top_n, 1):
        gap_str = f"{r.get('gap_pct', 0):+.1f}%"
        dt = r.get("day_type", "?")
        warning_short = r.get("day_type_warning", "")[:55]
        if len(r.get("day_type_warning", "")) > 55:
            warning_short += "…"
        print(
            f"  {i:>3}  {r['ticker']:<7}  {r['urgency']:>7.1f}  {r['uncertainty']:>6.1f}  "
            f"{r['signal']:<16}  {dt:<18}  {gap_str:>6}  {r['rr_ratio']:>5.1f}  "
            f"{warning_short}"
        )

    print(f"\n  {len(results)} total stocks passed filters | showing top {len(top_n)}")
    print(f"  Data: EQUS.MINI ohlcv-1d + ohlcv-1m via Databento\n")


def _main() -> None:
    """Console entry point — exposed as `aiedge-brooks-score` after `pip install -e .`."""
    parser = argparse.ArgumentParser(
        description="Brooks Price Action Gap Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aiedge-brooks-score                          # Run synthetic demo
  aiedge-brooks-score --mode scan              # Scan full universe
  aiedge-brooks-score --mode scan --top 10
  aiedge-brooks-score --mode scan --min-urgency 5 --max-uncertainty 4
  aiedge-brooks-score --mode scan --tickers "AAPL,NVDA,NOW,MRVL,TSLA"
        """,
    )
    parser.add_argument(
        "--mode", choices=["demo", "scan"], default="demo",
        help="'demo' runs synthetic examples, 'scan' runs live Databento scan (default: demo)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Show top N results in scan mode (default: 20)",
    )
    parser.add_argument(
        "--min-urgency", type=float, default=3.0,
        help="Minimum urgency score to include (default: 3.0)",
    )
    parser.add_argument(
        "--max-uncertainty", type=float, default=7.0,
        help="Maximum uncertainty score to include (default: 7.0)",
    )
    parser.add_argument(
        "--min-dollar-vol", type=float, default=LIQUIDITY_MIN_DOLLAR_VOL,
        help=f"Min avg dollar volume per 5-min bar (default: ${LIQUIDITY_MIN_DOLLAR_VOL:,.0f})",
    )
    parser.add_argument(
        "--tickers", type=str, default=None,
        help="Comma-separated list of tickers to scan (default: full universe)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log per-symbol scoring progress",
    )

    args = parser.parse_args()

    if args.mode == "demo":
        _run_demo()
    elif args.mode == "scan":
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
        _run_scan(args)


if __name__ == "__main__":
    _main()
