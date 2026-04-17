#!/usr/bin/env python3
"""
compare_exits.py — re-simulate every Pattern Lab detection under multiple
exit strategies and report which profile has the best win rate / expectancy.

The entry (signal bar, stop) stays fixed. Only the exit rule varies.
Reuses `chart_json` stored alongside each detection — specifically the
bars AFTER the signal bar — so we don't need to re-fetch market data.

Usage:
    python3 compare_exits.py                        # all live detections, default strategy bank
    python3 compare_exits.py --setup H1             # just one setup type
    python3 compare_exits.py --days 30              # last 30 days only
    python3 compare_exits.py --run-id 2026-04-17    # specific backtest run

Exit bank (edit exit_simulator.DEFAULT_STRATEGIES to change):
  fixed_0.5R, fixed_1R, fixed_1.5R, fixed_2R, fixed_3R,
  50@1R + 50@2R (no BE),  50@1R + BE + 50@2R,  50@0.5R + BE + 50@1.5R,
  33@1R + 33@2R + trail 1R runner
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exit_simulator import (  # noqa: E402
    DEFAULT_STRATEGIES,
    ExitStrategy,
    StrategyStats,
    simulate_trade,
)
from shared.pattern_lab import _connect, LIVE_ONLY  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_detections(
    setup_type: str | None = None,
    run_id: str | None = None,
    days: int | None = None,
    min_bars_after: int = 10,
) -> list[dict]:
    """Pull detections that have enough follow-on bars for simulation."""
    conn = _connect()
    try:
        where = []
        params: list = []

        if run_id is None:
            where.append("run_id IS NULL")
        elif run_id == "ANY":
            pass  # no run filter
        else:
            where.append("run_id = ?")
            params.append(run_id)

        if setup_type:
            where.append("setup_type = ?")
            params.append(setup_type)

        if days:
            where.append("date(detection_date) >= date('now', ?)")
            params.append(f"-{days} days")

        # Must have chart_json (where follow-on bars live)
        where.append("chart_json IS NOT NULL")

        sql = "SELECT * FROM detections"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY detected_at DESC"

        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        return _with_follow_bars(rows, min_bars_after)
    finally:
        conn.close()


def _with_follow_bars(rows: list[dict], min_bars: int) -> list[dict]:
    """Decode chart_json; extract bars AFTER the signal; drop too-short trades."""
    out: list[dict] = []
    for r in rows:
        raw = r.get("chart_json")
        if not raw:
            continue
        try:
            chart = json.loads(raw)
        except (TypeError, ValueError):
            continue

        bars = chart.get("bars") or []
        anno = chart.get("annotations") or {}
        sig_bar_info = anno.get("signalBar") or {}
        sig_time = sig_bar_info.get("time")
        if sig_time is None:
            continue

        # chart_json bars are ±windows around the signal bar;
        # take everything strictly AFTER sig_time
        follow = [b for b in bars if b.get("t", 0) > sig_time]
        if len(follow) < min_bars:
            continue
        r["_follow_bars"] = follow
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparison runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    detections: list[dict],
    strategies: tuple[ExitStrategy, ...] = DEFAULT_STRATEGIES,
) -> dict[str, dict[str, StrategyStats]]:
    """
    Returns nested dict: stats[setup_type][strategy_name] = StrategyStats
    Plus an "ALL" key aggregating across setup types.
    """
    stats: dict[str, dict[str, StrategyStats]] = defaultdict(
        lambda: {s.name: StrategyStats(strategy=s) for s in strategies}
    )

    for det in detections:
        setup_type = det.get("setup_type", "?")
        entry = det.get("entry_price")
        stop = det.get("stop_price")
        direction = det.get("direction")
        follow = det.get("_follow_bars") or []

        if entry is None or stop is None or not direction or not follow:
            continue

        for strat in strategies:
            sim = simulate_trade(entry, stop, direction, follow, strat)
            stats[setup_type][strat.name].record(sim)
            stats["ALL"][strat.name].record(sim)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_table(stats: dict[str, dict[str, StrategyStats]]) -> None:
    # One block per setup type
    setup_order = [k for k in stats.keys() if k != "ALL"] + ["ALL"]
    for setup in setup_order:
        s_map = stats[setup]
        n = next(iter(s_map.values())).total_trades
        if n == 0:
            continue

        print(f"\n{'═' * 100}")
        print(f"  {setup}  —  {n} trades simulated")
        print(f"{'═' * 100}")
        print(
            f"  {'strategy':<36} {'trades':>6} {'W':>4} {'L':>4} "
            f"{'S':>3} {'WR':>7} {'avg W':>8} {'avg L':>8} {'total R':>9} {'exp R':>7}"
        )
        print(f"  {'-' * 98}")

        # Sort strategies by expectancy descending
        sorted_names = sorted(
            s_map.keys(),
            key=lambda k: s_map[k].expectancy_r,
            reverse=True,
        )
        for name in sorted_names:
            st = s_map[name]
            print(
                f"  {name:<36} "
                f"{st.total_trades:>6} "
                f"{st.wins:>4} {st.losses:>4} {st.scratches:>3} "
                f"{st.win_rate * 100:>6.1f}% "
                f"{st.avg_win_r:>7.2f}R "
                f"{st.avg_loss_r:>7.2f}R "
                f"{st.total_r:>+8.1f}R "
                f"{st.expectancy_r:>+6.2f}R"
            )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", help="Filter by setup_type (H1, L2, FL1, ...)")
    p.add_argument("--run-id", help="Specific run_id, or 'ANY' for all")
    p.add_argument("--days", type=int, help="Last N days only")
    p.add_argument("--min-bars", type=int, default=10,
                   help="Minimum follow-on bars required (default 10)")
    args = p.parse_args()

    detections = load_detections(
        setup_type=args.setup,
        run_id=args.run_id,
        days=args.days,
        min_bars_after=args.min_bars,
    )

    if not detections:
        print("No detections with sufficient follow-on bars. Try broader filters.")
        return

    print(f"\nLoaded {len(detections)} detections with ≥{args.min_bars} "
          f"follow-on bars.")
    stats = run_comparison(detections)
    print_table(stats)


if __name__ == "__main__":
    main()
