#!/usr/bin/env python3
"""
recommend_exits.py — read Pattern Lab, run every exit strategy, and output
a clean per-setup recommendation with decision criteria applied.

Decision criteria (editable constants below):
  MIN_TRADES     — sample size needed to trust the stats
  MIN_EXPECTANCY — minimum expectancy (in R) to be "edge"
  MIN_WIN_RATE   — minimum win rate to be "comfortable"
  MIN_EDGE_LIFT  — how much better the winner must be vs current 2R baseline

Output:
  - A per-setup recommendation table
  - A Python dict snippet you can paste into bpa_detector.py as SETUP_TARGET_R
  - A list of setups where the data doesn't clear the bar (stay on default)

Usage:
  python3 recommend_exits.py                   # live detections
  python3 recommend_exits.py --run-id ANY      # include backtests too
  python3 recommend_exits.py --json            # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compare_exits import load_detections, run_comparison  # noqa: E402
from exit_simulator import DEFAULT_STRATEGIES, StrategyStats  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Decision thresholds  (tune if you want more/less aggressive recommendations)
# ─────────────────────────────────────────────────────────────────────────────

MIN_TRADES = 100           # below this, no recommendation
MIN_EXPECTANCY = 0.10      # in R — must clear to be called "edge"
MIN_WIN_RATE = 0.50        # Will's comfort threshold
MIN_EDGE_LIFT = 0.05       # winner must beat current 2R cap by this much

BASELINE_STRATEGY_NAME = "fixed_2R"


# ─────────────────────────────────────────────────────────────────────────────
# Recommender
# ─────────────────────────────────────────────────────────────────────────────

def recommend_per_setup(stats: dict[str, dict[str, StrategyStats]]) -> dict:
    """
    Returns a dict:
        {
          "recommendations": { "H1": {...}, "L1": {...}, ... },
          "verdict_by_setup": { "H1": "ADOPT new strategy" | "STAY on 2R" | "INSUFFICIENT DATA" },
          "paste_code": "SETUP_TARGET_R = { ... }",
        }
    """
    recs: dict[str, dict] = {}
    verdicts: dict[str, str] = {}

    for setup, s_map in stats.items():
        if setup == "ALL":
            continue

        sample = next(iter(s_map.values())).total_trades
        baseline = s_map.get(BASELINE_STRATEGY_NAME)
        if sample < MIN_TRADES:
            verdicts[setup] = f"INSUFFICIENT DATA ({sample} trades; need ≥{MIN_TRADES})"
            continue

        ranked = sorted(
            s_map.values(), key=lambda s: s.expectancy_r, reverse=True,
        )
        winner = ranked[0]
        baseline_exp = baseline.expectancy_r if baseline else 0.0

        meets_expectancy = winner.expectancy_r >= MIN_EXPECTANCY
        meets_wr = winner.win_rate >= MIN_WIN_RATE
        lift = winner.expectancy_r - baseline_exp
        beats_baseline = lift >= MIN_EDGE_LIFT

        if meets_expectancy and meets_wr and beats_baseline:
            verdict = "ADOPT"
        elif meets_expectancy and meets_wr and not beats_baseline:
            verdict = f"STAY ON 2R (lift {lift:+.2f}R < {MIN_EDGE_LIFT})"
        elif not meets_expectancy:
            verdict = f"NO EDGE (exp {winner.expectancy_r:+.2f}R)"
        else:  # not meets_wr
            verdict = f"LOW WR ({winner.win_rate * 100:.1f}%)"

        recs[setup] = {
            "winner": winner.strategy.name,
            "winner_expectancy": winner.expectancy_r,
            "winner_win_rate": winner.win_rate,
            "winner_total_r": winner.total_r,
            "baseline_2r_expectancy": baseline_exp,
            "lift_over_baseline": lift,
            "sample": sample,
            "verdict": verdict,
        }
        verdicts[setup] = verdict

    # Build paste-able code for bpa_detector.py
    r_multiplier_by_setup = {}
    for setup, rec in recs.items():
        if rec["verdict"] == "ADOPT":
            # Extract R from strategy name like "fixed_1.5R" or "50@1R_BE_50@2R"
            name = rec["winner"]
            if name.startswith("fixed_") and name.endswith("R"):
                mult = float(name[len("fixed_"):-1])
                r_multiplier_by_setup[setup] = mult

    paste_lines = ["SETUP_TARGET_R = {"]
    for setup in ("H1", "H2", "L1", "L2", "FL1", "FL2", "FH1", "FH2",
                  "spike_channel", "failed_bo"):
        if setup in r_multiplier_by_setup:
            paste_lines.append(f'    "{setup}": {r_multiplier_by_setup[setup]},   '
                               f'# adopted: {recs[setup]["winner"]}')
        else:
            verdict = verdicts.get(setup, "N/A")
            paste_lines.append(f'    "{setup}": 2.0,                              '
                               f'# staying on 2R ({verdict})')
    paste_lines.append("}")
    paste_code = "\n".join(paste_lines)

    return {
        "recommendations": recs,
        "verdict_by_setup": verdicts,
        "paste_code": paste_code,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendations(output: dict) -> None:
    recs = output["recommendations"]
    verdicts = output["verdict_by_setup"]

    print("\n" + "=" * 100)
    print("  PER-SETUP RECOMMENDATION")
    print("=" * 100)
    print(f"  Thresholds: ≥{MIN_TRADES} trades · expectancy ≥ {MIN_EXPECTANCY:+.2f}R · "
          f"WR ≥ {MIN_WIN_RATE * 100:.0f}% · lift ≥ {MIN_EDGE_LIFT:+.2f}R")
    print(f"  Baseline for comparison: {BASELINE_STRATEGY_NAME}")
    print()
    print(f"  {'setup':<14} {'best strategy':<32} {'n':>5} "
          f"{'WR':>6} {'exp R':>8} {'lift':>8} verdict")
    print("  " + "-" * 98)

    ordered_setups = sorted(recs.keys())
    for setup in ordered_setups:
        r = recs[setup]
        mark = "✓" if r["verdict"] == "ADOPT" else " "
        print(
            f"  {mark} {setup:<12} {r['winner']:<32} "
            f"{r['sample']:>5} "
            f"{r['winner_win_rate'] * 100:>5.1f}% "
            f"{r['winner_expectancy']:>+7.2f}R "
            f"{r['lift_over_baseline']:>+7.2f}R  {r['verdict']}"
        )

    # Setups with insufficient data
    insufficient = [s for s, v in verdicts.items()
                    if v.startswith("INSUFFICIENT")]
    if insufficient:
        print()
        for s in insufficient:
            print(f"  ⋯ {s:<12} {verdicts[s]}")

    # Paste-able code for the detector
    if any(r["verdict"] == "ADOPT" for r in recs.values()):
        print()
        print("=" * 100)
        print("  PASTE INTO bpa_detector.py TO ADOPT WINNERS")
        print("=" * 100)
        print()
        print(output["paste_code"])
        print()
        print("Then update each detector to use `SETUP_TARGET_R[setup_type] × risk`")
        print("as the target cap instead of the flat `TARGET_MAX_R_MULT`.")
    else:
        print()
        print("  No setups cleared the threshold. Stay on the flat 2R cap and revisit")
        print("  after more data (longer backfill, or another week of live).")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", help="Filter by run_id, or 'ANY' for all")
    p.add_argument("--days", type=int, help="Last N days only")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable JSON output")
    args = p.parse_args()

    detections = load_detections(run_id=args.run_id, days=args.days)
    if not detections:
        print("No detections found.")
        return

    stats = run_comparison(detections, strategies=DEFAULT_STRATEGIES)
    output = recommend_per_setup(stats)

    if args.json:
        # Make StrategyStats serializable
        recs_out = {k: v for k, v in output["recommendations"].items()}
        print(json.dumps({
            "recommendations": recs_out,
            "verdicts": output["verdict_by_setup"],
        }, indent=2))
    else:
        print_recommendations(output)


if __name__ == "__main__":
    main()
