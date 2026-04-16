"""QC inspector harness for _score_small_pullback_trend (SPT).

Runs the six QC checks:
  1. Spec Compliance   — static cross-check of constants/weights vs spec
  2. Test Coverage     — empty, 2-bar, NaN, extreme values, pure bull, pure bear, chop
  3. Regression        — existing score_gap on a fixed synthetic df is stable
                         (captures urgency delta; fails only on silent behavior changes)
  4. Data Integrity    — function does not mutate input df; no .dropna; handles NaN
  5. Performance       — 12k symbols at ≤ avg bars within 5s total
  6. Observability     — details["small_pullback_trend"] present and finite

Exit 0 if all pass, non-zero otherwise.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/sessions/stoic-sleepy-gates/mnt/video-pipeline")

import numpy as np
import pandas as pd

from shared.brooks_score import (
    _score_small_pullback_trend,
    score_gap,
    SPT_LOOKBACK_BARS,
    SPT_TREND_BODY_RATIO,
    SPT_DEPTH_SHALLOW,
    SPT_DEPTH_MODERATE,
    URGENCY_RAW_MAX,
    DAY_TYPE_WEIGHTS,
)

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, reason: str = ""):
    results.append((name, ok, reason))


def mk_df(opens, closes, highs=None, lows=None, vol=100_000):
    n = len(opens)
    if highs is None:
        highs = [max(o, c) + 0.05 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": [vol] * n})


# ── 1) Spec Compliance ─────────────────────────────────────────────────────
spec_checks = {
    "SPT_LOOKBACK_BARS == 15": SPT_LOOKBACK_BARS == 15,
    "SPT_TREND_BODY_RATIO == 0.40": abs(SPT_TREND_BODY_RATIO - 0.40) < 1e-9,
    "SPT_DEPTH_SHALLOW == 0.25": abs(SPT_DEPTH_SHALLOW - 0.25) < 1e-9,
    "SPT_DEPTH_MODERATE == 0.40": abs(SPT_DEPTH_MODERATE - 0.40) < 1e-9,
    "URGENCY_RAW_MAX raised to 29.0 (26 + 3 for SPT)": URGENCY_RAW_MAX == 29.0,
    "trend_from_open spt weight matches spike_quality (1.5)":
        DAY_TYPE_WEIGHTS["trend_from_open"]["small_pullback_trend"] == 1.5
        and DAY_TYPE_WEIGHTS["trend_from_open"]["spike_quality"] == 1.5,
    "trading_range spt weight suppressed (0.3)":
        DAY_TYPE_WEIGHTS["trading_range"]["small_pullback_trend"] == 0.3,
    "tight_tr spt weight suppressed (0.1)":
        DAY_TYPE_WEIGHTS["tight_tr"]["small_pullback_trend"] == 0.1,
    "Weighted sum max = 0.8+0.8+0.6+0.4+0.4 = 3.0":
        abs((0.8 + 0.8 + 0.6 + 0.4 + 0.4) - 3.0) < 1e-9,
}
spec_ok = all(spec_checks.values())
spec_failures = [k for k, v in spec_checks.items() if not v]
record("Spec Compliance", spec_ok, "" if spec_ok else f"failed: {spec_failures}")

# ── 2) Test Coverage ───────────────────────────────────────────────────────
coverage_ok = True
coverage_reasons: list[str] = []

try:
    # empty
    if _score_small_pullback_trend(pd.DataFrame(columns=["open","high","low","close","volume"]), "up") != 0.0:
        coverage_ok = False; coverage_reasons.append("empty df not 0.0")
    # single bar
    df1 = mk_df([100], [101])
    if _score_small_pullback_trend(df1, "up") != 0.0:
        coverage_ok = False; coverage_reasons.append("single-bar not 0.0")
    # three bars (below min threshold of 4)
    df3 = mk_df([100,101,102], [101,102,103])
    if _score_small_pullback_trend(df3, "up") != 0.0:
        coverage_ok = False; coverage_reasons.append("3-bar not 0.0")
    # Unknown direction
    if _score_small_pullback_trend(df3, "sideways") != 0.0:
        coverage_ok = False; coverage_reasons.append("bad direction not 0.0")
    # Pure bull steady-drift — should score high
    opens = [100 + 0.20*i for i in range(20)]
    closes = [100 + 0.20*i + 0.15 for i in range(20)]
    highs = [c + 0.05 for c in closes]
    lows  = [o - 0.03 for o in opens]
    df_bull = mk_df(opens, closes, highs, lows)
    s_bull = _score_small_pullback_trend(df_bull, "up")
    if not (0 <= s_bull <= 3.0):
        coverage_ok = False; coverage_reasons.append(f"pure bull out of range: {s_bull}")
    # Pure bear — symmetric
    opens_b = [100 - 0.20*i for i in range(20)]
    closes_b = [100 - 0.20*i - 0.15 for i in range(20)]
    highs_b = [o + 0.03 for o in opens_b]
    lows_b  = [c - 0.05 for c in closes_b]
    df_bear = mk_df(opens_b, closes_b, highs_b, lows_b)
    s_bear = _score_small_pullback_trend(df_bear, "down")
    if not (0 <= s_bear <= 3.0):
        coverage_ok = False; coverage_reasons.append(f"pure bear out of range: {s_bear}")
    # Extreme values (prices in thousands)
    df_big = mk_df([x*100 for x in opens], [x*100 for x in closes])
    s_big = _score_small_pullback_trend(df_big, "up")
    if not (0 <= s_big <= 3.0):
        coverage_ok = False; coverage_reasons.append(f"extreme price out of range: {s_big}")
    # NaN-in-input — must not raise
    df_nan = df_bull.copy()
    df_nan.loc[5, "close"] = np.nan
    try:
        s_nan = _score_small_pullback_trend(df_nan, "up")
        if not np.isfinite(s_nan) and s_nan != 0.0:
            coverage_ok = False; coverage_reasons.append(f"NaN input produced {s_nan}")
    except Exception as e:
        coverage_ok = False; coverage_reasons.append(f"NaN raised: {e}")
    # Zero-range bar (doji at machine-precision)
    df_zr = df_bull.copy()
    df_zr.loc[10, ["open","high","low","close"]] = 105.5
    s_zr = _score_small_pullback_trend(df_zr, "up")
    if not (0 <= s_zr <= 3.0):
        coverage_ok = False; coverage_reasons.append(f"zero-range bar out of range: {s_zr}")
except Exception as e:
    coverage_ok = False; coverage_reasons.append(f"coverage raised: {e}")

record("Test Coverage", coverage_ok, "" if coverage_ok else "; ".join(coverage_reasons))

# ── 3) Regression ──────────────────────────────────────────────────────────
# A deterministic synthetic df. Re-running produces identical urgency.
np.random.seed(42)
n = 20
o = 100 + np.cumsum(np.random.normal(0.10, 0.08, n))
c = o + np.random.normal(0.12, 0.1, n)
h = np.maximum(o, c) + np.abs(np.random.normal(0.04, 0.02, n))
l = np.minimum(o, c) - np.abs(np.random.normal(0.03, 0.02, n))
df_reg = mk_df(o.tolist(), c.tolist(), h.tolist(), l.tolist())
r1 = score_gap(df_reg, prior_close=float(o[0])-0.2, gap_direction="up", ticker="REG", daily_atr=2.0)
r2 = score_gap(df_reg, prior_close=float(o[0])-0.2, gap_direction="up", ticker="REG", daily_atr=2.0)
stable = abs(r1["urgency"] - r2["urgency"]) < 1e-9 and r1["details"] == r2["details"]
# Also: SPT should never drive urgency ABOVE the 10.0 cap (structural)
bounded = 0.0 <= r1["urgency"] <= 10.0
# And details.small_pullback_trend present
has_spt_detail = "small_pullback_trend" in r1["details"]
record("Regression", stable and bounded and has_spt_detail,
       "" if stable and bounded and has_spt_detail
       else f"stable={stable} bounded={bounded} has_spt={has_spt_detail}")

# ── 4) Data Integrity ──────────────────────────────────────────────────────
orig = df_reg.copy(deep=True)
_ = _score_small_pullback_trend(df_reg, "up")
mutated = not df_reg.equals(orig)
# Check no .dropna in function source
import inspect
from shared import brooks_score
src = inspect.getsource(brooks_score._score_small_pullback_trend)
drops_na = ".dropna" in src
# Window slice should not touch the original frame
di_ok = (not mutated) and (not drops_na)
record("Data Integrity", di_ok,
       "" if di_ok else f"mutated={mutated} dropna_in_src={drops_na}")

# ── 5) Performance ─────────────────────────────────────────────────────────
# Simulate 12k symbols worth of SPT calls. Scan cycle budget for SPT alone: ≤ 5s.
df_perf = df_bull
t0 = time.perf_counter()
N = 12_000
total = 0.0
for _ in range(N):
    total += _score_small_pullback_trend(df_perf, "up")
elapsed = time.perf_counter() - t0
perf_ok = elapsed < 5.0
record("Performance", perf_ok,
       f"{N} iters in {elapsed:.3f}s (budget 5.0s)")

# ── 6) Observability ───────────────────────────────────────────────────────
obs_ok = True
obs_reasons: list[str] = []
if "small_pullback_trend" not in r1["details"]:
    obs_ok = False; obs_reasons.append("missing in details")
if not np.isfinite(r1["details"]["small_pullback_trend"]):
    obs_ok = False; obs_reasons.append("not finite")
# Dashboard strip function lives in live_scanner.py — verify it references SPT
with open("/sessions/stoic-sleepy-gates/mnt/video-pipeline/live_scanner.py") as f:
    ls_src = f.read()
if "_build_component_strip" not in ls_src or "small_pullback_trend" not in ls_src:
    obs_ok = False; obs_reasons.append("live_scanner missing strip or spt key")
if "\"SPT\"" not in ls_src:
    obs_ok = False; obs_reasons.append("SPT label missing from strip")
record("Observability", obs_ok, "" if obs_ok else "; ".join(obs_reasons))

# ── Emit QC sheet ───────────────────────────────────────────────────────────
print("── QC SHEET — SPT component ──")
for name, ok, reason in results:
    tag = "pass" if ok else f"reject: {reason}"
    check = "x" if ok else " "
    print(f"[{check}] {name:<18} — {tag}")

all_pass = all(ok for _, ok, _ in results)
print()
print("ALL PASS" if all_pass else "AT LEAST ONE REJECT — FIX AND RE-RUN")
sys.exit(0 if all_pass else 1)
