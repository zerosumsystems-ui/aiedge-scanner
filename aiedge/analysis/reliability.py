"""Reliability (calibration) diagrams for probabilistic predictions.

Given a sequence of predicted win probabilities and the realized
binary outcomes, bucket the predictions and report the average
predicted probability vs the empirical hit rate per bucket.

A perfectly calibrated predictor lies on the diagonal. Systematic
over-confidence shows up as bars below diagonal; systematic
under-confidence as bars above.

Functions return plain dicts / DataFrames so callers can plot with
matplotlib, pipe to the aiedge.trade dashboard, or feed into a
Brier-score / log-loss monitor. No plotting here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def reliability_table(
    predicted: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bucketed calibration table.

    Inputs:
      predicted: P(win) forecasts in [0, 1]
      outcomes:  realized outcomes (1 = win, 0 = loss)
      n_bins:    number of equal-width [0, 1] bins (default 10)

    Returns a DataFrame with columns:
      bin_lo, bin_hi, n, mean_predicted, empirical_hit_rate, bin_midpoint
    — one row per bin that contained at least one observation.
    """
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size != y.size:
        raise ValueError(f"predicted/outcomes length mismatch: {p.size} vs {y.size}")

    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size == 0:
        return pd.DataFrame(columns=[
            "bin_lo", "bin_hi", "n", "mean_predicted",
            "empirical_hit_rate", "bin_midpoint",
        ])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_idx == b
        n = int(in_bin.sum())
        if n == 0:
            continue
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_predicted": round(float(np.mean(p[in_bin])), 4),
            "empirical_hit_rate": round(float(np.mean(y[in_bin])), 4),
            "bin_midpoint": round(float((edges[b] + edges[b + 1]) / 2), 4),
        })
    return pd.DataFrame(rows)


def brier_score(predicted: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error between predicted probability and realized outcome.

    Lower = better-calibrated. 0 = perfect. 0.25 = "random coin flip".
    """
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size != y.size:
        raise ValueError(f"predicted/outcomes length mismatch: {p.size} vs {y.size}")
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    predicted: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """ECE: weighted average of |predicted - empirical| across bins,
    weighted by bin count. Lower = better. Bounded [0, 1].
    """
    table = reliability_table(predicted, outcomes, n_bins)
    if table.empty:
        return 0.0
    total = table["n"].sum()
    if total == 0:
        return 0.0
    gaps = np.abs(table["mean_predicted"] - table["empirical_hit_rate"])
    weights = table["n"] / total
    return float(np.sum(gaps * weights))
