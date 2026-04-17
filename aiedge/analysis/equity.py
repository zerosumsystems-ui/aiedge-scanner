"""Trade-list → equity curve + risk metrics.

Consumes a list of trade dicts (or a DataFrame) with `entry`, `exit`,
`direction`, and optional `r_multiple` / `size`. Produces:

  - `equity_curve(trades)` — cumulative PnL series indexed by trade #
  - `sharpe(pnls, periods_per_year=252)` — annualized Sharpe (0% risk-free)
  - `sortino(pnls, periods_per_year=252)` — annualized Sortino
  - `max_drawdown(equity)` — max peak-to-trough drawdown as a fraction

All inputs/outputs are pure pandas/numpy. No I/O.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def equity_curve(pnls: Iterable[float]) -> pd.Series:
    """Cumulative-sum equity curve from per-trade PnL.

    The returned series is indexed 0..N-1 (trade number); add a starting
    capital yourself if needed. Empty input → empty series.
    """
    arr = np.asarray(list(pnls), dtype=float)
    return pd.Series(np.cumsum(arr), name="equity")


def sharpe(pnls: Iterable[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio with a 0% risk-free rate.

    Returns 0.0 if stddev is zero or the series is too short.
    """
    arr = np.asarray(list(pnls), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(arr) / sd * np.sqrt(periods_per_year))


def sortino(pnls: Iterable[float], periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio — divides by downside deviation only.

    Returns 0.0 if no negative returns or series is too short.
    """
    arr = np.asarray(list(pnls), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    downside = arr[arr < 0]
    if downside.size < 2:
        return 0.0
    sd = float(np.std(downside, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(arr) / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series | Iterable[float]) -> float:
    """Maximum peak-to-trough drawdown as a fraction of the running peak.

    Returns a non-negative float; 0.0 means no drawdown (monotonic up).
    Handles both absolute-equity series and cumulative-PnL series —
    same formula as long as values are strictly non-negative or you've
    already added a base capital.
    """
    arr = np.asarray(list(equity), dtype=float)
    if arr.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    # Guard zero/negative peaks so division is safe.
    running_peak = np.where(running_peak <= 0, 1e-12, running_peak)
    dd = (running_peak - arr) / running_peak
    return float(np.max(dd))


def summary_stats(pnls: Iterable[float], periods_per_year: int = 252) -> dict:
    """One-call convenience: compute all metrics at once.

    Returns `{total_pnl, win_rate, avg_win, avg_loss, expectancy,
    sharpe, sortino, max_drawdown, n_trades}`. Safe on empty input.
    """
    arr = np.asarray(list(pnls), dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "total_pnl": 0.0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "n_trades": 0,
        }

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = wins.size / arr.size
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "total_pnl": float(np.sum(arr)),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "expectancy": round(expectancy, 4),
        "sharpe": round(sharpe(arr, periods_per_year), 3),
        "sortino": round(sortino(arr, periods_per_year), 3),
        "max_drawdown": round(max_drawdown(equity_curve(arr)), 4),
        "n_trades": int(arr.size),
    }
