"""Failure-mode taxonomy for losing trades.

Every loss costs money; only some losses teach. This module sorts a
losing trade into one of a small vocabulary of failure categories so
post-mortems can aggregate on root cause rather than ticker:

  - `stop_flush`     — price stopped out within the first 1-2 bars
                       of entry; entry was too late or too tight.
  - `slow_bleed`     — gradual adverse drift; entry side was wrong
                       from the start.
  - `reversal`       — in-profit then reversed and hit stop; didn't
                       manage the winner.
  - `news_shock`     — large single-bar move past the stop (|bar| >>
                       daily ATR); unavoidable but flag it.
  - `chop`           — oscillated around entry, stopped on a random
                       tick; setup was a trading-range fade.
  - `unknown`        — fallback.

Each classifier is a dict-in / str-out function that consumes a trade
record with at minimum: `entry`, `stop`, `direction`, and a DataFrame
of post-entry bars. No I/O, no writes to storage.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


FailureReason = Literal[
    "stop_flush", "slow_bleed", "reversal", "news_shock", "chop", "unknown",
]


def classify_failure(
    trade: dict,
    post_entry_bars: pd.DataFrame,
    daily_atr: float | None = None,
) -> FailureReason:
    """Return a failure-mode label for one losing trade.

    `trade` must have: entry (float), stop (float), direction ("long"
    or "short"). `post_entry_bars` is the OHLC slice from entry bar
    through the stop-out bar (inclusive). `daily_atr` enables the
    `news_shock` branch — pass the ticker's 20-day ADR.

    Logic:
      1. If the stop was hit within the first 2 bars after entry →
         stop_flush.
      2. If any single bar's |close-open| exceeds 70% of daily_atr AND
         hit the stop → news_shock.
      3. If trade went into profit (max-favorable excursion ≥ 50% of
         initial risk) before reversing to hit stop → reversal.
      4. If MAE grew monotonically but slowly (no big shock bars and
         never profitable) → slow_bleed.
      5. Otherwise → chop.
    """
    entry = trade.get("entry")
    stop = trade.get("stop")
    direction = trade.get("direction")
    if entry is None or stop is None or direction not in ("long", "short"):
        return "unknown"
    if post_entry_bars is None or len(post_entry_bars) == 0:
        return "unknown"

    initial_risk = abs(entry - stop)
    if initial_risk <= 0:
        return "unknown"

    # ── 1. Stop flush — hit within first 2 bars ──
    first_two = post_entry_bars.iloc[:2]
    if direction == "long":
        if (first_two["low"] <= stop).any():
            return "stop_flush"
    else:
        if (first_two["high"] >= stop).any():
            return "stop_flush"

    # ── 2. News shock — any bar with huge body hit the stop ──
    if daily_atr and daily_atr > 0:
        bodies = (post_entry_bars["close"] - post_entry_bars["open"]).abs()
        if (bodies > 0.70 * daily_atr).any():
            # Check stop was hit
            if direction == "long" and (post_entry_bars["low"] <= stop).any():
                return "news_shock"
            if direction == "short" and (post_entry_bars["high"] >= stop).any():
                return "news_shock"

    # ── 3. Reversal — got into profit, then stopped out ──
    if direction == "long":
        mfe = float(post_entry_bars["high"].max()) - entry
    else:
        mfe = entry - float(post_entry_bars["low"].min())
    if mfe >= 0.50 * initial_risk:
        return "reversal"

    # ── 4. Slow bleed — monotonic adverse drift with small bars ──
    closes = post_entry_bars["close"].values
    if closes.size >= 3:
        if direction == "long":
            drift = closes[-1] - closes[0]
        else:
            drift = closes[0] - closes[-1]
        adverse = drift < 0
        # "monotonic-ish" = running average declines; avoid requiring strict monotonicity
        avg_window = min(3, closes.size)
        rolling = np.convolve(closes, np.ones(avg_window) / avg_window, mode="valid")
        if direction == "long":
            mono = np.all(np.diff(rolling) <= 0.02 * initial_risk)
        else:
            mono = np.all(np.diff(rolling) >= -0.02 * initial_risk)
        if adverse and mono:
            return "slow_bleed"

    return "chop"


def failure_breakdown(
    losses: list[dict],
    daily_atrs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate `classify_failure` across many losses.

    Each item of `losses` must have: `entry`, `stop`, `direction`,
    `ticker`, and a `post_entry_bars` DataFrame. `daily_atrs` is an
    optional ticker → ATR mapping.

    Returns a DataFrame with columns: failure_reason, n, pct.
    """
    atrs = daily_atrs or {}
    reasons: list[str] = []
    for loss in losses:
        reasons.append(classify_failure(
            loss,
            loss.get("post_entry_bars"),
            daily_atr=atrs.get(loss.get("ticker")),
        ))
    counts = pd.Series(reasons).value_counts()
    total = int(counts.sum()) if counts.size else 0
    return pd.DataFrame({
        "failure_reason": counts.index,
        "n": counts.values,
        "pct": [round(c / total, 4) if total else 0.0 for c in counts.values],
    })
