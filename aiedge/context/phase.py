"""Layer-1 cycle-phase classifier.

Classifies the recent chart into one of five Brooks cycle phases:
    bull_spike, bear_spike, bull_channel, bear_channel, trading_range

Output is a probability distribution (softmax over five raw scores)
plus the argmax label with its confidence.

This is context, not signal — the phase is a piece of state that
downstream signal logic can condition on, but the phase itself is
not tradeable.

Extracted from shared/brooks_score.py (Phase 3c).
"""

import numpy as np
import pandas as pd

from aiedge.features.candles import (
    DOJI_BODY_RATIO,
    _body,
    _close_position,
    _is_bear,
    _is_bull,
    _safe_range,
)


# ── Tunables ─────────────────────────────────────────────────────────
CYCLE_PHASE_CLASSIFIER_ENABLED = True    # kill-switch
CYCLE_PHASE_LOOKBACK_BARS = 15
CYCLE_PHASE_SOFTMAX_TEMP = 0.6           # tuned by eye; refine with labeled data later
CYCLE_PHASES = (
    "bull_spike",
    "bear_spike",
    "bull_channel",
    "bear_channel",
    "trading_range",
)


# ── Raw per-phase scores (each returns [0, 1]) ───────────────────────

def _cycle_bull_spike_raw(df: pd.DataFrame) -> float:
    """Raw [0,1] score for bull spike: recent bars are big bull trend bars with strong closes."""
    if len(df) < 3:
        return 0.0
    recent = df.tail(min(5, len(df)))
    bull_count = 0
    body_strength = 0.0
    close_pos_sum = 0.0
    for _, bar in recent.iterrows():
        if _is_bull(bar):
            bull_count += 1
            rng = _safe_range(bar)
            if rng > 0:
                body_strength += _body(bar) / rng
                close_pos_sum += _close_position(bar)
    n = len(recent)
    if n == 0:
        return 0.0
    bull_density = bull_count / n
    avg_body = body_strength / max(bull_count, 1)
    avg_close_pos = close_pos_sum / max(bull_count, 1)
    # all three must be high: many bull bars, big bodies, closes near highs
    raw = bull_density * avg_body * avg_close_pos
    # bonus if last bar broke recent swing high
    if len(df) >= CYCLE_PHASE_LOOKBACK_BARS:
        recent_high = df.iloc[-CYCLE_PHASE_LOOKBACK_BARS:-1]["high"].max()
        if df.iloc[-1]["close"] > recent_high:
            raw = min(1.0, raw * 1.3)
    return float(max(0.0, min(1.0, raw)))


def _cycle_bear_spike_raw(df: pd.DataFrame) -> float:
    """Mirror of bull spike for down direction."""
    if len(df) < 3:
        return 0.0
    recent = df.tail(min(5, len(df)))
    bear_count = 0
    body_strength = 0.0
    close_pos_sum = 0.0
    for _, bar in recent.iterrows():
        if _is_bear(bar):
            bear_count += 1
            rng = _safe_range(bar)
            if rng > 0:
                body_strength += _body(bar) / rng
                close_pos_sum += (1.0 - _close_position(bar))  # bear close is near LOW
    n = len(recent)
    if n == 0:
        return 0.0
    bear_density = bear_count / n
    avg_body = body_strength / max(bear_count, 1)
    avg_close_pos = close_pos_sum / max(bear_count, 1)
    raw = bear_density * avg_body * avg_close_pos
    if len(df) >= CYCLE_PHASE_LOOKBACK_BARS:
        recent_low = df.iloc[-CYCLE_PHASE_LOOKBACK_BARS:-1]["low"].min()
        if df.iloc[-1]["close"] < recent_low:
            raw = min(1.0, raw * 1.3)
    return float(max(0.0, min(1.0, raw)))


def _cycle_bull_channel_raw(df: pd.DataFrame) -> float:
    """Raw [0,1] for bull channel: sustained bull bias with smaller bodies, shallow pullbacks."""
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    # net drift up
    net_up = (closes[-1] - closes[0]) / max(highs.max() - lows.min(), 1e-9)
    drift = max(0.0, min(1.0, net_up * 2.0))   # 50% of range = full credit

    # higher-closes streak fraction
    higher = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    higher_frac = higher / (len(closes) - 1)

    # reject if bodies are TOO big (that's a spike, not a channel)
    avg_body_ratio = np.mean([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    channel_body_fit = 1.0 - abs(avg_body_ratio - 0.45) / 0.45   # peak at 0.45
    channel_body_fit = max(0.0, min(1.0, channel_body_fit))

    # shallow pullback: max drawdown from running peak stays small
    running_peak = np.maximum.accumulate(highs)
    drawdowns = (running_peak - lows) / np.maximum(running_peak, 1e-9)
    max_dd = drawdowns.max()
    shallow = max(0.0, 1.0 - max_dd * 5.0)   # 20% DD → zero credit

    raw = drift * 0.35 + higher_frac * 0.25 + channel_body_fit * 0.20 + shallow * 0.20
    return float(max(0.0, min(1.0, raw)))


def _cycle_bear_channel_raw(df: pd.DataFrame) -> float:
    """Mirror of bull channel for down direction."""
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    net_down = (closes[0] - closes[-1]) / max(highs.max() - lows.min(), 1e-9)
    drift = max(0.0, min(1.0, net_down * 2.0))

    lower = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
    lower_frac = lower / (len(closes) - 1)

    avg_body_ratio = np.mean([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    channel_body_fit = 1.0 - abs(avg_body_ratio - 0.45) / 0.45
    channel_body_fit = max(0.0, min(1.0, channel_body_fit))

    running_trough = np.minimum.accumulate(lows)
    rallies = (highs - running_trough) / np.maximum(running_trough, 1e-9)
    max_rally = rallies.max()
    shallow = max(0.0, 1.0 - max_rally * 5.0)

    raw = drift * 0.35 + lower_frac * 0.25 + channel_body_fit * 0.20 + shallow * 0.20
    return float(max(0.0, min(1.0, raw)))


def _cycle_trading_range_raw(df: pd.DataFrame) -> float:
    """Raw [0,1] for trading range: doji-heavy, balanced bull/bear, horizontal closes."""
    if len(df) < CYCLE_PHASE_LOOKBACK_BARS:
        return 0.0
    window = df.tail(CYCLE_PHASE_LOOKBACK_BARS)
    closes = window["close"].to_numpy()
    opens = window["open"].to_numpy()
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()

    # doji density
    body_ratios = np.array([
        abs(c - o) / max(h - l, 1e-9)
        for c, o, h, l in zip(closes, opens, highs, lows)
    ])
    doji_frac = float((body_ratios < DOJI_BODY_RATIO).mean())

    # bull / bear balance: closer to 0.5 = more rangebound
    bull_count = sum(1 for c, o in zip(closes, opens) if c > o)
    bull_frac = bull_count / len(closes)
    balance = 1.0 - abs(bull_frac - 0.5) * 2.0

    # low net drift relative to range
    total_range = highs.max() - lows.min()
    net_move = abs(closes[-1] - closes[0])
    drift_ratio = net_move / max(total_range, 1e-9)
    horizontal = max(0.0, 1.0 - drift_ratio * 2.5)   # drift > 40% of range → zero credit

    # close clustering: low stdev of closes relative to range
    if total_range > 0:
        close_stdev = closes.std()
        cluster = max(0.0, 1.0 - (close_stdev / total_range) * 4.0)
    else:
        cluster = 1.0

    raw = doji_frac * 0.3 + balance * 0.3 + horizontal * 0.25 + cluster * 0.15
    return float(max(0.0, min(1.0, raw)))


def _softmax(values: list[float], temperature: float) -> list[float]:
    """Numerically stable softmax."""
    arr = np.array(values) / max(temperature, 1e-9)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def classify_cycle_phase(df: pd.DataFrame) -> dict:
    """Layer-1 cycle-phase classifier.

    Returns a probability distribution over the five Brooks cycle phases
    (bull_spike, bear_spike, bull_channel, bear_channel, trading_range),
    plus the argmax label and its confidence.

    Returns {} if the classifier is disabled via kill-switch.
    """
    if not CYCLE_PHASE_CLASSIFIER_ENABLED:
        return {}
    if df is None or len(df) < 3:
        return {
            "probs": {p: 0.2 for p in CYCLE_PHASES},
            "top": "trading_range",
            "confidence": 0.2,
        }
    raw = [
        _cycle_bull_spike_raw(df),
        _cycle_bear_spike_raw(df),
        _cycle_bull_channel_raw(df),
        _cycle_bear_channel_raw(df),
        _cycle_trading_range_raw(df),
    ]
    probs = _softmax(raw, CYCLE_PHASE_SOFTMAX_TEMP)
    dist = dict(zip(CYCLE_PHASES, probs))
    top = max(dist, key=dist.get)
    return {
        "probs": {k: round(v, 3) for k, v in dist.items()},
        "top": top,
        "confidence": round(dist[top], 3),
        "raw": {k: round(v, 3) for k, v in zip(CYCLE_PHASES, raw)},
    }
