"""
Standalone Brooks Price Action (BPA) setup detector.
Self-contained — no dependency on AI Edge.

Detects: H1, H2, L1, L2, FL1, FL2, spike & channel, failed breakout.
Each detector takes a price series and returns setup details.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BPASetup:
    detected: bool
    setup_type: str  # "H1", "H2", "L1", "L2", "FL1", "FL2", "spike_channel", "failed_bo"
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    confidence: float = 0.0
    bar_index: int = -1  # which bar triggered the setup


def detect_all(df: pd.DataFrame) -> list[BPASetup]:
    """
    Run all BPA detectors on a DataFrame with columns: open, high, low, close, volume.
    Returns list of detected setups sorted by bar_index descending (most recent first).
    """
    if len(df) < 10:
        return []

    setups = []
    for detector in [
        _detect_h1, _detect_h2, _detect_l1, _detect_l2,
        _detect_fl1, _detect_fl2,
        _detect_spike_and_channel,
        _detect_failed_breakout,
    ]:
        try:
            result = detector(df)
            if result and result.detected:
                setups.append(result)
        except Exception as e:
            logger.warning(f"BPA detector {detector.__name__} failed: {e}")

    setups.sort(key=lambda s: s.bar_index, reverse=True)
    return setups


def _bar_is_bull(row) -> bool:
    return row["close"] > row["open"]


def _bar_is_bear(row) -> bool:
    return row["close"] < row["open"]


def _bar_body(row) -> float:
    return abs(row["close"] - row["open"])


def _bar_range(row) -> float:
    return row["high"] - row["low"]


def _detect_h1(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    H1: First pullback in a bull trend.
    Look for: uptrend → first bear bar (or doji) → bull bar closing above prior bar high.
    """
    if len(df) < 5:
        return None

    # Check for uptrend: at least 3 of last 7 bars are bull with higher closes
    recent = df.iloc[-7:]
    bull_count = sum(1 for _, r in recent.iterrows() if _bar_is_bull(r))
    if bull_count < 4:
        return None

    # Look for pullback bar followed by entry bar
    for i in range(len(df) - 3, len(df) - 1):
        pb_bar = df.iloc[i]
        entry_bar = df.iloc[i + 1] if i + 1 < len(df) else None
        if entry_bar is None:
            continue

        # Pullback bar should be bearish
        if not _bar_is_bear(pb_bar):
            continue

        # Check this is the FIRST pullback (no bear bars in prior 3 bars)
        prior_bears = sum(1 for j in range(max(0, i - 3), i) if _bar_is_bear(df.iloc[j]))
        if prior_bears > 0:
            continue

        # Entry bar should be bullish and close above pullback bar high
        if _bar_is_bull(entry_bar) and entry_bar["close"] > pb_bar["high"]:
            entry_price = pb_bar["high"]
            stop_price = pb_bar["low"]
            risk = entry_price - stop_price
            target_price = entry_price + (2 * risk)

            return BPASetup(
                detected=True,
                setup_type="H1",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.65,
                bar_index=i + 1,
            )

    return None


def _detect_h2(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    H2: Second pullback in a bull trend (higher probability than H1).
    Look for: uptrend → first pullback → continuation → second pullback → bull entry.
    """
    if len(df) < 10:
        return None

    # Need uptrend context
    recent = df.iloc[-12:]
    bull_count = sum(1 for _, r in recent.iterrows() if _bar_is_bull(r))
    if bull_count < 5:
        return None

    # Count pullbacks (consecutive bear bars)
    pullback_count = 0
    in_pullback = False
    last_pb_end = -1

    for i in range(len(df) - 10, len(df) - 1):
        if _bar_is_bear(df.iloc[i]):
            if not in_pullback:
                pullback_count += 1
                in_pullback = True
        else:
            if in_pullback:
                last_pb_end = i
            in_pullback = False

    if pullback_count < 2:
        return None

    # Entry on the bar after the second pullback ends
    if last_pb_end > 0 and last_pb_end < len(df) - 1:
        entry_bar = df.iloc[last_pb_end]
        pb_low = df.iloc[last_pb_end - 1]["low"] if last_pb_end > 0 else df.iloc[last_pb_end]["low"]

        if _bar_is_bull(entry_bar):
            entry_price = entry_bar["high"]
            stop_price = pb_low
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            target_price = entry_price + (2 * risk)

            return BPASetup(
                detected=True,
                setup_type="H2",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.72,
                bar_index=last_pb_end,
            )

    return None


def _detect_l1(df: pd.DataFrame) -> Optional[BPASetup]:
    """L1: First pullback in a bear trend (mirror of H1)."""
    if len(df) < 5:
        return None

    recent = df.iloc[-7:]
    bear_count = sum(1 for _, r in recent.iterrows() if _bar_is_bear(r))
    if bear_count < 4:
        return None

    for i in range(len(df) - 3, len(df) - 1):
        pb_bar = df.iloc[i]
        entry_bar = df.iloc[i + 1] if i + 1 < len(df) else None
        if entry_bar is None:
            continue

        if not _bar_is_bull(pb_bar):
            continue

        prior_bulls = sum(1 for j in range(max(0, i - 3), i) if _bar_is_bull(df.iloc[j]))
        if prior_bulls > 0:
            continue

        if _bar_is_bear(entry_bar) and entry_bar["close"] < pb_bar["low"]:
            entry_price = pb_bar["low"]
            stop_price = pb_bar["high"]
            risk = stop_price - entry_price
            target_price = entry_price - (2 * risk)

            return BPASetup(
                detected=True,
                setup_type="L1",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.65,
                bar_index=i + 1,
            )

    return None


def _detect_l2(df: pd.DataFrame) -> Optional[BPASetup]:
    """L2: Second pullback in a bear trend (mirror of H2)."""
    if len(df) < 10:
        return None

    recent = df.iloc[-12:]
    bear_count = sum(1 for _, r in recent.iterrows() if _bar_is_bear(r))
    if bear_count < 5:
        return None

    pullback_count = 0
    in_pullback = False
    last_pb_end = -1

    for i in range(len(df) - 10, len(df) - 1):
        if _bar_is_bull(df.iloc[i]):
            if not in_pullback:
                pullback_count += 1
                in_pullback = True
        else:
            if in_pullback:
                last_pb_end = i
            in_pullback = False

    if pullback_count < 2:
        return None

    if last_pb_end > 0 and last_pb_end < len(df) - 1:
        entry_bar = df.iloc[last_pb_end]
        pb_high = df.iloc[last_pb_end - 1]["high"] if last_pb_end > 0 else df.iloc[last_pb_end]["high"]

        if _bar_is_bear(entry_bar):
            entry_price = entry_bar["low"]
            stop_price = pb_high
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            target_price = entry_price - (2 * risk)

            return BPASetup(
                detected=True,
                setup_type="L2",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.72,
                bar_index=last_pb_end,
            )

    return None


def _detect_fl1(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    FL1 (Failed L1): Bear trend attempts L1 short, but the pullback
    bar closes strongly bullish, trapping bears. Entry is long.
    """
    if len(df) < 7:
        return None

    recent = df.iloc[-7:]
    bear_count = sum(1 for _, r in recent.iterrows() if _bar_is_bear(r))
    if bear_count < 3:
        return None

    for i in range(len(df) - 3, len(df) - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1] if i + 1 < len(df) else None
        if next_bar is None:
            continue

        # Bull bar that closes above the prior swing high = failed L1
        if _bar_is_bull(bar) and _bar_body(bar) > 0.6 * _bar_range(bar):
            prior_high = df.iloc[max(0, i - 3):i]["high"].max()
            if bar["close"] > prior_high and _bar_is_bull(next_bar):
                entry_price = bar["high"]
                stop_price = bar["low"]
                risk = entry_price - stop_price
                if risk <= 0:
                    continue
                target_price = entry_price + (2 * risk)

                return BPASetup(
                    detected=True,
                    setup_type="FL1",
                    entry=round(entry_price, 2),
                    stop=round(stop_price, 2),
                    target=round(target_price, 2),
                    confidence=0.60,
                    bar_index=i,
                )

    return None


def _detect_fl2(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    FL2 (Failed L2): Second bear pullback fails, trapping bears.
    Strong bull reversal bar after two pullback attempts in a bear trend.
    """
    if len(df) < 12:
        return None

    # Need bear context that's weakening
    recent = df.iloc[-12:]
    bear_count = sum(1 for _, r in recent.iterrows() if _bar_is_bear(r))
    if bear_count < 4:
        return None

    # Look for the pattern at the end of the series
    for i in range(len(df) - 4, len(df) - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1] if i + 1 < len(df) else None
        if next_bar is None:
            continue

        # Strong bull reversal bar
        if _bar_is_bull(bar) and _bar_body(bar) > 0.5 * _bar_range(bar):
            # Should be near the low of recent range
            recent_low = df.iloc[max(0, i - 5):i]["low"].min()
            if bar["low"] <= recent_low * 1.005:  # within 0.5% of recent low
                if _bar_is_bull(next_bar):
                    entry_price = bar["high"]
                    stop_price = bar["low"]
                    risk = entry_price - stop_price
                    if risk <= 0:
                        continue
                    target_price = entry_price + (2 * risk)

                    return BPASetup(
                        detected=True,
                        setup_type="FL2",
                        entry=round(entry_price, 2),
                        stop=round(stop_price, 2),
                        target=round(target_price, 2),
                        confidence=0.68,
                        bar_index=i,
                    )

    return None


def _detect_spike_and_channel(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    Spike & Channel: Strong directional move (spike) followed by a
    weaker channel in the same direction. Trade the channel breakout reversal.
    """
    if len(df) < 20:
        return None

    # Look for a spike in the first third, then a channel in the rest
    third = len(df) // 3
    spike_section = df.iloc[:third]
    channel_section = df.iloc[third:]

    # Detect bull spike: >70% of bars are bull, large range
    spike_bulls = sum(1 for _, r in spike_section.iterrows() if _bar_is_bull(r))
    spike_is_bull = spike_bulls > 0.7 * len(spike_section)

    spike_bears = sum(1 for _, r in spike_section.iterrows() if _bar_is_bear(r))
    spike_is_bear = spike_bears > 0.7 * len(spike_section)

    if not (spike_is_bull or spike_is_bear):
        return None

    # Channel should be weaker trend in same direction
    channel_range = channel_section["high"].max() - channel_section["low"].min()
    spike_range = spike_section["high"].max() - spike_section["low"].min()

    if channel_range > spike_range:
        return None  # Channel shouldn't be bigger than spike

    # Check for channel trend line break at the end
    last_bar = df.iloc[-1]
    second_last = df.iloc[-2]

    if spike_is_bull:
        # Bull spike + bull channel → look for bear reversal at end
        channel_low_trend = channel_section["low"].min()
        if _bar_is_bear(last_bar) and last_bar["close"] < second_last["low"]:
            entry_price = second_last["low"]
            stop_price = channel_section["high"].max()
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            target_price = entry_price - risk  # 1:1 for channel reversal

            return BPASetup(
                detected=True,
                setup_type="spike_channel",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.62,
                bar_index=len(df) - 1,
            )
    else:
        # Bear spike + bear channel → look for bull reversal at end
        if _bar_is_bull(last_bar) and last_bar["close"] > second_last["high"]:
            entry_price = second_last["high"]
            stop_price = channel_section["low"].min()
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            target_price = entry_price + risk

            return BPASetup(
                detected=True,
                setup_type="spike_channel",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.62,
                bar_index=len(df) - 1,
            )

    return None


def _detect_failed_breakout(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    Failed Breakout: Price breaks above/below a key level (recent high/low)
    but immediately reverses. Trade the reversal.
    """
    if len(df) < 10:
        return None

    # Find recent range (excluding last 2 bars)
    range_section = df.iloc[-10:-2]
    range_high = range_section["high"].max()
    range_low = range_section["low"].min()

    breakout_bar = df.iloc[-2]
    reversal_bar = df.iloc[-1]

    # Failed bull breakout: breaks above range high, then reverses
    if breakout_bar["high"] > range_high and _bar_is_bear(reversal_bar):
        if reversal_bar["close"] < range_high:
            entry_price = range_high
            stop_price = breakout_bar["high"]
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            target_price = entry_price - (1.5 * risk)

            return BPASetup(
                detected=True,
                setup_type="failed_bo",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.70,
                bar_index=len(df) - 1,
            )

    # Failed bear breakout: breaks below range low, then reverses
    if breakout_bar["low"] < range_low and _bar_is_bull(reversal_bar):
        if reversal_bar["close"] > range_low:
            entry_price = range_low
            stop_price = breakout_bar["low"]
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            target_price = entry_price + (1.5 * risk)

            return BPASetup(
                detected=True,
                setup_type="failed_bo",
                entry=round(entry_price, 2),
                stop=round(stop_price, 2),
                target=round(target_price, 2),
                confidence=0.70,
                bar_index=len(df) - 1,
            )

    return None
