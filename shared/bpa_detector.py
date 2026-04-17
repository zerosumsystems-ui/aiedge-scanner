"""
Brooks Price Action (BPA) setup detector.

Self-contained — no dependency on AI Edge. Detects Brooks' canonical setups:
    H1, H2, L1, L2       — bar-count pullback entries
    FL1, FL2, FH1, FH2   — failed pullback reversals
    failed_bo            — failed breakout from a trading range
    spike_channel        — spike-then-channel continuation

Each detector runs on a DataFrame with columns: open, high, low, close, volume.
The signal bar is always the last bar of the DataFrame — the scanner calls this
on each new bar close. A setup fires when the signal bar is valid AND the
preceding structure matches Brooks' definition.

Source of truth for definitions:
    vault/Brooks PA/patterns/*.md
    ~/Brooks-Price-Action/skills/brooks-price-action/references/
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
    setup_type: str
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    confidence: float = 0.0
    bar_index: int = -1  # index of the signal bar in the input DataFrame
    entry_mode: str = "stop"  # "stop" | "limit" | "market" — Brooks' three entry methods
    # stop  = buy/sell on a subsequent bar trading through entry price
    # limit = resting order at entry; fills if price retraces to you
    # market = fill at the signal bar's close (not currently used)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_all(df: pd.DataFrame, adr: Optional[float] = None) -> list[BPASetup]:
    """
    Run every detector, return a list sorted most-recent signal bar first.

    If `adr` (average daily range for the instrument) is provided, apply
    cross-instrument normalization to stops and targets: minimum risk = 0.2×ADR,
    minimum reward = 1.5×ADR. This keeps a $5 stock's setups comparable to a
    $500 stock's and prevents tiny-bar stops from producing unrealistic R:R.
    """
    if len(df) < 10:
        return []

    setups: list[BPASetup] = []
    for detector in (
        _detect_h1, _detect_h2, _detect_l1, _detect_l2,
        _detect_fl1, _detect_fl2, _detect_fh1, _detect_fh2,
        _detect_spike_and_channel,
        _detect_failed_breakout,
    ):
        try:
            result = detector(df)
            if result and result.detected:
                if adr is not None and adr > 0:
                    result = _apply_adr_floors(result, adr)
                setups.append(result)
        except Exception as e:
            logger.warning(f"BPA detector {detector.__name__} failed: {e}")

    setups.sort(key=lambda s: s.bar_index, reverse=True)
    return setups


ADR_MIN_RISK_FRAC = 0.2
ADR_MIN_REWARD_FRAC = 1.5


def _apply_adr_floors(setup: BPASetup, adr: float) -> BPASetup:
    """Widen stops and targets to ADR-based minimums for cross-instrument parity."""
    if setup.entry is None or setup.stop is None or setup.target is None:
        return setup

    entry, stop, target = setup.entry, setup.stop, setup.target
    is_long = entry > stop
    min_risk = adr * ADR_MIN_RISK_FRAC
    min_reward = adr * ADR_MIN_REWARD_FRAC

    if is_long:
        if entry - stop < min_risk:
            stop = entry - min_risk
        if target - entry < min_reward:
            target = entry + min_reward
    else:
        if stop - entry < min_risk:
            stop = entry + min_risk
        if entry - target < min_reward:
            target = entry - min_reward

    return BPASetup(
        detected=setup.detected, setup_type=setup.setup_type,
        entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
        confidence=setup.confidence, bar_index=setup.bar_index,
        entry_mode=setup.entry_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bar classification — Brooks-grade
#
# Brooks distinguishes trend bars (body dominates, close near extreme) from
# weak bars and dojis. The old `close > open` test treats a pinbar with a
# long upper tail as bullish, which is backwards. These helpers replace it.
# ─────────────────────────────────────────────────────────────────────────────

TREND_BODY_RATIO = 0.50      # body must be ≥ 50% of bar range
TREND_CLOSE_TOP_THIRD = 2.0 / 3.0  # close in top/bottom third
SIGNAL_TAIL_RATIO = 0.25     # signal bar rejection tail < 25% of range
DOJI_BODY_RATIO = 0.30       # doji = body < 30% of range


def _bar_range(row) -> float:
    return float(row["high"]) - float(row["low"])


def _bar_body(row) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _body_ratio(row) -> float:
    r = _bar_range(row)
    if r <= 0:
        return 0.0
    return _bar_body(row) / r


def _is_doji(row) -> bool:
    return _body_ratio(row) < DOJI_BODY_RATIO


def _is_bull_trend_bar(row) -> bool:
    """Bull close, body ≥ 50% of range, close in top third."""
    if float(row["close"]) <= float(row["open"]):
        return False
    r = _bar_range(row)
    if r <= 0:
        return False
    if _body_ratio(row) < TREND_BODY_RATIO:
        return False
    close_position = (float(row["close"]) - float(row["low"])) / r
    return close_position >= TREND_CLOSE_TOP_THIRD


def _is_bear_trend_bar(row) -> bool:
    """Bear close, body ≥ 50% of range, close in bottom third."""
    if float(row["close"]) >= float(row["open"]):
        return False
    r = _bar_range(row)
    if r <= 0:
        return False
    if _body_ratio(row) < TREND_BODY_RATIO:
        return False
    close_position = (float(row["high"]) - float(row["close"])) / r
    return close_position >= TREND_CLOSE_TOP_THIRD


def _is_bull_signal_bar(row) -> bool:
    """Bull trend bar whose upper tail is small — not rejected at the high."""
    if not _is_bull_trend_bar(row):
        return False
    r = _bar_range(row)
    if r <= 0:
        return False
    upper_tail = float(row["high"]) - float(row["close"])
    return (upper_tail / r) < SIGNAL_TAIL_RATIO


def _is_bear_signal_bar(row) -> bool:
    """Bear trend bar whose lower tail is small — not rejected at the low."""
    if not _is_bear_trend_bar(row):
        return False
    r = _bar_range(row)
    if r <= 0:
        return False
    lower_tail = float(row["close"]) - float(row["low"])
    return (lower_tail / r) < SIGNAL_TAIL_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# Structural helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_bull_pullback(df: pd.DataFrame, signal_idx: int,
                        max_lookback: int = 8,
                        max_pb_len: int = 5) -> Optional[tuple[int, int, float]]:
    """
    For a long setup at signal_idx, find the pullback immediately preceding it.
    Pullback = contiguous bars after a leg-top that stayed below the leg-top
    high, with at least one bear trend bar or doji.

    H1 uses the default (max_pb_len=5, tight pullback). H2 uses a larger
    max_pb_len to allow the extended pullback that contains the failed H1.

    Returns (leg_top_idx, pullback_start_idx, pullback_high) or None.
    """
    if signal_idx < 2:
        return None
    start = max(0, signal_idx - max_lookback)
    if signal_idx - start < 2:
        return None

    # leg-top = highest high in [start, signal_idx)
    window = df.iloc[start:signal_idx]
    leg_top_rel = int(np.argmax(window["high"].values))
    leg_top_idx = start + leg_top_rel

    if leg_top_idx >= signal_idx - 1:
        return None

    pullback_start = leg_top_idx + 1
    pullback_end_excl = signal_idx
    pb_slice = df.iloc[pullback_start:pullback_end_excl]

    pb_len = len(pb_slice)
    if not (1 <= pb_len <= max_pb_len):
        return None

    if not any(_is_bear_trend_bar(r) or _is_doji(r) for _, r in pb_slice.iterrows()):
        return None

    leg_top_high = float(df.iloc[leg_top_idx]["high"])
    pullback_high = float(pb_slice["high"].max())
    if pullback_high > leg_top_high:
        return None  # pullback made a new high above leg-top = new leg, not a pullback

    return leg_top_idx, pullback_start, pullback_high


def _find_bear_pullback(df: pd.DataFrame, signal_idx: int,
                        max_lookback: int = 8,
                        max_pb_len: int = 5) -> Optional[tuple[int, int, float]]:
    """Mirror of _find_bull_pullback for short setups.

    Returns (leg_bottom_idx, pullback_start_idx, pullback_low) or None.
    """
    if signal_idx < 2:
        return None
    start = max(0, signal_idx - max_lookback)
    if signal_idx - start < 2:
        return None

    window = df.iloc[start:signal_idx]
    leg_bottom_rel = int(np.argmin(window["low"].values))
    leg_bottom_idx = start + leg_bottom_rel

    if leg_bottom_idx >= signal_idx - 1:
        return None

    pullback_start = leg_bottom_idx + 1
    pullback_end_excl = signal_idx
    pb_slice = df.iloc[pullback_start:pullback_end_excl]

    pb_len = len(pb_slice)
    if not (1 <= pb_len <= max_pb_len):
        return None

    if not any(_is_bull_trend_bar(r) or _is_doji(r) for _, r in pb_slice.iterrows()):
        return None

    leg_bottom_low = float(df.iloc[leg_bottom_idx]["low"])
    pullback_low = float(pb_slice["low"].min())
    if pullback_low < leg_bottom_low:
        return None  # pullback broke below leg-bottom = new leg, not a rally

    return leg_bottom_idx, pullback_start, pullback_low


def _leg_measured_move_up(df: pd.DataFrame, leg_top_idx: int,
                           lookback: int = 8) -> float:
    """Height of the up-leg ending at leg_top_idx (leg_top − prior swing low)."""
    origin_start = max(0, leg_top_idx - lookback)
    if origin_start > leg_top_idx:
        return 0.0
    origin_low = float(df.iloc[origin_start:leg_top_idx + 1]["low"].min())
    leg_high = float(df.iloc[leg_top_idx]["high"])
    return max(0.0, leg_high - origin_low)


def _leg_measured_move_down(df: pd.DataFrame, leg_bottom_idx: int,
                             lookback: int = 8) -> float:
    """Height of the down-leg ending at leg_bottom_idx (prior swing high − leg_bottom)."""
    origin_start = max(0, leg_bottom_idx - lookback)
    if origin_start > leg_bottom_idx:
        return 0.0
    origin_high = float(df.iloc[origin_start:leg_bottom_idx + 1]["high"].max())
    leg_low = float(df.iloc[leg_bottom_idx]["low"])
    return max(0.0, origin_high - leg_low)


def _has_up_leg(df: pd.DataFrame, end_idx_excl: int, min_bars: int = 3) -> bool:
    """Confirm the bars ending at end_idx_excl-1 represent an up-leg."""
    start = max(0, end_idx_excl - 8)
    if end_idx_excl - start < min_bars:
        return False
    leg = df.iloc[start:end_idx_excl]
    # Net progress: last close above first open
    if float(leg.iloc[-1]["close"]) <= float(leg.iloc[0]["open"]):
        return False
    # At least one bull trend bar somewhere in the leg
    if not any(_is_bull_trend_bar(r) for _, r in leg.iterrows()):
        return False
    return True


def _has_down_leg(df: pd.DataFrame, end_idx_excl: int, min_bars: int = 3) -> bool:
    """Confirm the bars ending at end_idx_excl-1 represent a down-leg."""
    start = max(0, end_idx_excl - 8)
    if end_idx_excl - start < min_bars:
        return False
    leg = df.iloc[start:end_idx_excl]
    if float(leg.iloc[-1]["close"]) >= float(leg.iloc[0]["open"]):
        return False
    if not any(_is_bear_trend_bar(r) for _, r in leg.iterrows()):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# H1 / L1 — first pullback in an established trend
# ─────────────────────────────────────────────────────────────────────────────

def _detect_h1(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    Brooks H1: first pullback in a bull trend.

    Structure: up-leg → pullback (1–5 bars, may include dojis or bear bars) →
    bull signal bar whose high breaks the prior bar's high.

    The signal bar is the last bar of df.
    """
    if len(df) < 7:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bull_signal_bar(signal):
        return None
    # Must break above prior bar high (the H1 trigger condition)
    if float(signal["high"]) <= float(df.iloc[signal_idx - 1]["high"]):
        return None

    pb = _find_bull_pullback(df, signal_idx, max_lookback=8)
    if pb is None:
        return None
    leg_top_idx, pb_start, pb_high = pb

    # Up-leg must exist before the pullback
    if not _has_up_leg(df, leg_top_idx + 1):
        return None

    # This must be the FIRST such pullback-break — i.e., no prior bull signal
    # bar broke a prior-bar high within the up-leg-pullback-signal window.
    # (We leave H2 detection to flag the second instance.)

    entry_price = float(signal["high"])
    stop_price = float(signal["low"])
    risk = entry_price - stop_price
    if risk <= 0:
        return None

    # Target: measured move of the preceding up-leg (Brooks standard)
    mm = _leg_measured_move_up(df, leg_top_idx)
    target_price = entry_price + mm if mm > 0 else entry_price + 2 * risk
    if target_price - entry_price < 2 * risk:
        target_price = entry_price + 2 * risk

    return BPASetup(
        detected=True,
        setup_type="H1",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.65,
        bar_index=signal_idx,
        entry_mode="stop",
    )


def _detect_l1(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks L1: mirror of H1 in a bear trend."""
    if len(df) < 7:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bear_signal_bar(signal):
        return None
    if float(signal["low"]) >= float(df.iloc[signal_idx - 1]["low"]):
        return None

    pb = _find_bear_pullback(df, signal_idx, max_lookback=8)
    if pb is None:
        return None
    leg_bottom_idx, pb_start, pb_low = pb

    if not _has_down_leg(df, leg_bottom_idx + 1):
        return None

    entry_price = float(signal["low"])
    stop_price = float(signal["high"])
    risk = stop_price - entry_price
    if risk <= 0:
        return None

    mm = _leg_measured_move_down(df, leg_bottom_idx)
    target_price = entry_price - mm if mm > 0 else entry_price - 2 * risk
    if entry_price - target_price < 2 * risk:
        target_price = entry_price - 2 * risk

    return BPASetup(
        detected=True,
        setup_type="L1",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.65,
        bar_index=signal_idx,
        entry_mode="stop",
    )


# ─────────────────────────────────────────────────────────────────────────────
# H2 / L2 — within-pullback bar counting
#
# Brooks H2 is NOT a second pullback after a continuation leg that made a new
# trend high. It is the second bull signal bar WITHIN the same pullback, after
# the first one (H1) failed — i.e. after H1 triggered, price traded below H1's
# low before the next bar-counting signal fired. The pullback is a single
# extended correction containing both H1 and H2.
# ─────────────────────────────────────────────────────────────────────────────

def _detect_h2(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    Brooks H2: second buy signal inside an extended pullback.

    Requires:
      1. Last bar is a bull signal bar that breaks the prior bar's high.
      2. One extended pullback below a prior leg-top (up to ~12 bars).
      3. A prior bull signal bar (the failed H1) exists inside that pullback.
      4. After the prior H1, price made a lower low than H1's low (H1 failed).
    """
    if len(df) < 10:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bull_signal_bar(signal):
        return None
    if float(signal["high"]) <= float(df.iloc[signal_idx - 1]["high"]):
        return None

    # Extended pullback: allow up to 12 bars, lookback to 15
    pb = _find_bull_pullback(df, signal_idx, max_lookback=15, max_pb_len=12)
    if pb is None:
        return None
    leg_top_idx, pb_start, pb_high = pb

    # Need at least 3 bars of pullback so a prior H1 has room to form and fail
    if signal_idx - pb_start < 3:
        return None

    # Search for a prior bull signal bar that broke its prior bar's high,
    # located inside the pullback (between pb_start and signal_idx - 2)
    prior_h1_idx: Optional[int] = None
    for i in range(signal_idx - 2, pb_start - 1, -1):
        if i < 1:
            break
        bar = df.iloc[i]
        if not _is_bull_signal_bar(bar):
            continue
        if float(bar["high"]) <= float(df.iloc[i - 1]["high"]):
            continue
        prior_h1_idx = i
        break
    if prior_h1_idx is None:
        return None

    # H1 failure: between prior H1 and signal, price must have traded below H1's low
    between = df.iloc[prior_h1_idx + 1:signal_idx]
    if between.empty:
        return None
    if float(between["low"].min()) >= float(df.iloc[prior_h1_idx]["low"]):
        return None  # H1 didn't fail — not an H2

    entry_price = float(signal["high"])
    stop_price = float(signal["low"])
    risk = entry_price - stop_price
    if risk <= 0:
        return None

    # Target: measured move of the preceding up-leg
    mm = _leg_measured_move_up(df, leg_top_idx)
    target_price = entry_price + mm if mm > 0 else entry_price + 2 * risk
    if target_price - entry_price < 2 * risk:
        target_price = entry_price + 2 * risk

    return BPASetup(
        detected=True,
        setup_type="H2",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.72,
        bar_index=signal_idx,
        entry_mode="stop",
    )


def _detect_l2(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks L2: mirror of H2 — second sell signal inside an extended rally."""
    if len(df) < 10:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bear_signal_bar(signal):
        return None
    if float(signal["low"]) >= float(df.iloc[signal_idx - 1]["low"]):
        return None

    pb = _find_bear_pullback(df, signal_idx, max_lookback=15, max_pb_len=12)
    if pb is None:
        return None
    leg_bottom_idx, pb_start, pb_low = pb

    if signal_idx - pb_start < 3:
        return None

    prior_l1_idx: Optional[int] = None
    for i in range(signal_idx - 2, pb_start - 1, -1):
        if i < 1:
            break
        bar = df.iloc[i]
        if not _is_bear_signal_bar(bar):
            continue
        if float(bar["low"]) >= float(df.iloc[i - 1]["low"]):
            continue
        prior_l1_idx = i
        break
    if prior_l1_idx is None:
        return None

    between = df.iloc[prior_l1_idx + 1:signal_idx]
    if between.empty:
        return None
    if float(between["high"].max()) <= float(df.iloc[prior_l1_idx]["high"]):
        return None  # L1 didn't fail

    entry_price = float(signal["low"])
    stop_price = float(signal["high"])
    risk = stop_price - entry_price
    if risk <= 0:
        return None

    mm = _leg_measured_move_down(df, leg_bottom_idx)
    target_price = entry_price - mm if mm > 0 else entry_price - 2 * risk
    if entry_price - target_price < 2 * risk:
        target_price = entry_price - 2 * risk

    return BPASetup(
        detected=True,
        setup_type="L2",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.72,
        bar_index=signal_idx,
        entry_mode="stop",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FL1 / FL2 / FH1 / FH2 — failed pullbacks
#
# An FL1 is: an L1 short that triggered (or attempted to) but failed —
# either the bar after the L1 signal bar never traded below the L1 low,
# or it triggered and immediately reversed. Entry is long on a confirming
# bull signal bar.
# ─────────────────────────────────────────────────────────────────────────────

def _detect_fl1(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks Failed L1: L1 short attempt fails, entering long on the reversal."""
    if len(df) < 8:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bull_signal_bar(signal):
        return None

    # Search last 1-4 bars before the signal bar for the failed L1 signal bar
    for offset in range(1, 5):
        l1_idx = signal_idx - offset
        if l1_idx < 2:
            continue
        l1_bar = df.iloc[l1_idx]
        if not _is_bear_signal_bar(l1_bar):
            continue
        # Valid L1 structural: must have had a pullback up in a bear trend
        pb = _find_bear_pullback(df, l1_idx, max_lookback=6)
        if pb is None:
            continue
        _, _, _ = pb
        if not _has_down_leg(df, l1_idx - 1):
            continue

        # Failure of L1: between L1 and signal, no bar traded below L1 low;
        # OR the signal bar itself closes back above L1 high.
        between = df.iloc[l1_idx + 1:signal_idx]
        failed = False
        if not between.empty and float(between["low"].min()) >= float(l1_bar["low"]):
            failed = True
        if float(signal["close"]) > float(l1_bar["high"]):
            failed = True
        if not failed:
            continue

        # FL1 — LIMIT buy at the failed L1's low (Brooks: "buy the trap").
        # Stop sits just below that low; if price trades back through, the
        # failure has itself failed.
        l1_low = float(l1_bar["low"])
        buffer = max(0.05, _bar_range(l1_bar) * 0.25)
        entry_price = l1_low
        stop_price = l1_low - buffer
        risk = entry_price - stop_price
        if risk <= 0:
            continue

        # Target: measured move of the failed bear leg
        origin_start = max(0, l1_idx - 6)
        leg_high = float(df.iloc[origin_start:l1_idx]["high"].max())
        mm_target = entry_price + (leg_high - l1_low)
        target_price = max(mm_target, entry_price + 2 * risk)

        return BPASetup(
            detected=True,
            setup_type="FL1",
            entry=round(entry_price, 2),
            stop=round(stop_price, 2),
            target=round(target_price, 2),
            confidence=0.60,
            bar_index=signal_idx,
            entry_mode="limit",
        )

    return None


def _detect_fl2(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks Failed L2: the second L short attempt fails (bears have tried twice)."""
    if len(df) < 11:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bull_signal_bar(signal):
        return None

    # Find L2 (most recent failed short) — bear signal bar within last 4 bars
    l2_idx = None
    for offset in range(1, 5):
        i = signal_idx - offset
        if i < 4:
            continue
        bar = df.iloc[i]
        if not _is_bear_signal_bar(bar):
            continue
        if _find_bear_pullback(df, i, max_lookback=5) is None:
            continue
        l2_idx = i
        break
    if l2_idx is None:
        return None

    # Find prior L1 before L2, with a pullback-up between them
    l1_idx = None
    for i in range(l2_idx - 2, max(0, l2_idx - 12), -1):
        if i < 2:
            break
        bar = df.iloc[i]
        if not _is_bear_signal_bar(bar):
            continue
        if _find_bear_pullback(df, i, max_lookback=5) is None:
            continue
        between = df.iloc[i + 1:l2_idx]
        if between.empty:
            continue
        # Must see a bull retrace between L1 and L2
        if any(_is_bull_trend_bar(r) or _is_doji(r) for _, r in between.iterrows()):
            l1_idx = i
            break
    if l1_idx is None:
        return None

    # L2 failed
    between_l2 = df.iloc[l2_idx + 1:signal_idx]
    l2_bar = df.iloc[l2_idx]
    failed = False
    if not between_l2.empty and float(between_l2["low"].min()) >= float(l2_bar["low"]):
        failed = True
    if float(signal["close"]) > float(l2_bar["high"]):
        failed = True
    if not failed:
        return None

    # FL2 — LIMIT buy at the failed L2's low
    l2_low = float(l2_bar["low"])
    buffer = max(0.05, _bar_range(l2_bar) * 0.25)
    entry_price = l2_low
    stop_price = l2_low - buffer
    risk = entry_price - stop_price
    if risk <= 0:
        return None

    origin_start = max(0, l1_idx - 6)
    leg_high = float(df.iloc[origin_start:l1_idx]["high"].max())
    leg_low = float(df.iloc[l1_idx:signal_idx]["low"].min())
    mm_target = entry_price + (leg_high - leg_low)
    target_price = max(mm_target, entry_price + 2 * risk)

    return BPASetup(
        detected=True,
        setup_type="FL2",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.68,
        bar_index=signal_idx,
        entry_mode="limit",
    )


def _detect_fh1(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks Failed H1: mirror of FL1 — an H1 long attempt fails, entering short."""
    if len(df) < 8:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bear_signal_bar(signal):
        return None

    for offset in range(1, 5):
        h1_idx = signal_idx - offset
        if h1_idx < 2:
            continue
        h1_bar = df.iloc[h1_idx]
        if not _is_bull_signal_bar(h1_bar):
            continue
        if _find_bull_pullback(df, h1_idx, max_lookback=6) is None:
            continue
        if not _has_up_leg(df, h1_idx - 1):
            continue

        between = df.iloc[h1_idx + 1:signal_idx]
        failed = False
        if not between.empty and float(between["high"].max()) <= float(h1_bar["high"]):
            failed = True
        if float(signal["close"]) < float(h1_bar["low"]):
            failed = True
        if not failed:
            continue

        # FH1 — LIMIT sell at the failed H1's high (mirror of FL1)
        h1_high = float(h1_bar["high"])
        buffer = max(0.05, _bar_range(h1_bar) * 0.25)
        entry_price = h1_high
        stop_price = h1_high + buffer
        risk = stop_price - entry_price
        if risk <= 0:
            continue

        origin_start = max(0, h1_idx - 6)
        leg_low = float(df.iloc[origin_start:h1_idx]["low"].min())
        mm_target = entry_price - (h1_high - leg_low)
        target_price = min(mm_target, entry_price - 2 * risk)

        return BPASetup(
            detected=True,
            setup_type="FH1",
            entry=round(entry_price, 2),
            stop=round(stop_price, 2),
            target=round(target_price, 2),
            confidence=0.60,
            bar_index=signal_idx,
            entry_mode="limit",
        )

    return None


def _detect_fh2(df: pd.DataFrame) -> Optional[BPASetup]:
    """Brooks Failed H2: mirror of FL2 — the second H long attempt fails."""
    if len(df) < 11:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]
    if not _is_bear_signal_bar(signal):
        return None

    h2_idx = None
    for offset in range(1, 5):
        i = signal_idx - offset
        if i < 4:
            continue
        bar = df.iloc[i]
        if not _is_bull_signal_bar(bar):
            continue
        if _find_bull_pullback(df, i, max_lookback=5) is None:
            continue
        h2_idx = i
        break
    if h2_idx is None:
        return None

    h1_idx = None
    for i in range(h2_idx - 2, max(0, h2_idx - 12), -1):
        if i < 2:
            break
        bar = df.iloc[i]
        if not _is_bull_signal_bar(bar):
            continue
        if _find_bull_pullback(df, i, max_lookback=5) is None:
            continue
        between = df.iloc[i + 1:h2_idx]
        if between.empty:
            continue
        if any(_is_bear_trend_bar(r) or _is_doji(r) for _, r in between.iterrows()):
            h1_idx = i
            break
    if h1_idx is None:
        return None

    between_h2 = df.iloc[h2_idx + 1:signal_idx]
    h2_bar = df.iloc[h2_idx]
    failed = False
    if not between_h2.empty and float(between_h2["high"].max()) <= float(h2_bar["high"]):
        failed = True
    if float(signal["close"]) < float(h2_bar["low"]):
        failed = True
    if not failed:
        return None

    # FH2 — LIMIT sell at the failed H2's high
    h2_high = float(h2_bar["high"])
    buffer = max(0.05, _bar_range(h2_bar) * 0.25)
    entry_price = h2_high
    stop_price = h2_high + buffer
    risk = stop_price - entry_price
    if risk <= 0:
        return None

    origin_start = max(0, h1_idx - 6)
    leg_low = float(df.iloc[origin_start:h1_idx]["low"].min())
    leg_high = float(df.iloc[h1_idx:signal_idx]["high"].max())
    mm_target = entry_price - (leg_high - leg_low)
    target_price = min(mm_target, entry_price - 2 * risk)

    return BPASetup(
        detected=True,
        setup_type="FH2",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.68,
        bar_index=signal_idx,
        entry_mode="limit",
    )


# ─────────────────────────────────────────────────────────────────────────────
# spike_channel — CONTINUATION (not a reversal)
#
# Brooks: spike (1-5 strong trend bars) → shallower channel drifting in the
# same direction → pullback inside the channel → with-trend signal bar.
# Entry is with the trend. Target is a measured move of the spike.
# ─────────────────────────────────────────────────────────────────────────────

def _find_spike(df: pd.DataFrame, direction: str,
                search_end_idx: int, min_len: int = 2, max_len: int = 5
                ) -> Optional[tuple[int, int]]:
    """Find the most recent spike of `min_len..max_len` trend bars in direction."""
    check = _is_bull_trend_bar if direction == "bull" else _is_bear_trend_bar
    start_search = max(0, search_end_idx - 20)

    # Walk backward, look for contiguous trend bars
    best: Optional[tuple[int, int]] = None
    i = search_end_idx - 1
    while i >= start_search:
        if check(df.iloc[i]):
            # Extend left to find spike start
            spike_end = i
            spike_start = i
            while spike_start - 1 >= start_search and check(df.iloc[spike_start - 1]):
                spike_start -= 1
            length = spike_end - spike_start + 1
            if min_len <= length <= max_len:
                best = (spike_start, spike_end)
                break
            # If longer than max_len, still acceptable — clip to max_len on the right
            if length > max_len:
                best = (spike_end - max_len + 1, spike_end)
                break
            i = spike_start - 1
            continue
        i -= 1

    return best


def _detect_spike_and_channel(df: pd.DataFrame) -> Optional[BPASetup]:
    """Spike-and-channel continuation. Matches vault/Brooks PA/patterns/spike_channel.md."""
    if len(df) < 10:
        return None

    signal_idx = len(df) - 1
    signal = df.iloc[signal_idx]

    for direction in ("bull", "bear"):
        if direction == "bull" and not _is_bull_signal_bar(signal):
            continue
        if direction == "bear" and not _is_bear_signal_bar(signal):
            continue

        # Signal bar must confirm the with-trend entry
        if direction == "bull" and float(signal["high"]) <= float(df.iloc[signal_idx - 1]["high"]):
            continue
        if direction == "bear" and float(signal["low"]) >= float(df.iloc[signal_idx - 1]["low"]):
            continue

        # Pullback immediately before signal (shallow inside the channel)
        if direction == "bull":
            pb = _find_bull_pullback(df, signal_idx, max_lookback=5)
        else:
            pb = _find_bear_pullback(df, signal_idx, max_lookback=5)
        if pb is None:
            continue
        leg_extreme_idx, pb_start, _ = pb

        # Spike: 2-5 contiguous trend bars, at or before pb_start, with at least
        # 3 bars of channel between spike end and pb_start
        spike = _find_spike(df, direction, search_end_idx=pb_start - 2)
        if spike is None:
            continue
        spike_start, spike_end = spike
        channel_start = spike_end + 1
        channel_end_excl = pb_start
        if channel_end_excl - channel_start < 3:
            continue

        # Channel drifts in spike direction and at a shallower slope than the spike
        spike_slice = df.iloc[spike_start:spike_end + 1]
        channel_slice = df.iloc[channel_start:channel_end_excl]
        if direction == "bull":
            spike_move = float(spike_slice["close"].iloc[-1]) - float(spike_slice["open"].iloc[0])
            ch_move = float(channel_slice["close"].iloc[-1]) - float(channel_slice["close"].iloc[0])
            if spike_move <= 0 or ch_move <= 0:
                continue
        else:
            spike_move = float(spike_slice["open"].iloc[0]) - float(spike_slice["close"].iloc[-1])
            ch_move = float(channel_slice["close"].iloc[0]) - float(channel_slice["close"].iloc[-1])
            if spike_move <= 0 or ch_move <= 0:
                continue

        spike_slope = spike_move / max(1, len(spike_slice))
        ch_slope = ch_move / max(1, len(channel_slice))
        if ch_slope >= spike_slope:
            continue  # channel must be shallower

        # Measured move off spike
        if direction == "bull":
            entry_price = float(signal["high"])
            stop_price = float(signal["low"])
            risk = entry_price - stop_price
            if risk <= 0:
                continue
            target_price = entry_price + spike_move
            if target_price <= entry_price:
                target_price = entry_price + 2 * risk
        else:
            entry_price = float(signal["low"])
            stop_price = float(signal["high"])
            risk = stop_price - entry_price
            if risk <= 0:
                continue
            target_price = entry_price - spike_move
            if target_price >= entry_price:
                target_price = entry_price - 2 * risk

        return BPASetup(
            detected=True,
            setup_type="spike_channel",
            entry=round(entry_price, 2),
            stop=round(stop_price, 2),
            target=round(target_price, 2),
            confidence=0.65,
            bar_index=signal_idx,
            entry_mode="stop",
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# failed_bo — failed breakout from a real trading range
# ─────────────────────────────────────────────────────────────────────────────

RANGE_MIN_BARS = 15
RANGE_MIN_BOUNDARY_TESTS = 2
RANGE_WIDTH_AVG_RANGE_MULT = 3.0
BREAKOUT_LOOKBACK_BARS = 3  # breakout and failure must occur in last N bars


def _detect_failed_breakout(df: pd.DataFrame) -> Optional[BPASetup]:
    """
    Failed breakout from a real trading range. The range must be:
      - At least RANGE_MIN_BARS long
      - With width ≥ RANGE_WIDTH_AVG_RANGE_MULT × the avg bar range within it
      - With ≥ RANGE_MIN_BOUNDARY_TESTS tests of each boundary (high and low)

    A breakout is a bar closing beyond a boundary within the last
    BREAKOUT_LOOKBACK_BARS, and the failure is a subsequent bar closing back
    inside the range.
    """
    if len(df) < RANGE_MIN_BARS + BREAKOUT_LOOKBACK_BARS:
        return None

    range_end_excl = len(df) - BREAKOUT_LOOKBACK_BARS
    range_start = max(0, range_end_excl - 20)  # use up to 20 bars for range
    range_slice = df.iloc[range_start:range_end_excl]
    if len(range_slice) < RANGE_MIN_BARS:
        return None

    RH = float(range_slice["high"].max())
    RL = float(range_slice["low"].min())
    width = RH - RL
    avg_bar_range = float((range_slice["high"] - range_slice["low"]).mean())
    if avg_bar_range <= 0:
        return None
    if width < RANGE_WIDTH_AVG_RANGE_MULT * avg_bar_range:
        return None  # range too narrow to matter

    tol = 0.25 * avg_bar_range
    high_tests = int(sum(1 for _, r in range_slice.iterrows() if float(r["high"]) >= RH - tol))
    low_tests = int(sum(1 for _, r in range_slice.iterrows() if float(r["low"]) <= RL + tol))
    if high_tests < RANGE_MIN_BOUNDARY_TESTS or low_tests < RANGE_MIN_BOUNDARY_TESTS:
        return None

    # Strict 2-bar pattern: breakout bar = signal_idx - 1, failure bar = signal_idx.
    # This means the detector fires EXACTLY on the bar that completes the failure,
    # never on stale signals where the failure happened earlier in the window.
    signal_idx = len(df) - 1
    breakout_bar = df.iloc[signal_idx - 1]
    failure_bar = df.iloc[signal_idx]

    breakout_direction: Optional[str] = None
    if float(breakout_bar["close"]) > RH and float(failure_bar["close"]) < RH:
        breakout_direction = "up"
    elif float(breakout_bar["close"]) < RL and float(failure_bar["close"]) > RL:
        breakout_direction = "down"
    else:
        return None

    failure_abs_idx = signal_idx

    if breakout_direction == "up":
        entry_price = RH  # short at the reclaimed boundary
        stop_price = float(breakout_bar["high"])
        target_price = RL
        risk = stop_price - entry_price
    else:
        entry_price = RL
        stop_price = float(breakout_bar["low"])
        target_price = RH
        risk = entry_price - stop_price

    if risk <= 0:
        return None

    return BPASetup(
        detected=True,
        setup_type="failed_bo",
        entry=round(entry_price, 2),
        stop=round(stop_price, 2),
        target=round(target_price, 2),
        confidence=0.70,
        bar_index=failure_abs_idx,
        entry_mode="limit",
    )
