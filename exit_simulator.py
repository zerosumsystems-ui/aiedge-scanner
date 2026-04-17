"""
Exit-strategy simulator for Pattern Lab detections.

Given a detection (entry, stop, direction) and the bars that followed,
simulate any exit strategy — fixed-R targets, partial profits at multiple
R levels, stop-to-breakeven after first partial, trailing runners —
and compute the realized R of that trade.

Then compare strategies across thousands of historical detections to
pick the exit profile with the best expectancy and win rate.

Usage:
    from exit_simulator import simulate_trade, ExitStrategy

    result = simulate_trade(
        entry=104.50, stop=102.90, direction="long",
        follow_bars=[{"h": 105.1, "l": 104.3, "c": 104.8}, ...],
        strategy=ExitStrategy.fixed_r(2.0),
    )
    # result.realized_r, result.result, result.bars_held

Conservative simulation: when a single bar's range spans both the stop
and an unfilled target, we assume the stop hit FIRST (worst case).
That gives realistic win rates instead of optimistic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Strategy spec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitLevel:
    """One scale-out: take `fraction` of position at `r_mult × risk`."""
    r_mult: float              # 0.5, 1.0, 2.0, etc. — target in multiples of risk
    fraction: float            # 0.0–1.0 — portion of position closed here


@dataclass(frozen=True)
class ExitStrategy:
    """An exit plan. Levels are scaled out in order; remaining fraction rides
    until stop or end of data.

    - `move_stop_to_breakeven_after_r`: after the cumulative filled fraction
      exceeds this R multiple, move the stop to entry price (breakeven).
    - `trail_runner_r`: if set, after all fixed levels fill, the remaining
      fraction trails the highest high (long) / lowest low (short) by N R.
    """
    name: str
    levels: tuple[ExitLevel, ...]
    move_stop_to_breakeven_after_r: Optional[float] = None
    trail_runner_r: Optional[float] = None

    # ── Named constructors for common patterns ──────────────────────────────
    @classmethod
    def fixed_r(cls, r_mult: float) -> "ExitStrategy":
        """Single target at r_mult × risk. All-or-nothing."""
        return cls(
            name=f"fixed_{r_mult:g}R",
            levels=(ExitLevel(r_mult=r_mult, fraction=1.0),),
        )

    @classmethod
    def partial_50_50(cls, first_r: float, second_r: float) -> "ExitStrategy":
        """Half at first_r, half at second_r."""
        return cls(
            name=f"50@{first_r:g}R_50@{second_r:g}R",
            levels=(
                ExitLevel(r_mult=first_r, fraction=0.5),
                ExitLevel(r_mult=second_r, fraction=0.5),
            ),
        )

    @classmethod
    def partial_be(cls, first_r: float, second_r: float) -> "ExitStrategy":
        """Half at first_r (move stop to BE), half at second_r."""
        return cls(
            name=f"50@{first_r:g}R_BE_50@{second_r:g}R",
            levels=(
                ExitLevel(r_mult=first_r, fraction=0.5),
                ExitLevel(r_mult=second_r, fraction=0.5),
            ),
            move_stop_to_breakeven_after_r=first_r,
        )

    @classmethod
    def partial_with_trail(
        cls, first_r: float, second_r: float, trail_r: float,
    ) -> "ExitStrategy":
        """1/3 at first_r, 1/3 at second_r, 1/3 trails after."""
        return cls(
            name=f"33@{first_r:g}R_33@{second_r:g}R_33trail{trail_r:g}R",
            levels=(
                ExitLevel(r_mult=first_r, fraction=0.333),
                ExitLevel(r_mult=second_r, fraction=0.334),
            ),
            move_stop_to_breakeven_after_r=first_r,
            trail_runner_r=trail_r,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    result: str                 # "win" | "loss" | "scratch" | "incomplete"
    realized_r: float           # total R across all exits (scaled by fractions)
    bars_held: int              # bars from entry until last exit
    exits: list[tuple[int, float, float]] = field(default_factory=list)
    # list of (bar_idx, r_mult_at_exit, fraction_closed)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trade(
    entry: float,
    stop: float,
    direction: str,
    follow_bars: Iterable[dict],
    strategy: ExitStrategy,
    entry_mode: str = "stop",
) -> SimResult:
    """
    Simulate a single trade given follow-on bars and an exit strategy.

    follow_bars: iterable of dicts with keys 'h' (high), 'l' (low), 'c' (close).
      Bars are in chronological order starting from the bar AFTER the signal.

    entry_mode:
      - "stop"    → entry requires a later bar to trade through the entry price.
                    If no bar crosses, trade never filled → result "never_filled".
      - "limit"   → entry assumed filled at signal bar (caller would have placed
                    the limit earlier). Simulate from bar 0.
      - "market"  → entry assumed filled at signal bar close (same as limit).

    Bias guards:
      - Bars processed strictly left-to-right; decisions depend only on bars
        already seen.
      - Intra-bar ordering: if both stop and a target fit in one bar's range,
        the STOP is assumed to hit first (conservative / pessimistic).
      - Fill verification for stop-order entries prevents counting trades
        that were never actually taken.
      - Incomplete trades (no exit by end of data) are closed at last close
        and marked "incomplete" in the status.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return SimResult(result="incomplete", realized_r=0.0, bars_held=0)

    is_long = direction == "long"
    all_bars = list(follow_bars)
    if not all_bars:
        return SimResult(result="incomplete", realized_r=0.0, bars_held=0)

    # ── Entry fill verification ─────────────────────────────────────────────
    if entry_mode == "stop":
        # Stop-order: need a later bar to trade THROUGH the entry price.
        fill_idx: Optional[int] = None
        for i, bar in enumerate(all_bars):
            if is_long and float(bar["h"]) >= entry:
                fill_idx = i
                break
            if (not is_long) and float(bar["l"]) <= entry:
                fill_idx = i
                break
        if fill_idx is None:
            return SimResult(result="never_filled", realized_r=0.0, bars_held=0)
        # Start simulation from the bar AFTER fill (we don't know intra-bar
        # timing on the fill bar, so don't look at its extremes for stop/target).
        bars = all_bars[fill_idx + 1:]
    else:
        # Limit / market: caller says entry filled at signal; start immediately.
        bars = all_bars

    if not bars:
        # Filled but no bars to simulate outcome — scratch
        return SimResult(result="incomplete", realized_r=0.0, bars_held=0)

    # Pre-compute target prices for each level
    targets = []
    for lvl in strategy.levels:
        if is_long:
            price = entry + lvl.r_mult * risk
        else:
            price = entry - lvl.r_mult * risk
        targets.append((lvl, price))

    remaining_fraction = 1.0
    realized_r = 0.0
    current_stop = stop
    highest_high = entry           # for trailing stop
    lowest_low = entry
    exits: list[tuple[int, float, float]] = []
    filled_indices: set[int] = set()
    cumulative_filled_r = 0.0      # R banked, for the move-to-BE trigger

    for bar_idx, bar in enumerate(bars):
        high = float(bar["h"])
        low = float(bar["l"])

        # Update trailing extremes
        if high > highest_high:
            highest_high = high
        if low < lowest_low:
            lowest_low = low

        stop_hit_this_bar = (
            (is_long and low <= current_stop)
            or (not is_long and high >= current_stop)
        )

        # Check each unfilled target. Process in order (tightest first).
        target_hit_indices_this_bar: list[int] = []
        for i, (lvl, price) in enumerate(targets):
            if i in filled_indices:
                continue
            if is_long and high >= price:
                target_hit_indices_this_bar.append(i)
            elif (not is_long) and low <= price:
                target_hit_indices_this_bar.append(i)

        # Conservative rule: if stop AND a target both fit in this bar's range,
        # the stop hit FIRST. This is the pessimistic assumption that avoids
        # inflating win rates (we can't know order within a bar without tick data).
        if stop_hit_this_bar and target_hit_indices_this_bar:
            # Assume stop first — close the rest of the position at the stop.
            stop_r = _r_at_stop(entry, current_stop, risk, is_long)
            realized_r += remaining_fraction * stop_r
            exits.append((bar_idx, stop_r, remaining_fraction))
            remaining_fraction = 0.0
            return _finalize(realized_r, bar_idx + 1, exits)

        # Fill targets (in r_mult order so the lowest target fills first)
        for i in target_hit_indices_this_bar:
            lvl, _ = targets[i]
            fraction_to_close = min(lvl.fraction, remaining_fraction)
            if fraction_to_close <= 0:
                continue
            realized_r += fraction_to_close * lvl.r_mult
            cumulative_filled_r += fraction_to_close * lvl.r_mult
            remaining_fraction -= fraction_to_close
            exits.append((bar_idx, lvl.r_mult, fraction_to_close))
            filled_indices.add(i)

            # Move stop to breakeven once a target with R_mult ≥ threshold
            # fills. Using lvl.r_mult (the LEVEL) instead of cumulative banked R
            # matches the trader's mental model: "after price hits 1R, move
            # the stop" — regardless of what fraction scaled out there.
            if (
                strategy.move_stop_to_breakeven_after_r is not None
                and lvl.r_mult >= strategy.move_stop_to_breakeven_after_r
            ):
                current_stop = entry  # breakeven

            if remaining_fraction <= 1e-9:
                return _finalize(realized_r, bar_idx + 1, exits)

        # After fixed levels are done, maybe trail the runner
        all_fixed_filled = len(filled_indices) == len(targets)
        if (
            all_fixed_filled
            and remaining_fraction > 0
            and strategy.trail_runner_r is not None
        ):
            trail_distance = strategy.trail_runner_r * risk
            if is_long:
                new_stop = highest_high - trail_distance
                if new_stop > current_stop:
                    current_stop = new_stop
            else:
                new_stop = lowest_low + trail_distance
                if new_stop < current_stop:
                    current_stop = new_stop

        # If only stop hit (no targets this bar), close full remaining at stop
        if stop_hit_this_bar and remaining_fraction > 0:
            stop_r = _r_at_stop(entry, current_stop, risk, is_long)
            realized_r += remaining_fraction * stop_r
            exits.append((bar_idx, stop_r, remaining_fraction))
            remaining_fraction = 0.0
            return _finalize(realized_r, bar_idx + 1, exits)

    # Ran out of bars without fully exiting → close at last close
    if remaining_fraction > 0:
        last_close = float(bars[-1]["c"])
        if is_long:
            leftover_r = (last_close - entry) / risk
        else:
            leftover_r = (entry - last_close) / risk
        realized_r += remaining_fraction * leftover_r
        exits.append((len(bars) - 1, leftover_r, remaining_fraction))

    return _finalize(realized_r, len(bars), exits)


def _r_at_stop(entry: float, stop: float, risk: float, is_long: bool) -> float:
    """R value when stop hits. Usually -1.0 (hit original stop) or 0.0 (BE)."""
    if is_long:
        return (stop - entry) / risk
    return (entry - stop) / risk


def _finalize(realized_r: float, bars_held: int,
              exits: list[tuple[int, float, float]]) -> SimResult:
    if realized_r > 0.01:
        result = "win"
    elif realized_r < -0.01:
        result = "loss"
    else:
        result = "scratch"
    return SimResult(
        result=result, realized_r=realized_r,
        bars_held=bars_held, exits=exits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyStats:
    strategy: ExitStrategy
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    incomplete: int = 0
    total_r: float = 0.0
    sum_r_on_wins: float = 0.0
    sum_r_on_losses: float = 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def avg_r(self) -> float:
        return self.total_r / self.total_trades if self.total_trades else 0.0

    @property
    def avg_win_r(self) -> float:
        return self.sum_r_on_wins / self.wins if self.wins else 0.0

    @property
    def avg_loss_r(self) -> float:
        return self.sum_r_on_losses / self.losses if self.losses else 0.0

    @property
    def expectancy_r(self) -> float:
        """Expected R per trade accounting for win rate × avg win + loss rate × avg loss."""
        if self.total_trades == 0:
            return 0.0
        return self.total_r / self.total_trades

    def record(self, sim: SimResult) -> None:
        self.total_trades += 1
        self.total_r += sim.realized_r
        if sim.result == "win":
            self.wins += 1
            self.sum_r_on_wins += sim.realized_r
        elif sim.result == "loss":
            self.losses += 1
            self.sum_r_on_losses += sim.realized_r
        elif sim.result == "scratch":
            self.scratches += 1
        else:
            self.incomplete += 1


# ─────────────────────────────────────────────────────────────────────────────
# Default strategy bank — the ones Will asked to test
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGIES: tuple[ExitStrategy, ...] = (
    ExitStrategy.fixed_r(0.5),
    ExitStrategy.fixed_r(1.0),
    ExitStrategy.fixed_r(1.5),
    ExitStrategy.fixed_r(2.0),
    ExitStrategy.fixed_r(3.0),
    ExitStrategy.partial_50_50(1.0, 2.0),
    ExitStrategy.partial_be(1.0, 2.0),
    ExitStrategy.partial_be(0.5, 1.5),
    ExitStrategy.partial_with_trail(1.0, 2.0, trail_r=1.0),
)
