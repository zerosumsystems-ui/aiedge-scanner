"""Empirical-prior lookup with fallback hierarchy.

Wraps `aiedge.storage.priors_store.PriorsStore` with a small bit of
intelligence: if the exact stratum has too few observations, back off
to a less-specific key until we find one with enough data.

Lookup order (most specific → most general):
  1. (setup_type, regime, htf_alignment, day_type)
  2. (setup_type, regime, htf_alignment, "any")
  3. (setup_type, regime, "any", "any")
  4. (setup_type, "any", "any", "any")
  5. overall static fallback (e.g., 0.50 for a coin-flip prior)

`record_outcome` always writes the fully-specific key — the fallback
only kicks in at read time.

Used by the signal aggregator (Phase 5+) to compute trader's-equation
edge from empirical hit rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiedge.storage.priors_store import PriorsStore


MIN_SAMPLES_FOR_SPECIFIC = 30   # below this, back off to a broader key
DEFAULT_PRIOR_PWIN = 0.50       # last-resort coin-flip


@dataclass(frozen=True)
class PriorLookup:
    """Result of a prior lookup: the probability + which key level matched."""
    p_win: float
    wins: int
    losses: int
    matched_level: str    # "exact" | "regime+align" | "regime" | "setup" | "default"

    @property
    def n(self) -> int:
        return self.wins + self.losses


def p_win(
    store: PriorsStore,
    setup_type: str,
    regime: str,
    htf_alignment: str,
    day_type: str,
    min_samples: int = MIN_SAMPLES_FOR_SPECIFIC,
) -> PriorLookup:
    """Probability of winning this setup given the full context, with fallback.

    Tries the most-specific key first; falls back to progressively
    broader keys when a level has < `min_samples` observations. If
    nothing hits, returns `DEFAULT_PRIOR_PWIN` at the "default" level.
    """
    # Level 1: exact
    wins, losses = store.get(setup_type, regime, htf_alignment, day_type)
    if wins + losses >= min_samples:
        return PriorLookup(
            p_win=_rate(wins, losses),
            wins=wins, losses=losses, matched_level="exact",
        )

    # Level 2: drop day_type
    rollup_w, rollup_l = _rollup(store, setup_type, regime=regime,
                                  htf_alignment=htf_alignment)
    if rollup_w + rollup_l >= min_samples:
        return PriorLookup(
            p_win=_rate(rollup_w, rollup_l),
            wins=rollup_w, losses=rollup_l,
            matched_level="regime+align",
        )

    # Level 3: drop htf_alignment
    rollup_w, rollup_l = _rollup(store, setup_type, regime=regime)
    if rollup_w + rollup_l >= min_samples:
        return PriorLookup(
            p_win=_rate(rollup_w, rollup_l),
            wins=rollup_w, losses=rollup_l,
            matched_level="regime",
        )

    # Level 4: setup_type only
    rollup_w, rollup_l = _rollup(store, setup_type)
    if rollup_w + rollup_l >= min_samples:
        return PriorLookup(
            p_win=_rate(rollup_w, rollup_l),
            wins=rollup_w, losses=rollup_l,
            matched_level="setup",
        )

    # Level 5: fall back to coin-flip, include whatever we have
    return PriorLookup(
        p_win=DEFAULT_PRIOR_PWIN,
        wins=rollup_w, losses=rollup_l,
        matched_level="default",
    )


def trader_equation_edge(prior: PriorLookup, reward: float, risk: float) -> float:
    """Expected-value edge per unit risk.

    `edge = p_win * reward - p_loss * risk`. Positive edge = take the
    trade; negative = skip.
    """
    if risk <= 0:
        return 0.0
    p_loss = 1.0 - prior.p_win
    return prior.p_win * reward - p_loss * risk


def _rate(wins: int, losses: int) -> float:
    total = wins + losses
    return wins / total if total > 0 else DEFAULT_PRIOR_PWIN


def _rollup(store: PriorsStore, setup_type: str,
            regime: str | None = None,
            htf_alignment: str | None = None) -> tuple[int, int]:
    """Sum wins/losses across dimensions not constrained."""
    rows = store.all_by_setup(setup_type)
    wins = losses = 0
    for row in rows:
        if regime is not None and row["regime"] != regime:
            continue
        if htf_alignment is not None and row["htf_alignment"] != htf_alignment:
            continue
        wins += row["wins"]
        losses += row["losses"]
    return wins, losses
