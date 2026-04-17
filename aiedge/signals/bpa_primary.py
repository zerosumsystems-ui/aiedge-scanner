"""BPA-as-primary signal aggregator (Phase 5 architecture).

The scanner's original path: score 15 components → normalize to urgency
+ uncertainty → `_determine_signal` maps them to a label → `bpa_alignment`
is a post-hoc overlay that can upgrade/downgrade. BPA detection is an
afterthought.

This module inverts that: bpa_detector hits ARE the primary input.
Component scores become filters/modifiers, and the trader's equation
(p_win × reward vs p_loss × risk) is what decides whether to surface
each candidate.

Pipeline:

    bpa_detector.detect_all(df)
        → filter to in-direction + confidence + recency
        → for each surviving setup:
            - classify regime (features.regime.realized_vol_tercile)
            - classify HTF alignment (context.htf.classify_htf_alignment)
            - classify day type (context.daytype._classify_day_type)
            - look up empirical prior p_win(setup_type, regime, alignment, day_type)
            - compute trader-equation edge(prior, reward, risk)
            - keep only setups with positive edge
        → sort by edge desc
        → return top-N as ranked candidates

This module is introduced alongside the existing `aggregator.py`; the
live scanner opts in via `BPA_PRIMARY_ENABLED`. Once the calibration
loop has enough data in `priors_store` to populate the strata, we flip
the flag and retire `_score_bpa_patterns`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from aiedge.context.htf import classify_htf_alignment
from aiedge.features.regime import realized_vol_tercile
from aiedge.risk.priors import PriorLookup, p_win, trader_equation_edge
from aiedge.signals.bpa import (
    BPA_INTEGRATION_ENABLED, BPA_MIN_CONFIDENCE, BPA_MIN_DF_LEN,
    BPA_RECENCY_BARS, _BPA_AVAILABLE, _bpa_detect_all,
)
from aiedge.storage.priors_store import PriorsStore


# Kill-switch — flip to True once priors_store has enough data to
# make the edge calculation meaningful on real symbols.
BPA_PRIMARY_ENABLED = False

# Min sample count per stratum before we trust an empirical prior.
# Below this, fall back to the default (0.5 coin-flip) and the edge
# calculation loses its teeth — candidates still surface but ranking
# is down to reward/risk ratio alone.
BPA_PRIMARY_MIN_SAMPLES = 30


@dataclass(frozen=True)
class SetupCandidate:
    """One BPA-detected setup scored through the Phase 5 pipeline."""
    setup_type: str
    direction: str                        # "long" | "short"
    entry: float
    stop: float
    target: float
    confidence: float
    bar_index: int
    entry_mode: str
    regime: str                           # low|mid|high
    htf_alignment: str                    # aligned|mixed|opposed|no_data
    day_type: str
    prior: PriorLookup
    edge: float                           # p_win*reward - p_loss*risk
    reward_risk: float                    # |target-entry| / |entry-stop|


@dataclass
class AggregatorInputs:
    """Plumbing required to evaluate BPA setups against priors + context.

    Kept in a dataclass so the live scanner can build it once per scan
    and reuse across tickers in a cycle.
    """
    daily_closes_by_ticker: dict[str, list[float]] = field(default_factory=dict)
    weekly_closes_by_ticker: dict[str, list[float]] = field(default_factory=dict)
    daily_atrs_by_ticker: dict[str, list[float]] = field(default_factory=dict)
    priors_store: Optional[PriorsStore] = None


def _reward_risk(entry: float, stop: float, target: float) -> float:
    """Absolute reward/risk ratio; 0 if either leg is non-positive."""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0 or reward <= 0:
        return 0.0
    return reward / risk


def _direction_from_setup(setup_type: str) -> str:
    if setup_type in ("H1", "H2", "FL1", "FL2"):
        return "long"
    if setup_type in ("L1", "L2", "FH1", "FH2"):
        return "short"
    return "unknown"


def evaluate_setups(
    df: pd.DataFrame,
    ticker: str,
    day_type: str,
    inputs: AggregatorInputs,
) -> list[SetupCandidate]:
    """Run the BPA-primary pipeline for one symbol's scored df.

    Returns a list of surviving candidates sorted by `edge` descending.
    Empty list if: BPA disabled, detector unavailable, df too short,
    no confirmed setups, or all candidates have non-positive edge (when
    a priors_store is supplied with enough data).

    When `inputs.priors_store` is None or has insufficient data, the
    prior defaults to 0.5 — edge then reduces to `0.5 * (reward - risk)`
    and ranking falls back to pure R:R.
    """
    if not BPA_INTEGRATION_ENABLED or not _BPA_AVAILABLE or _bpa_detect_all is None:
        return []
    if len(df) < BPA_MIN_DF_LEN:
        return []

    try:
        raw_setups = _bpa_detect_all(df)
    except Exception:
        return []
    if not raw_setups:
        return []

    last_bar = len(df) - 1
    filtered = [
        s for s in raw_setups
        if s.confidence >= BPA_MIN_CONFIDENCE
        and s.bar_index >= last_bar - BPA_RECENCY_BARS
    ]
    if not filtered:
        return []

    # Context features — shared across all setups for this ticker
    daily_closes = inputs.daily_closes_by_ticker.get(ticker, [])
    weekly_closes = inputs.weekly_closes_by_ticker.get(ticker, [])
    daily_atrs = inputs.daily_atrs_by_ticker.get(ticker, [])
    regime = realized_vol_tercile(daily_closes) if daily_closes else "mid"

    out: list[SetupCandidate] = []
    for s in filtered:
        direction = _direction_from_setup(s.setup_type)
        if direction == "unknown":
            continue
        if s.entry is None or s.stop is None or s.target is None:
            continue

        htf = classify_htf_alignment(daily_closes, weekly_closes, direction)
        htf_alignment = htf["alignment"]

        # Empirical prior lookup with the full fallback hierarchy
        if inputs.priors_store is not None:
            prior = p_win(
                inputs.priors_store,
                setup_type=s.setup_type,
                regime=regime,
                htf_alignment=htf_alignment,
                day_type=day_type,
                min_samples=BPA_PRIMARY_MIN_SAMPLES,
            )
        else:
            prior = PriorLookup(p_win=0.5, wins=0, losses=0, matched_level="default")

        reward = abs(s.target - s.entry)
        risk = abs(s.entry - s.stop)
        edge = trader_equation_edge(prior, reward, risk)
        rr = _reward_risk(s.entry, s.stop, s.target)

        out.append(SetupCandidate(
            setup_type=s.setup_type,
            direction=direction,
            entry=float(s.entry),
            stop=float(s.stop),
            target=float(s.target),
            confidence=round(float(s.confidence), 2),
            bar_index=s.bar_index,
            entry_mode=s.entry_mode,
            regime=regime,
            htf_alignment=htf_alignment,
            day_type=day_type,
            prior=prior,
            edge=round(edge, 3),
            reward_risk=round(rr, 2),
        ))

    # Rank by edge desc; reject strictly-negative edges if we have real
    # prior data backing them (don't reject when the store was empty
    # — those defaulted to 0.5 and edge is just R:R - 1)
    has_real_priors = any(c.prior.matched_level != "default" for c in out)
    if has_real_priors:
        out = [c for c in out if c.edge > 0]
    out.sort(key=lambda c: c.edge, reverse=True)
    return out


def candidates_to_dicts(candidates: list[SetupCandidate]) -> list[dict]:
    """Serialize for dashboard / API / pattern_lab logging."""
    return [
        {
            "type": c.setup_type,
            "direction": c.direction,
            "entry": c.entry,
            "stop": c.stop,
            "target": c.target,
            "confidence": c.confidence,
            "bar_index": c.bar_index,
            "entry_mode": c.entry_mode,
            "regime": c.regime,
            "htf_alignment": c.htf_alignment,
            "day_type": c.day_type,
            "p_win": round(c.prior.p_win, 3),
            "prior_level": c.prior.matched_level,
            "prior_n": c.prior.n,
            "edge": c.edge,
            "rr_ratio": c.reward_risk,
        }
        for c in candidates
    ]
