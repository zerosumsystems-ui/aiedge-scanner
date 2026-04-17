"""Cross-instrument correlation + clustering.

Useful for position-sizing and dedup: if QQQ, XLK, SMH all show a
BUY signal but they're 0.95 correlated, that's effectively ONE bet,
not three. This module turns a wide bar DataFrame (one column per
ticker of log returns or close prices) into a correlation matrix
and a greedy single-linkage cluster assignment.

Functions are pure numpy/pandas — no plotting, no I/O.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Per-column log returns of a wide close-price DataFrame.

    First row is dropped (NaN from the diff). Zero / negative closes
    yield NaN rows, which downstream `correlation_matrix` handles via
    `DataFrame.corr(min_periods=...)`.
    """
    safe = closes.where(closes > 0)
    return np.log(safe).diff().dropna(how="all")


def correlation_matrix(returns: pd.DataFrame, min_periods: int = 5) -> pd.DataFrame:
    """Pearson correlation across columns of a returns DataFrame.

    `min_periods` guards against thin overlap — any pair with fewer
    aligned rows returns NaN in the output.
    """
    return returns.corr(min_periods=min_periods)


def cluster_by_threshold(
    corr: pd.DataFrame,
    threshold: float = 0.80,
) -> list[list[str]]:
    """Greedy single-linkage clustering: merge two tickers into the
    same cluster if their correlation is ≥ `threshold`.

    Returns a list of clusters (each a list of tickers). Ticker order
    inside a cluster is insertion order; cluster order is the order
    tickers first appear in `corr`.

    Tickers with any NaN correlations are assigned to their own
    singleton clusters.
    """
    tickers: Sequence[str] = list(corr.columns)
    # ticker → cluster index
    assignment: dict[str, int] = {}
    clusters: list[list[str]] = []

    for t in tickers:
        # Find an existing cluster this ticker links to
        linked = None
        for cluster_idx, cluster in enumerate(clusters):
            for member in cluster:
                val = corr.loc[t, member] if t in corr.index and member in corr.columns else np.nan
                if pd.notna(val) and val >= threshold:
                    linked = cluster_idx
                    break
            if linked is not None:
                break

        if linked is None:
            clusters.append([t])
            assignment[t] = len(clusters) - 1
        else:
            clusters[linked].append(t)
            assignment[t] = linked

    return clusters


def dedup_correlated(
    candidates: list[dict],
    corr: pd.DataFrame,
    threshold: float = 0.80,
    key: str = "ticker",
    rank_key: str = "urgency",
) -> list[dict]:
    """Given a list of candidate trade dicts sorted by `rank_key` desc,
    keep the top-ranked candidate from each correlation cluster.

    This complements the ETF-family dedup in signals/postprocess.py:
    that's a static hand-maintained map (QQQ ↔ TQQQ ↔ SQQQ); this is
    a data-driven filter that catches correlations the family map
    doesn't know about (e.g., SPY ↔ QQQ on a tech-heavy day).
    """
    if not candidates:
        return []

    clusters = cluster_by_threshold(corr, threshold)
    ticker_to_cluster: dict[str, int] = {}
    for cluster_idx, cluster in enumerate(clusters):
        for t in cluster:
            ticker_to_cluster[t] = cluster_idx

    seen_clusters: set[int] = set()
    out: list[dict] = []
    for c in candidates:
        t = c.get(key)
        cluster_idx = ticker_to_cluster.get(t)
        # Ticker outside the corr matrix → pass through unchanged
        if cluster_idx is None:
            out.append(c)
            continue
        if cluster_idx in seen_clusters:
            continue
        seen_clusters.add(cluster_idx)
        out.append(c)
    return out
