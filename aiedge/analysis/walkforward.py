"""Rolling train/test splits for walk-forward validation.

A parameter fit on all history overfits to regimes that may never
repeat. Walk-forward validation trains on a trailing window, tests on
the immediately-following window, then rolls both forward. This module
generates those splits — it does NOT run any model — so the same splits
can be shared across calibration, equity, and reliability analyses.

All three generators are lazy: they yield `(train_idx, test_idx)`
tuples of positional indices into the input series / dataframe.
Caller slices with `.iloc[...]`.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def rolling_window(
    n: int, train_size: int, test_size: int, step: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Rolling window of fixed-size train/test pairs.

    `n` is the total length. Yields tuples of positional index arrays
    — the train window slides forward by `step` (default = test_size)
    each iteration. The last partial test window is skipped (no
    truncation).

    Raises ValueError for non-positive train/test sizes or train_size + test_size > n.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError(f"train_size and test_size must be positive (got {train_size}, {test_size})")
    if train_size + test_size > n:
        raise ValueError(f"train_size ({train_size}) + test_size ({test_size}) exceeds n ({n})")
    step = step or test_size
    if step <= 0:
        raise ValueError(f"step must be positive (got {step})")

    start = 0
    while start + train_size + test_size <= n:
        train_idx = np.arange(start, start + train_size)
        test_idx = np.arange(start + train_size, start + train_size + test_size)
        yield train_idx, test_idx
        start += step


def expanding_window(
    n: int, min_train: int, test_size: int, step: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward: train window grows each iteration.

    Train window starts at index 0 and grows by `step` (default =
    test_size). Test window is always `test_size` bars immediately
    after the train window.
    """
    if min_train <= 0 or test_size <= 0:
        raise ValueError(f"min_train and test_size must be positive (got {min_train}, {test_size})")
    if min_train + test_size > n:
        raise ValueError(f"min_train ({min_train}) + test_size ({test_size}) exceeds n ({n})")
    step = step or test_size
    if step <= 0:
        raise ValueError(f"step must be positive (got {step})")

    train_end = min_train
    while train_end + test_size <= n:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        yield train_idx, test_idx
        train_end += step


def by_date(
    dates: pd.Series | pd.DatetimeIndex,
    train_days: int,
    test_days: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward splits keyed by *calendar days* rather than row count.

    Useful when detections per day vary wildly (a spiky name might have
    10 signals one day, 0 the next). Sort by `dates` ascending; yields
    (train_idx, test_idx) where train covers `train_days` of unique
    calendar days and test covers the next `test_days`.
    """
    if train_days <= 0 or test_days <= 0:
        raise ValueError(f"train_days and test_days must be positive (got {train_days}, {test_days})")

    dt = pd.to_datetime(pd.Series(dates)).dt.normalize()
    unique_days = sorted(dt.unique())
    total_days = len(unique_days)
    if total_days < train_days + test_days:
        return

    day_to_indices: dict = {}
    for i, d in enumerate(dt):
        day_to_indices.setdefault(d, []).append(i)

    start = 0
    while start + train_days + test_days <= total_days:
        train_days_set = unique_days[start:start + train_days]
        test_days_set = unique_days[start + train_days:start + train_days + test_days]

        train_idx = np.array(
            sorted([i for d in train_days_set for i in day_to_indices.get(d, [])]),
            dtype=int,
        )
        test_idx = np.array(
            sorted([i for d in test_days_set for i in day_to_indices.get(d, [])]),
            dtype=int,
        )
        yield train_idx, test_idx
        start += test_days
