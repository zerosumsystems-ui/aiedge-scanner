"""Exponential moving average — pure math on a closes array.

Extracted from shared/brooks_score.py (Phase 3b).
"""

import numpy as np


# Default EMA period. Historically lived in brooks_score.py; travels
# with the function now.
EMA_PERIOD = 20


def _compute_ema(closes: np.ndarray, period: int = EMA_PERIOD) -> np.ndarray:
    """Simple EMA calculation. Returns array same length as input.

    Standard EMA with multiplier = 2 / (period + 1). First value is
    seeded from closes[0] (not pre-warmed with an SMA), which keeps
    this identical to the legacy implementation — do not "improve"
    without regression-checking downstream scoring.
    """
    if len(closes) == 0:
        return np.array([])
    ema = np.zeros_like(closes, dtype=float)
    multiplier = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = closes[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema
