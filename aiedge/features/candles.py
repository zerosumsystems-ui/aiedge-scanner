"""Candle / bar helpers — pure math on a single OHLC row.

Every function takes a dict-like row with keys 'open', 'high', 'low',
'close' and returns a scalar. Zero dependencies, zero state.

Extracted from shared/brooks_score.py (Phase 3a of the scanner
refactor). Names preserve the leading underscore for backwards
compatibility with existing call sites in brooks_score.py and its
consumers — the underscore is historical, not semantic.
"""

# Minimum bar range to avoid division by zero.
# Historically lived in brooks_score.py; travels with the helpers now.
MIN_RANGE = 0.001

# Doji threshold: body < 30% of range = doji.
# Used by phase classifier and uncertainty scoring.
DOJI_BODY_RATIO = 0.30


def _safe_range(row) -> float:
    """Bar range, floored to avoid division by zero."""
    return max(row["high"] - row["low"], MIN_RANGE)


def _body(row) -> float:
    """Absolute body size (|close - open|)."""
    return abs(row["close"] - row["open"])


def _body_ratio(row) -> float:
    """Body as fraction of range. 0 = doji, 1 = full trend bar."""
    return _body(row) / _safe_range(row)


def _is_bull(row) -> bool:
    """True if close > open."""
    return row["close"] > row["open"]


def _is_bear(row) -> bool:
    """True if close < open."""
    return row["close"] < row["open"]


def _lower_tail_pct(row) -> float:
    """Lower tail as fraction of range.

    For a bull bar this is (open - low) / range.
    For a bear bar this is (close - low) / range.
    """
    rng = _safe_range(row)
    body_bottom = min(row["open"], row["close"])
    return (body_bottom - row["low"]) / rng


def _upper_tail_pct(row) -> float:
    """Upper tail as fraction of range.

    For a bull bar this is (high - close) / range.
    For a bear bar this is (high - open) / range.
    """
    rng = _safe_range(row)
    body_top = max(row["open"], row["close"])
    return (row["high"] - body_top) / rng


def _close_position(row) -> float:
    """Where the close sits within the bar's range.

    0.0 = close at low, 1.0 = close at high.
    """
    rng = _safe_range(row)
    return (row["close"] - row["low"]) / rng
