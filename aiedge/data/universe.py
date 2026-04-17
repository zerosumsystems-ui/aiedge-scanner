"""Default scanner universe.

Pulls the ~500-symbol gap-scan universe from the screener pipeline
with a hard-coded S&P-leaders fallback if the screener isn't
importable (smoke tests, standalone research notebooks, etc.).

Extracted from shared/brooks_score.py (Phase 3i).
"""

import logging

logger = logging.getLogger(__name__)


_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "AVGO", "CVX",
    "LLY", "MRK", "ABBV", "PEP", "KO", "COST", "ADBE", "CRM", "WMT",
    "TMO", "ACN", "MCD", "CSCO", "ABT", "DHR", "LIN", "NEE", "TXN",
    "AMGN", "PM", "RTX", "LOW", "UNP", "HON", "IBM", "QCOM", "SPGI",
    "AMAT", "DE", "GE", "CAT", "BKNG", "NOW", "ISRG", "ADP", "MRVL",
    "SPY", "QQQ", "IWM", "SMH", "XLE", "XLF",
]


def _get_default_universe() -> list[str]:
    """Import the full symbol universe from the screener.

    Falls back to a small default if screener isn't available.
    """
    try:
        from stages.screener import _default_gap_universe
        return _default_gap_universe()
    except ImportError:
        logger.warning("Could not import screener universe, using fallback S&P leaders")
        return list(_FALLBACK_UNIVERSE)
