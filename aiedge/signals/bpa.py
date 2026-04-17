"""BPA pattern scorer — aligns Brooks setup detections with gap direction.

Consumes Setup objects produced by `shared/bpa_detector.detect_all` and
returns an alignment score (−1.0 … +2.0) that downstream aggregators
use to upgrade / downgrade signals.

Extracted from shared/brooks_score.py (Phase 3h). The BPA detector
module itself will migrate to aiedge/signals/bpa_detector.py in a
later phase; for now the import still points at shared/.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── BPA detector integration (guarded — graceful fallback if module missing) ──
try:
    from shared.bpa_detector import detect_all as _bpa_detect_all
    _BPA_AVAILABLE = True
except ImportError:
    _bpa_detect_all = None
    _BPA_AVAILABLE = False


# ── Tunables ─────────────────────────────────────────────────────────
BPA_INTEGRATION_ENABLED = True        # False → disables BPA overlay + bear-flip detection

# ── BPA pattern integration ──
BPA_LONG_SETUP_TYPES = frozenset({"H1", "H2", "FL1", "FL2"})
BPA_SHORT_SETUP_TYPES = frozenset({"L1", "L2"})
BPA_COUNTER_TYPES = frozenset({"spike_channel", "failed_bo"})
BPA_MIN_CONFIDENCE = 0.60
BPA_RECENCY_BARS = 8        # only consider setups detected in last N bars of the DataFrame
BPA_MIN_DF_LEN = 12          # minimum bars before running BPA detectors


def _score_bpa_patterns(
    df: pd.DataFrame,
    gap_direction: str,
    gap_fill_status: str,
) -> tuple[float, list[dict]]:
    """Run bpa_detector on df and score how well detected patterns align with direction.

    Returns (bpa_alignment_score, bpa_active_setups):
      +2.0  H2/L2 in-direction (highest-confidence pullback)
      +1.5  H1/L1 in-direction
      +1.0  FL1/FL2 after gap fill recovery (reversal confirmation)
      +0.5  failed_bo or spike_channel in-direction
       0.0  no recent in-direction setup
      -1.0  strong opposing setup (L2 on gap-up, H2 on gap-down)

    The bpa_active_setups list carries serialized setup dicts for details/dashboard.
    """
    if not BPA_INTEGRATION_ENABLED or not _BPA_AVAILABLE or _bpa_detect_all is None:
        return 0.0, []
    if len(df) < BPA_MIN_DF_LEN:
        return 0.0, []

    try:
        raw_setups = _bpa_detect_all(df)
    except Exception:
        logger.warning("bpa_detector.detect_all raised — skipping BPA scoring")
        return 0.0, []

    if not raw_setups:
        return 0.0, []

    # Filter by recency and confidence
    last_bar = len(df) - 1
    filtered = [
        s for s in raw_setups
        if s.confidence >= BPA_MIN_CONFIDENCE
        and s.bar_index >= last_bar - BPA_RECENCY_BARS
    ]
    if not filtered:
        return 0.0, []

    # Determine which types are "in-direction" vs "opposing"
    if gap_direction == "up":
        in_dir_types = BPA_LONG_SETUP_TYPES
        opp_types = BPA_SHORT_SETUP_TYPES
    else:
        in_dir_types = BPA_SHORT_SETUP_TYPES
        opp_types = BPA_LONG_SETUP_TYPES

    best_score = 0.0
    for s in filtered:
        st = s.setup_type
        if st in in_dir_types:
            if st in ("H2", "L2"):
                best_score = max(best_score, 2.0)
            elif st in ("H1", "L1"):
                best_score = max(best_score, 1.5)
            elif st in ("FL1", "FL2") and gap_fill_status == "filled_recovered":
                best_score = max(best_score, 1.0)
            elif st in ("FL1", "FL2"):
                best_score = max(best_score, 0.5)
        elif st in BPA_COUNTER_TYPES:
            best_score = max(best_score, 0.5)
        elif st in opp_types:
            if st in ("H2", "L2"):
                best_score = min(best_score, -1.0)
            else:
                best_score = min(best_score, -0.5)

    # Serialize for details dict
    active = [
        {
            "type": s.setup_type,
            "entry": s.entry,
            "stop": s.stop,
            "target": s.target,
            "confidence": round(s.confidence, 2),
            "bar_index": s.bar_index,
            "entry_mode": s.entry_mode,
        }
        for s in filtered[:3]
    ]

    return round(best_score, 2), active
