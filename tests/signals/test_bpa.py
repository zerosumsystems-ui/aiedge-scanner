"""Unit tests for aiedge.signals.bpa._score_bpa_patterns."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals import bpa as bpa_mod
from aiedge.signals.bpa import (
    BPA_MIN_CONFIDENCE,
    BPA_MIN_DF_LEN,
    BPA_RECENCY_BARS,
    _score_bpa_patterns,
)
from shared.bpa_detector import BPASetup


def trend_up_df(n: int = 20) -> pd.DataFrame:
    """Minimal df long enough to satisfy BPA_MIN_DF_LEN."""
    return pd.DataFrame({
        "open": [100 + i for i in range(n)],
        "high": [101 + i for i in range(n)],
        "low": [99 + i for i in range(n)],
        "close": [101 + i for i in range(n)],
        "volume": [100_000] * n,
    })


def mk_setup(setup_type: str, bar_index: int = 18,
             confidence: float = 0.8, entry: float = 110.0,
             stop: float = 109.0, target: float = 115.0,
             entry_mode: str = "stop") -> BPASetup:
    return BPASetup(
        detected=True,
        setup_type=setup_type,
        entry=entry,
        stop=stop,
        target=target,
        confidence=confidence,
        bar_index=bar_index,
        entry_mode=entry_mode,
    )


class ScoreBpaPatternsTests(unittest.TestCase):

    def test_disabled_returns_zero(self):
        with patch.object(bpa_mod, "BPA_INTEGRATION_ENABLED", False):
            score, active = _score_bpa_patterns(trend_up_df(), "up", "held")
            self.assertEqual(score, 0.0)
            self.assertEqual(active, [])

    def test_detector_unavailable_returns_zero(self):
        with patch.object(bpa_mod, "_BPA_AVAILABLE", False), \
             patch.object(bpa_mod, "_bpa_detect_all", None):
            score, active = _score_bpa_patterns(trend_up_df(), "up", "held")
            self.assertEqual(score, 0.0)

    def test_short_df_returns_zero(self):
        short_df = trend_up_df(BPA_MIN_DF_LEN - 1)
        score, _ = _score_bpa_patterns(short_df, "up", "held")
        self.assertEqual(score, 0.0)

    def test_detector_exception_returns_zero(self):
        def boom(df, adr=None):
            raise RuntimeError("detector broke")
        with patch.object(bpa_mod, "_bpa_detect_all", boom):
            score, active = _score_bpa_patterns(trend_up_df(), "up", "held")
            self.assertEqual(score, 0.0)
            self.assertEqual(active, [])

    def test_empty_setups_returns_zero(self):
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: []):
            score, active = _score_bpa_patterns(trend_up_df(), "up", "held")
            self.assertEqual(score, 0.0)

    def test_low_confidence_filtered(self):
        weak = [mk_setup("H2", confidence=BPA_MIN_CONFIDENCE - 0.01)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: weak):
            score, active = _score_bpa_patterns(trend_up_df(), "up", "held")
            self.assertEqual(score, 0.0)
            self.assertEqual(active, [])

    def test_old_detection_filtered_by_recency(self):
        # bar_index too old relative to end of df
        old = [mk_setup("H2", bar_index=0, confidence=0.9)]
        df = trend_up_df(30)  # last_bar=29, recency window=21
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: old):
            score, active = _score_bpa_patterns(df, "up", "held")
            self.assertEqual(score, 0.0)

    def test_h2_in_direction_bull_scores_plus_2(self):
        setups = [mk_setup("H2", bar_index=18)]
        df = trend_up_df(20)
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, active = _score_bpa_patterns(df, "up", "held")
            self.assertEqual(score, 2.0)
            self.assertEqual(active[0]["type"], "H2")

    def test_h1_in_direction_bull_scores_plus_1_5(self):
        setups = [mk_setup("H1", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, 1.5)

    def test_fl2_with_fill_recovered_scores_plus_1(self):
        setups = [mk_setup("FL2", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "filled_recovered")
            self.assertEqual(score, 1.0)

    def test_fl2_without_fill_recovered_scores_plus_0_5(self):
        setups = [mk_setup("FL2", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, 0.5)

    def test_counter_type_scores_plus_0_5(self):
        setups = [mk_setup("spike_channel", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, 0.5)

    def test_opposing_l2_on_gap_up_scores_neg_1(self):
        setups = [mk_setup("L2", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, -1.0)

    def test_opposing_l1_on_gap_up_scores_neg_0_5(self):
        setups = [mk_setup("L1", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, -0.5)

    def test_bear_direction_mirrors_bull(self):
        # L2 is long-side for bear direction
        setups = [mk_setup("L2", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, _ = _score_bpa_patterns(trend_up_df(20), "down", "held")
            self.assertEqual(score, 2.0)

    def test_multiple_setups_returns_best_in_direction(self):
        setups = [
            mk_setup("H1", bar_index=17, confidence=0.7),
            mk_setup("H2", bar_index=18, confidence=0.8),
            mk_setup("FL1", bar_index=19, confidence=0.9),
        ]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            score, active = _score_bpa_patterns(trend_up_df(20), "up", "held")
            self.assertEqual(score, 2.0)  # H2 wins
            self.assertLessEqual(len(active), 3)

    def test_active_setups_serialize_expected_keys(self):
        setups = [mk_setup("H2", bar_index=18)]
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: setups):
            _, active = _score_bpa_patterns(trend_up_df(20), "up", "held")
        for key in ("type", "entry", "stop", "target", "confidence",
                    "bar_index", "entry_mode"):
            self.assertIn(key, active[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
