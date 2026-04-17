"""Tests for aiedge.signals.bpa_primary — the Phase 5 BPA-as-primary aggregator."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.signals import bpa as bpa_mod
from aiedge.signals import bpa_primary
from aiedge.signals.bpa_primary import (
    AggregatorInputs, SetupCandidate, candidates_to_dicts, evaluate_setups,
)
from aiedge.storage.priors_store import PriorsStore
from shared.bpa_detector import BPASetup


def df_n(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [100 + i for i in range(n)],
        "high": [101 + i for i in range(n)],
        "low":  [99 + i for i in range(n)],
        "close":[101 + i for i in range(n)],
        "volume": [1000] * n,
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


def _mk_inputs(store: PriorsStore | None = None) -> AggregatorInputs:
    return AggregatorInputs(
        daily_closes_by_ticker={"AAPL": [100 + i for i in range(30)]},
        weekly_closes_by_ticker={"AAPL": [100 + i * 3 for i in range(15)]},
        priors_store=store,
    )


class EvaluateSetupsTests(unittest.TestCase):

    def test_disabled_returns_empty(self):
        with patch.object(bpa_mod, "BPA_INTEGRATION_ENABLED", False):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
            self.assertEqual(out, [])

    def test_short_df_returns_empty(self):
        out = evaluate_setups(df_n(5), "AAPL", "trend_from_open", _mk_inputs())
        self.assertEqual(out, [])

    def test_no_setups_returns_empty(self):
        with patch.object(bpa_mod, "_bpa_detect_all", lambda df, adr=None: []):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
            self.assertEqual(out, [])

    def test_low_confidence_filtered(self):
        setups = [mk_setup("H2", confidence=0.3)]
        with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
            self.assertEqual(out, [])

    def test_with_no_priors_surfaces_positive_rr_setup(self):
        # reward 5, risk 1 → 5:1 R:R, edge = 0.5*5 - 0.5*1 = 2.0
        setups = [mk_setup("H2", entry=110, stop=109, target=115)]
        with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].setup_type, "H2")
        self.assertEqual(out[0].direction, "long")
        self.assertGreater(out[0].edge, 0)

    def test_ranks_multiple_setups_by_edge(self):
        setups = [
            mk_setup("H2", entry=110, stop=109, target=111, bar_index=18),   # RR=1
            mk_setup("H1", entry=110, stop=109, target=115, bar_index=19),   # RR=5
        ]
        with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].setup_type, "H1")  # better RR wins

    def test_negative_edge_rejected_when_priors_present(self):
        # Prior: 0 wins, 30 losses → p_win ≈ 0, edge negative
        with PriorsStore(":memory:") as store:
            for _ in range(30):
                store.record_outcome("H2", "high", "aligned", "trend_from_open", won=False)

            setups = [mk_setup("H2", entry=110, stop=109, target=111)]
            inputs = _mk_inputs(store)
            with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
                out = evaluate_setups(df_n(), "AAPL", "trend_from_open", inputs)
            self.assertEqual(out, [])

    def test_positive_edge_surfaces_with_priors(self):
        with PriorsStore(":memory:") as store:
            for _ in range(30):
                store.record_outcome("H2", "high", "aligned", "trend_from_open", won=True)

            setups = [mk_setup("H2", entry=110, stop=109, target=115)]
            inputs = _mk_inputs(store)
            with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
                out = evaluate_setups(df_n(), "AAPL", "trend_from_open", inputs)
            self.assertEqual(len(out), 1)
            self.assertGreater(out[0].edge, 0)
            # prior matched at some level
            self.assertIn(out[0].prior.matched_level,
                         ("exact", "regime+align", "regime", "setup"))

    def test_bear_setup_gets_short_direction(self):
        setups = [mk_setup("L2", entry=100, stop=101, target=95)]
        with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
            out = evaluate_setups(df_n(), "AAPL", "trend_from_open", _mk_inputs())
        self.assertEqual(out[0].direction, "short")


class CandidatesToDictsTests(unittest.TestCase):

    def test_serialization_keys(self):
        setups = [mk_setup("H2")]
        with patch.object(bpa_primary, "_bpa_detect_all", lambda df, adr=None: setups):
            candidates = evaluate_setups(df_n(), "AAPL", "tfo", _mk_inputs())
        dicts = candidates_to_dicts(candidates)
        self.assertEqual(len(dicts), 1)
        expected_keys = {"type", "direction", "entry", "stop", "target",
                         "confidence", "bar_index", "entry_mode", "regime",
                         "htf_alignment", "day_type", "p_win", "prior_level",
                         "prior_n", "edge", "rr_ratio"}
        self.assertEqual(set(dicts[0].keys()), expected_keys)

    def test_empty_candidates_empty_dicts(self):
        self.assertEqual(candidates_to_dicts([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
