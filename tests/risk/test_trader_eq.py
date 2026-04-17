"""Unit tests for aiedge.risk.trader_eq."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiedge.risk.trader_eq import _compute_risk_reward


def mk_df(opens, closes, highs=None, lows=None):
    if highs is None:
        highs = [max(o, c) + 0.05 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def trend_up_df(n: int = 10, step: float = 1.0) -> pd.DataFrame:
    opens = [100 + step * i for i in range(n)]
    closes = [100 + step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


def trend_down_df(n: int = 10, step: float = 1.0) -> pd.DataFrame:
    opens = [200 - step * i for i in range(n)]
    closes = [200 - step * (i + 1) for i in range(n)]
    return mk_df(opens, closes)


class ComputeRiskRewardTests(unittest.TestCase):

    def test_bull_trend_has_positive_reward(self):
        df = trend_up_df(10)
        risk, reward, rr = _compute_risk_reward(df, "up", prior_close=99.0, spike_bars=4)
        self.assertGreater(risk, 0)
        self.assertGreaterEqual(reward, 0)
        self.assertGreaterEqual(rr, 0)

    def test_bear_trend_has_positive_reward(self):
        df = trend_down_df(10)
        risk, reward, rr = _compute_risk_reward(df, "down", prior_close=201.0, spike_bars=4)
        self.assertGreater(risk, 0)
        self.assertGreaterEqual(reward, 0)

    def test_risk_never_below_min_floor(self):
        # flat df — tiny bar ranges
        df = mk_df([100.0] * 10, [100.01] * 10,
                   highs=[100.02] * 10, lows=[99.99] * 10)
        risk, _, _ = _compute_risk_reward(df, "up", prior_close=99.0, spike_bars=2)
        # min_risk = max(avg_bar_range * 0.5, MIN_RANGE) — always positive
        self.assertGreater(risk, 0.0)

    def test_rr_zero_when_risk_at_floor(self):
        # Flat df — current equals recent extreme, risk = min_risk floor.
        df = mk_df([100.0] * 10, [100.0] * 10,
                   highs=[100.0] * 10, lows=[100.0] * 10)
        _risk, _reward, rr = _compute_risk_reward(df, "up", prior_close=99.0, spike_bars=2)
        # rr_ratio returns 0.0 when risk == min_risk per the function contract
        self.assertEqual(rr, 0.0)

    def test_rr_override_flips_to_short(self):
        df = trend_up_df(10)
        # Bull trend, but compute short-side R/R (bear-flip scenario)
        risk_long, _, _ = _compute_risk_reward(df, "up", 99.0, spike_bars=4)
        risk_short, _, _ = _compute_risk_reward(df, "up", 99.0, spike_bars=4,
                                                 rr_direction_override="down")
        # Different direction → different risk reading
        self.assertNotEqual(risk_long, risk_short)

    def test_returns_rounded_tuple(self):
        df = trend_up_df(10)
        result = _compute_risk_reward(df, "up", 99.0, spike_bars=4)
        self.assertEqual(len(result), 3)
        for v in result:
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
