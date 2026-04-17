"""Brooks Price Action Gap Scorer — compat shim (ROADMAP C1).

Public API:
  - score_gap          → aiedge.signals.pipeline
  - score_multiple     → aiedge.runners.batch
  - scan_universe      → aiedge.runners.batch

All helpers/constants live under `aiedge/`; this module re-exports them
so existing callers keep working. CLI: bin/brooks_score_cli.py.

TODO C2: port tools/scratch callers off the private-helper re-exports
(`_body`, `_score_*`, `_detect_*`, etc.) so this shim can be dropped.
"""

# Top-level entry points (ROADMAP C1 moves).
from aiedge.signals.pipeline import score_gap  # noqa: F401
from aiedge.runners.batch import score_multiple, scan_universe  # noqa: F401

# Features (candles, EMA, swings, volatility, session).
from aiedge.features.candles import (  # noqa: F401
    DOJI_BODY_RATIO, MIN_RANGE, _body, _body_ratio, _close_position,
    _is_bear, _is_bull, _lower_tail_pct, _safe_range, _upper_tail_pct,
)
from aiedge.features.ema import EMA_PERIOD, _compute_ema  # noqa: F401
from aiedge.features.session import OPENING_RANGE_BARS, _opening_range  # noqa: F401
from aiedge.features.swings import _find_swing_highs, _find_swing_lows  # noqa: F401
from aiedge.features.volatility import _compute_daily_atr  # noqa: F401

# Context (day-type, cycle-phase, session-shape).
from aiedge.context.daytype import (  # noqa: F401
    CHOP_RATIO_THRESHOLD, DAY_TYPE_WEIGHTS, OR_TRADING_RANGE, OR_TRENDING_TR_HIGH,
    OR_TRENDING_TR_LOW, SPIKE_MIN_BARS, STRONG_BODY_RATIO, WARMUP_BARS,
    _apply_day_type_weight, _classify_day_type, _compute_two_sided_ratio,
)
from aiedge.context.phase import (  # noqa: F401
    CYCLE_PHASE_CLASSIFIER_ENABLED, CYCLE_PHASE_LOOKBACK_BARS, CYCLE_PHASE_SOFTMAX_TEMP,
    CYCLE_PHASES, _cycle_bear_channel_raw, _cycle_bear_spike_raw, _cycle_bull_channel_raw,
    _cycle_bull_spike_raw, _cycle_trading_range_raw, _softmax, classify_cycle_phase,
)
from aiedge.context.shape import (  # noqa: F401
    SESSION_SHAPE_CLASSIFIER_ENABLED, SESSION_SHAPE_SOFTMAX_TEMP, SESSION_SHAPE_WARMUP_MINUTES,
    SESSION_SHAPES, _shape_opening_reversal_raw, _shape_spike_and_channel_raw,
    _shape_trend_from_open_raw, _shape_trend_resumption_raw, _shape_trend_reversal_raw,
    classify_session_shape,
)

# Data + risk.
from aiedge.data.normalize import _normalize_databento_df  # noqa: F401
from aiedge.data.resample import SCAN_BAR_SCHEMA, SCAN_RESAMPLE, _resample_to_5min  # noqa: F401
from aiedge.data.universe import _get_default_universe  # noqa: F401
from aiedge.risk.trader_eq import _compute_risk_reward  # noqa: F401

# Signal layer (aggregator, BPA overlay, summary, components).
from aiedge.signals.aggregator import (  # noqa: F401
    FAILED_GAP_MIN_FRAC_ADR, SIGNAL_BUY_INTRADAY, SIGNAL_SELL_INTRADAY, UNCERTAINTY_HIGH,
    UNCERTAINTY_LOW, UNCERTAINTY_MED, UNCERTAINTY_TRAP, URGENCY_HIGH,
    _detect_phase, _determine_signal,
)
from aiedge.signals.bpa import (  # noqa: F401
    BPA_COUNTER_TYPES, BPA_INTEGRATION_ENABLED, BPA_LONG_SETUP_TYPES, BPA_MIN_CONFIDENCE,
    BPA_MIN_DF_LEN, BPA_RECENCY_BARS, BPA_SHORT_SETUP_TYPES, _score_bpa_patterns,
)
from aiedge.signals.summary import _generate_summary  # noqa: F401
from aiedge.signals.components import (  # noqa: F401
    BARS_STUCK_THRESHOLD, BEAR_SPIKE_RATIO, BODY_GAP_BONUS, BODY_OVERLAP_HIGH, CLOSE_TOP_PCT,
    COLOR_ALT_HIGH, DEEP_PULLBACK_PCT, DOJI_RATIO_HIGH, FAILED_SETUP_BONUS,
    GAP_INTEGRITY_POST_FILL_EVAL, GAP_PARTIAL_FILL_PCT, GAP_RECOVERY_PARTIAL_BULL,
    GAP_RECOVERY_PARTIAL_PCT, GAP_RECOVERY_STRONG_BULL, GAP_RECOVERY_STRONG_PCT,
    LIQUIDITY_GAP_PCT, LIQUIDITY_GAPS_HIGH, LIQUIDITY_GAPS_LOW, LIQUIDITY_GAPS_MODERATE,
    LIQUIDITY_MIN_DOLLAR_VOL, LIQUIDITY_SKIP_BARS, MA_GAP_BARS_MODERATE, MA_GAP_BARS_STRONG,
    MAX_SPIKE_BARS_SCORED, MICRO_GAP_BONUS, MIDPOINT_TOLERANCE, MODERATE_PULLBACK_PCT,
    NO_PULLBACK_BAR_THRESHOLD, POST_FILL_EVAL_BARS, REVERSAL_HIGH, REVERSAL_MODERATE,
    SHALLOW_PULLBACK_PCT, SPIKE_DURATION_MODERATE, SPIKE_DURATION_STRONG, SPIKE_RETRACE_LIMIT,
    SPT_DEPTH_MODERATE, SPT_DEPTH_SHALLOW, SPT_LOOKBACK_BARS, SPT_TREND_BODY_RATIO,
    STRONG_TREND_WINDOW, TAIL_BAD_PCT, TAIL_CLEAN_PCT, TAIL_CLEAN_RATIO, TAIL_MAX_PCT,
    TIGHT_RANGE_BARS_MODERATE, TIGHT_RANGE_BARS_STRONG, TIGHT_RANGE_PCT,
    TRENDING_SWINGS_MODERATE, TRENDING_SWINGS_STRONG, TWO_SIDED_HIGH, TWO_SIDED_MODERATE,
    TWO_SIDED_VERY_HIGH, UNCERTAINTY_ANALYSIS_WINDOW, UNCERTAINTY_RAW_MAX, URGENCY_RAW_MAX,
    VOLUME_SPIKE_EXTREME, VOLUME_SPIKE_STRONG, _check_liquidity, _find_first_pullback,
    _score_body_gaps, _score_failed_counter_setups, _score_follow_through, _score_gap_integrity,
    _score_levels_broken, _score_liquidity_gaps, _score_ma_separation, _score_majority_trend_bars,
    _score_micro_gaps, _score_small_pullback_trend, _score_spike_duration, _score_spike_quality,
    _score_tail_quality, _score_trending_everything, _score_trending_swings,
    _score_two_sided_ratio, _score_uncertainty, _score_volume_confirmation,
)

# Leftover constants kept for back-compat; not referenced elsewhere in repo.
OR_TREND_FROM_OPEN = 0.25
TRADING_RANGE_OVERLAP_BARS = 10
