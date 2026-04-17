#!/usr/bin/env python3
"""live_scanner.py — compat shim for `aiedge.runners.live`.

The actual scanner (state, threads, CLI entry point, and every helper)
now lives in `aiedge.runners.live`. This file preserves the legacy
`python live_scanner.py` invocation and the `from live_scanner import X`
API so existing scripts in `tools/`, `scratch/`, and `tests/` keep
working unchanged.

If you're modifying scanner logic, edit `aiedge/runners/live.py` —
NOT this file.
"""

from aiedge.runners.live import *  # noqa: F401,F403  (re-export the public surface)
from aiedge.runners.live import (  # noqa: F401  (explicit re-export of private names)
    _BOLD,
    _COLORS,
    _DOWN,
    _HTML_FOOT,
    _HTML_HEAD,
    _RST,
    _SIG_CSS,
    _TICKER_TO_FAMILY,
    _UP,
    _W,
    _adr_mult_tier,
    _backfill_intraday_bars_pure,
    _bar_html,
    _build_card_html,
    _build_component_strip,
    _compute_movement,
    _dedup_etf_families,
    _fetch_ohlcv1m_range,
    _fmt_delta,
    _fmt_movement,
    _format_note_text,
    _format_note_text_pure,
    _generate_dashboard,
    _generate_dashboard_pure,
    _log_pattern_lab_detections,
    _movement_arrow,
    _movement_html,
    _next_scan_after,
    _next_scan_after_pure,
    _next_scan_time_str,
    _next_scan_time_str_pure,
    _post_to_aiedge,
    _post_to_aiedge_pure,
    _prev_trading_days,
    _print_leaderboard_pure,
    _rank_arrow_html,
    _replay_session,
    _serialize_bars,
    _serialize_key_levels,
    _serialize_scan_payload,
    _serialize_scan_payload_pure,
    _signal_badge,
    _timeout_handler,
    _update_pattern_lab_outcomes,
    main,
)

if __name__ == "__main__":
    main()
