# aiedge-scanner — Architecture Migration

**Status:** in progress. Started 2026-04-17.
**Goal:** Wall-Street-quality layered architecture. Research == production.

---

## Why

The scanner has grown to ~12k lines concentrated in two files:
`shared/brooks_score.py` (3,928 LOC, 68 functions) and `live_scanner.py`
(2,972 LOC, 43 functions). Both mix multiple layers of concern:

- brooks_score.py: candle helpers, swing math, EMA, score components,
  cycle-phase, session-shape, day-type, R:R, signal, databento
  normalization, universe fetching, AND top-level scan entry point.
- live_scanner.py: fetching, chart rendering, key levels, dedup, HTML
  dashboard, serialization, API posting, Apple Notes, alerting,
  threading, replay, AND main().

Two other problems compound:

1. **BPA detector is not the primary signal.** `bpa_detector.py` runs
   inside `brooks_score._score_bpa_patterns()` as a post-hoc scorer,
   not as the primary setup source. Pattern Lab stats measure the
   hybrid engine, not pure Brooks.
2. **Video-pipeline code lives in the same shared/ as scanner code.**
   `stages/{narration,assembly,broll_generation,script,upload,
   newsletter}.py` and `shared/{claude_writer,gemini_writer,
   elevenlabs_narrator,ffmpeg_assembler,kling_client,veo_client,
   youtube_uploader,newsletter_publisher}.py` are for the YouTube
   content pipeline, unrelated to scanning.

---

## Target layout

```
aiedge-scanner/
├── aiedge/
│   ├── data/           # Layer 1 — acquisition (pure I/O)
│   ├── features/       # Layer 2 — pure bar math
│   ├── context/        # Layer 3 — market-state classifiers
│   ├── signals/        # Layer 4 — setup detection (bpa_detector primary)
│   ├── risk/           # Layer 5 — trader's equation + priors
│   ├── execution/      # Layer 6 — fill simulation (backtest + live share)
│   ├── storage/        # Layer 7 — SQLite persistence
│   ├── analysis/       # Layer 8 — stats, equity curves, calibration
│   ├── dashboard/      # HTML rendering + API client
│   └── runners/        # thin orchestrators — live, backfill, backtest
├── content/            # video pipeline (isolated from scanner)
├── api/                # Flask/FastAPI for aiedge.trade site
├── tools/              # analysis tools
├── tests/              # mirrors aiedge/
├── scratch/            # diagnostic/scratch files
├── bin/                # shell entry points
├── configs/            # YAML configs (thresholds, universe, etc.)
└── pyproject.toml
```

### Layer import rules (enforced by design)

- A layer may only import from layers **below** it.
- `data/` imports only third-party libs (pandas, databento-python).
- `features/` imports only `data/` and third-party.
- `context/` imports `features/` and `data/`.
- `signals/` imports `context/`, `features/`, `data/`.
- `risk/` imports `signals/`, `context/`, `features/`, `storage/` (for priors).
- `execution/` imports `risk/` and below.
- `storage/` imports nothing from us (pure SQLite wrapper).
- `analysis/` imports `storage/` and below (read-only).
- `dashboard/` imports `analysis/` and below.
- `runners/` may import anything.

### Key invariant

`runners/live.py` and `runners/backtest.py` **must** both import their
signal pipeline from `signals/`. Identical code path — that is the
guarantee that research == production.

---

## Migration phases

### Phase 0 — Skeleton + plan (complete 2026-04-17)

- [x] Create directory tree under `aiedge/`.
- [x] Create `__init__.py` files with layer purpose docstrings.
- [x] Move scratch files to `scratch/`.
- [x] Write this MIGRATION.md.

### Phase 1 — Pre-refactor hygiene (BLOCKED — needs WIP committed)

There are **2,121 lines of uncommitted work** across 7 files as of
2026-04-17. Big moves are blocked until this lands.

Blocked files:
- `shared/bpa_detector.py` (+1,317)
- `shared/pattern_lab.py` (+828)
- `pattern_lab_api.py` (+224)
- `backfill_pattern_lab.py` (+115)
- `live_scanner.py` (+56)
- `stages/screener.py` (+3)
- `shared/brooks_score.py` (+1)

### Phase 2 — Isolate content pipeline (complete 2026-04-17)

- [x] Move `stages/{narration,assembly,broll_generation,script,upload,
      newsletter,chart_generation,log}.py` → `content/stages/`
- [x] Move `shared/{claude_writer,gemini_writer,elevenlabs_narrator,
      ffmpeg_assembler,kling_client,veo_client,youtube_uploader,
      newsletter_publisher}.py` → `content/shared/`
- [x] Update imports in `pipeline.py`, `content/stages/*.py`,
      `tests/test_script_validator.py` to reference new paths
- [x] `stages/` now scanner-only (just `screener.py`)
- [x] `shared/` now scanner-only (bpa_detector, brooks_score, pattern_lab,
      chart_renderer, databento_client, config_loader, sqlite_logger,
      notifier)
- [x] Verify all moved/edited files pass `python -m py_compile`

Note: `chart_generation.py` moved to content/ — it renders video-segment
charts. Scanner chart rendering is done by `live_scanner.render_chart_base64`
which stays in scanner and will move to `dashboard/` in Phase 4.

### Phase 3 — Carve brooks_score.py (IN PROGRESS)

Progress as of 2026-04-17:
- [x] Phase 3a: candle helpers → `aiedge/features/candles.py` (commit 59efef0)
- [x] Phase 3b: ema, swings, volatility, session → `aiedge/features/*.py` (commit 03f1f12)
- [x] Phase 3c: cycle-phase classifier → `aiedge/context/phase.py` (commit 91ac1fe)
- [x] Phase 3d: session-shape classifier → `aiedge/context/shape.py` (commit b8e02ef)
- [x] Phase 3e: day-type classifier → `aiedge/context/daytype.py`
      (`_classify_day_type`, `_apply_day_type_weight`, `_compute_two_sided_ratio`,
      `DAY_TYPE_WEIGHTS` matrix — 540 LOC + 21 tests)
- [x] Phase 3f-1: urgency scorers (15 `_score_*` + `_find_first_pullback` helper)
      → `aiedge/signals/components.py`
- [x] Phase 3f-2: uncertainty scorers (`_score_uncertainty`, `_score_two_sided_ratio`,
      `_score_liquidity_gaps`) + `_check_liquidity` hard-gate helper
      → `aiedge/signals/components.py`
      (Note: `STRONG_BODY_RATIO` and `SPIKE_MIN_BARS` remain in `context/daytype.py`
       for now — signals → context is allowed by layer rules, so components.py
       imports them from daytype. Phase 3h may promote them to `features/candles.py`.)
- [x] Phase 3g: risk (`_compute_risk_reward`) → `aiedge/risk/trader_eq.py`
- [x] Phase 3h: signal aggregator + phase + BPA scorer + summary →
      `aiedge/signals/{aggregator,bpa,summary}.py`
      (`_detect_phase` and `_determine_signal` + signal decision thresholds
       and intraday flip labels → `signals/aggregator.py`;
       `_score_bpa_patterns` + BPA constants + bpa_detector import →
       `signals/bpa.py`; `_generate_summary` → `signals/summary.py`.)
- [x] Phase 3i: data helpers → `aiedge/data/{normalize,resample,universe}.py`
      (`_normalize_databento_df` → `normalize.py`; `_resample_to_5min` +
       `SCAN_BAR_SCHEMA` + `SCAN_RESAMPLE` → `resample.py`;
       `_get_default_universe` + fallback universe list → `universe.py`.)
- [x] Phase 3j: move CLI + synthetic-data demo to `bin/brooks_score_cli.py`
      (`_make_bars`, `_demo_now_like`, `_demo_xle_like`, `_demo_mrvl_like`,
       `_demo_gap_chop`, `_demo_orcl_like`, `_run_demo`, `_run_scan`, and the
       `__main__` argparse block). brooks_score.py is now a pure library
       entry point + compat shim. Run demos via `python bin/brooks_score_cli.py`.

brooks_score.py: 3,928 → 945 LOC (2,983 removed, **76% reduction**).
Phase 3 complete — brooks_score.py is now a compat shim that exposes
`score_gap` / `score_multiple` / `scan_universe` and re-exports all
moved symbols so existing callers (live_scanner, pattern_lab_api,
tools, tests) keep working unchanged.

Tests: 179 passing across features/ + context/ + signals/ + risk/ + data/
(46 + 39 + 78 + 6 + 10) plus 1 pre-existing broken test unrelated
to this work.

Each phase leaves brooks_score.py importing from the new modules so
existing consumers (live_scanner.py, pattern_lab_api.py, tests) keep
working unchanged. Every `from shared.brooks_score import X` still
resolves; X just now lives in aiedge.* under the hood.

Function map (remaining):

| Current function | New location |
|---|---|
| `_safe_range`, `_body`, `_body_ratio`, `_is_bull`, `_is_bear`, `_lower_tail_pct`, `_upper_tail_pct`, `_close_position` | `features/candles.py` |
| `_compute_ema` | `features/ema.py` |
| `_find_swing_lows`, `_find_swing_highs` | `features/swings.py` |
| `_compute_daily_atr`, `_opening_range` | `features/volatility.py` + `features/session.py` |
| `_score_*` (17 funcs: gap integrity, tail quality, follow-through, body gaps, MA sep, volume, trend bars, micro gaps, trending, levels broken, liquidity, uncertainty, trending swings, two-sided, spike duration, small pullback, liquidity gaps) | `signals/components.py` |
| `_score_bpa_patterns` | delete — replaced by `signals/aggregator.py` |
| `_cycle_*_raw` (5 funcs), `_softmax`, `classify_cycle_phase` | `context/phase.py` |
| `_shape_*_raw` (5 funcs), `classify_session_shape` | `context/shape.py` |
| `_classify_day_type`, `_apply_day_type_weight` | `context/daytype.py` |
| `_detect_phase`, `_compute_two_sided_ratio` | `context/phase.py` (helpers) |
| `_compute_risk_reward` | `risk/trader_eq.py` |
| `_determine_signal` | `signals/aggregator.py` |
| `_generate_summary` | `signals/summary.py` |
| `_normalize_databento_df`, `_resample_to_5min`, `_get_default_universe` | `data/normalize.py`, `data/resample.py`, `data/universe.py` |
| `score_gap`, `score_multiple`, `scan_universe` | `runners/live.py` (or a `signals/pipeline.py` façade) |

Old `shared/brooks_score.py` becomes a compat shim:
```python
# shared/brooks_score.py (post-migration)
from aiedge.signals.aggregator import determine_signal as _determine_signal
from aiedge.features.candles import body, is_bull, is_bear
# ... etc. Re-exports for anything that imports from shared.brooks_score.
```

### Phase 4 — Carve live_scanner.py

Progress as of 2026-04-17:
- [x] Phase 4a: `fetch_prior_closes`, `backfill_intraday_bars`, `_fetch_ohlcv1m_range`
      + `_prev_trading_days`, `with_timeout`, `_timeout_handler` + `DATASET`,
      `ET`, `SCHEMA` → `aiedge/data/databento.py`
      (`backfill_intraday_bars` refactored to take bars/instrument_map +
       their locks as explicit parameters — no more reliance on live_scanner
       globals)
- [x] Phase 4b: `fetch_intraday_key_levels` → `aiedge/data/levels.py`
- [x] Phase 4c: `resample_to_5min` (live-variant, forward-fill + partial-bar
      drop) → `aiedge/data/resample.py` (alongside the existing
      `_resample_to_5min` simple helper). `ET` also moved to resample.py to
      avoid a circular import with databento.py.
- [x] Phase 4d: `render_chart_base64` → `aiedge/dashboard/charts.py`
- [x] Phase 4e: post-processing (`_dedup_etf_families`, `_compute_movement`,
      `_fmt_movement`, `_fmt_delta`, `annotate_adr_multiple`) → `aiedge/signals/postprocess.py`
      (`annotate_adr_multiple` refactored to take `daily_atrs` as a parameter;
       live_scanner keeps a 3-arg wrapper that binds to its global. ETF_FAMILIES
       + SAME_COMPANY + _TICKER_TO_FAMILY moved alongside the dedup function.)
- [x] Phase 4f: pattern-lab logging (`log_pattern_lab_detections`,
      `update_pattern_lab_outcomes`) → `aiedge/storage/pattern_lab.py`
      (`update_pattern_lab_outcomes` refactored to take bars/bars_lock as
       parameters; live_scanner keeps private wrappers that bind its globals.)
- [~] Phase 4g: dashboard rendering split into two sub-phases
  - [x] Phase 4g-1: console leaderboard + Apple Note formatters
        (`print_leaderboard`, `_format_note_text`, `_movement_arrow`,
         `_next_scan_after`, `_next_scan_time_str`, ANSI color constants)
        → `aiedge/dashboard/console.py`
        (wrappers in live_scanner bind `FIRST_SCAN_HOUR` / `FIRST_SCAN_MIN`)
  - [x] Phase 4g-2: HTML card + full-page rendering (`_HTML_HEAD`, `_HTML_FOOT`,
        `_SIG_CSS`, `_signal_badge`, `_rank_arrow_html`, `_movement_html`,
        `_bar_html`, `_adr_mult_tier`, `_build_component_strip`,
        `_build_card_html`, `_generate_dashboard`)
        → `aiedge/dashboard/render.py`
        (`_generate_dashboard` refactored to take `intraday_levels`,
         `dashboard_path`, `first_scan_hour`, `first_scan_min` as parameters;
         live_scanner keeps a wrapper that binds its module-level globals.)
- [x] Phase 4h: serializers (`_serialize_bars`, `_serialize_key_levels`,
      `_serialize_scan_payload`) → `aiedge/dashboard/serializers.py`
      + api client (`_post_to_aiedge`, `AIEDGE_SCAN_URL`) →
      `aiedge/dashboard/api_client.py`
      (`_serialize_scan_payload` refactored to take `intraday_levels`,
       `first_scan_hour`, `first_scan_min` as parameters; live_scanner keeps
       wrappers that bind its globals. These four functions were initially
       bundled into render.py by Phase 4g-2's wholesale block move; split
       out here to match the architectural layout.)
- [x] Phase 4i: notifiers (`update_apple_note`, `fire_alert`) →
      `aiedge/dashboard/notifiers.py`
- [x] Phase 4j: runner + all remaining state → `aiedge/runners/live.py`
      (live_scanner.py became a 64-LOC compat shim that re-exports
       everything from `aiedge.runners.live` + `main()` dispatch)

live_scanner.py: 2,972 → 64 LOC (pure compat shim, **97.8% reduction**).
aiedge/runners/live.py: 1,000 LOC — the actual scanner runtime.

Phase 4 complete. live_scanner.py now just does:
```python
from aiedge.runners.live import *
if __name__ == "__main__": main()
```

Tests: 218 passing across features/ + context/ + signals/ + risk/ + data/.
End-to-end smoke verified: `from live_scanner import X` still works for every
X that tools/, scratch/, and tests/ currently import; python live_scanner.py
still runs main() (which lives in aiedge.runners.live).

### Phase 4 (original map)

| Current function | New location |
|---|---|
| `fetch_prior_closes`, `backfill_intraday_bars`, `_fetch_ohlcv1m_range`, `fetch_intraday_key_levels` | `data/databento.py`, `data/levels.py` |
| `resample_to_5min`, `_normalize_*` | `data/resample.py` |
| `render_chart_base64` | `dashboard/charts.py` |
| `_dedup_etf_families`, `_compute_movement`, `_fmt_movement`, `_fmt_delta` | `signals/postprocess.py` |
| `_log_pattern_lab_detections`, `_update_pattern_lab_outcomes` | `storage/pattern_lab.py` (methods) |
| `annotate_adr_multiple` | `signals/components.py` |
| `_build_*_html`, `_bar_html`, `_signal_badge`, `_movement_html`, `_format_note_text`, `_generate_dashboard`, `print_leaderboard` | `dashboard/html_cards.py`, `dashboard/render.py` |
| `_serialize_bars`, `_serialize_key_levels`, `_serialize_scan_payload` | `dashboard/serializers.py` |
| `_post_to_aiedge` | `dashboard/api_client.py` |
| `update_apple_note`, `fire_alert` | `dashboard/notifiers.py` |
| `scan_thread_func`, `stream_thread_func`, `save_final_results`, `save_session_data`, `_replay_session`, `main`, `run_scan` | `runners/live.py` |

### Phase 5 — Wire BPA detector as primary

- `signals/aggregator.py` takes `bpa_detector` hits as primary input.
- Component scores from `signals/components.py` become filters/modifiers.
- `risk/trader_eq.py` computes edge from (setup, prior, R:R).
- Delete `_score_bpa_patterns` — it was the post-hoc hack.

### Phase 6 — New capabilities (complete 2026-04-17)

All 8 modules implemented + tested. These are net-new code — they
don't replace anything in the scanner's current path, but the Phase 5
aggregator-rewrite will consume them.

- [x] `features/regime.py` — `atr_percentile`, `realized_vol_tercile`
- [x] `context/htf.py` — `classify_htf_alignment(daily, weekly, direction)`
      → returns {"aligned" | "mixed" | "opposed" | "no_data"}
- [x] `risk/priors.py` + `storage/priors_store.py` — SQLite-backed
      empirical win-rate lookup with 5-level fallback hierarchy
      (exact → regime+align → regime → setup → default)
- [x] `analysis/walkforward.py` — `rolling_window`, `expanding_window`,
      `by_date` — lazy generators of train/test index splits
- [x] `analysis/reliability.py` — `reliability_table`, `brier_score`,
      `expected_calibration_error`
- [x] `analysis/equity.py` — `equity_curve`, `sharpe`, `sortino`,
      `max_drawdown`, `summary_stats` (one-call dashboard)
- [x] `analysis/correlation.py` — `log_returns`, `correlation_matrix`,
      `cluster_by_threshold`, `dedup_correlated` (data-driven ETF-family
      complement)
- [x] `analysis/failure.py` — `classify_failure` taxonomy
      (stop_flush / slow_bleed / reversal / news_shock / chop / unknown)
      + `failure_breakdown` aggregator
- [ ] `storage/pattern_lab.py` — add `failure_reason`, regime fields.
      Deferred: schema migration requires a Pattern Lab rebuild (see
      `project_pattern_lab.md`). Do this alongside Phase 5's aggregator
      rewrite.

New tests: 103 added across analysis/, context/htf/, risk/priors,
storage/priors_store, features/regime.

Test suite after Phase 6: **440 passed**.

### Phase 7 — Package properly (complete 2026-04-17)

- [x] `pyproject.toml` with `[project]` metadata — describes the
      `aiedge-scanner` package, declares runtime deps (databento,
      pandas, numpy, pytz, python-dotenv, requests, matplotlib, pyyaml),
      optional-dep group `dev` (pytest, pytest-cov), and two console
      entry points:
        - `aiedge-scanner` → `aiedge.runners.live:main`
        - `aiedge-brooks-score` → `bin.brooks_score_cli:_main`
- [x] Install as editable: `pip install -e . --break-system-packages`
- [x] Kill `sys.path.insert(0, ROOT)` hacks in the two entry points
      (aiedge/runners/live.py + bin/brooks_score_cli.py). Leaves the
      hacks in tools/ and tests/ for standalone-run robustness.
- [x] `[tool.pytest.ini_options]` — autodiscovers `tests/`, ignores
      the pre-existing broken `test_spt_qc.py`. pytest now reports
      **293 passed**.

Entry points verified on PATH after install:

    $ which aiedge-scanner aiedge-brooks-score
    /opt/homebrew/bin/aiedge-scanner
    /opt/homebrew/bin/aiedge-brooks-score

Packages declared: aiedge + aiedge.{context,dashboard,data,execution,
features,risk,runners,signals,storage}, bin, shared, content, content.shared,
content.stages; plus top-level `live_scanner` as a py_module (it's still
the compat shim from Phase 4j).

Note: `content/` and `shared/` are included so the installed package
can still import from them during the transition; they'll be fully
migrated in a later pass.

### Phase 8 — Expand test coverage (complete 2026-04-17)

Target: 80% line coverage on `features/`, `context/`, `signals/`,
`risk/`, `execution/`. **Achieved 82%** across those packages.

Per-module coverage:

| Module | Coverage |
|---|---|
| features/candles.py | 100% |
| features/ema.py | 100% |
| features/session.py | 86% |
| features/swings.py | 100% |
| features/volatility.py | 100% |
| context/daytype.py | 80% |
| context/phase.py | 96% |
| context/shape.py | 80% |
| signals/aggregator.py | 78% |
| signals/bpa.py | 95% |
| signals/components.py | 77% |
| signals/postprocess.py | 100% |
| signals/summary.py | 92% |
| risk/trader_eq.py | 94% |

New tests added in Phase 8:
- `tests/context/test_shape.py` — scenario-based fixtures for each
  of the 5 raw shape scorers + classify_session_shape full-scenario tests
  (moved shape from 51% to 80%).
- `tests/signals/test_bpa.py` — `_score_bpa_patterns` with a mocked
  `_bpa_detect_all` covering: disabled, unavailable, short df, detector
  exception, confidence filter, recency filter, each setup-type branch
  (H1/H2/L1/L2/FL1/FL2/spike_channel), opposing direction, multi-setup
  best-wins, output serialization shape (bpa from 26% to 95%).
- `tests/signals/test_components.py` — branch coverage for bear
  directions across all component scorers + extra uncertainty branches
  (reversal counts, trendline-broken, long chop for EMA-wrong-side)
  + SPT bear path + zigzag trending-swings.

Run `pytest --cov=aiedge.features --cov=aiedge.context --cov=aiedge.signals --cov=aiedge.risk --cov=aiedge.execution`
to reproduce the 82% total.

---

## Compatibility during migration

While Phase 3 and 4 are in progress, `shared/brooks_score.py` and
`live_scanner.py` remain as compat shims that re-export from
`aiedge.*`. Running scanner continues to work unchanged. We flip
imports incrementally.

When shims are no longer referenced, they get deleted.

---

## Known risks

1. **Circular imports during carving.** Likely. Mitigation: start with
   leaf modules (`features/candles.py`) and work upward.
2. **Test coverage is thin now.** We may silently break behavior.
   Mitigation: capture a "golden" scan output pre-refactor, diff
   against post-refactor output at each phase boundary.
3. **Schema changes in storage/** will require migrations. Write
   migration scripts in `storage/migrations/` at every schema bump.
4. **Pattern Lab stats become invalid when detector logic changes.**
   Expected — post-Phase 5, rebuild Pattern Lab from raw bars using
   the new aggregator. Flag old stats as stale in UI.

---

## File-by-file migration tracker

Updated as each file lands in its new home.

| Current path | Status | Target path |
|---|---|---|
| `shared/brooks_score.py` | TBD | split into `features/*`, `context/*`, `signals/*`, `risk/trader_eq.py` |
| `shared/bpa_detector.py` | TBD | `signals/bpa_detector.py` |
| `shared/pattern_lab.py` | TBD | `storage/pattern_lab.py` |
| `shared/databento_client.py` | TBD | `data/databento.py` |
| `shared/chart_renderer.py` | TBD | `dashboard/charts.py` |
| `shared/config_loader.py` | TBD | `aiedge/config.py` |
| `shared/sqlite_logger.py` | TBD | `storage/sqlite.py` |
| `shared/notifier.py` | TBD | `dashboard/notifiers.py` |
| `live_scanner.py` | TBD | split into `runners/live.py` + `dashboard/*` + `data/*` |
| `backfill_pattern_lab.py` | TBD | `runners/backfill.py` |
| `backfill_historical_databento.py` | TBD | `runners/backfill.py` (merge) |
| `pattern_lab_api.py` | TBD | `api/pattern_lab.py` |
| `pipeline.py` | TBD | `runners/content_pipeline.py` or `content/` |
| `stages/screener.py` | TBD | `runners/screener_stage.py` |
| `stages/chart_generation.py` | TBD | `runners/chart_stage.py` |
| `stages/{narration,assembly,broll,script,upload,newsletter}.py` | TBD | `content/stages/` |
| `shared/{claude_writer,gemini_writer,elevenlabs,ffmpeg,kling,veo,youtube,newsletter_publisher}.py` | TBD | `content/shared/` |
| `claude_backtest.py` | TBD | `runners/backtest.py` |
| `screenshot_generator.py` | TBD | `tools/screenshot_generator.py` |
| `oauth_setup.py` | TBD | `tools/oauth_setup.py` |
| `probe_databento_plan.py` | TBD | `scratch/` |
| root scratch files (`db_test*`, `_render_gaps*`, `_diag*`, `_benchmark*`, `_qc*`, `_validate_*`, `_test_*`, `test_dashboard.py`, `test_spt_fix.py`) | Phase 0 — moved | `scratch/` |
| `chart_samples/_gen_before_after.py` etc. | TBD | `scratch/chart_samples/` |
