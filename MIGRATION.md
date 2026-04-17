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
- [ ] Phase 3i: data helpers (`_normalize_databento_df`, `_resample_to_5min`,
      `_get_default_universe`) → `aiedge/data/{normalize,resample,universe}.py`
- [ ] Phase 3j: misc helpers (`_detect_phase`, `_opening_range` helper callers)

brooks_score.py: 3,928 → 1,383 LOC (2,545 removed so far, 64.8%).
Tests: 169 passing in features/ + context/ + signals/ + risk/
(46 features + 39 context + 78 signals + 6 risk) plus 1 pre-existing
broken test unrelated to this work.

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

Map:

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

### Phase 6 — New capabilities

Added in service of the 7-step plan from the previous session:

- `features/regime.py` — vol regime (ATR percentile, realized vol tercile)
- `context/htf.py` — daily/weekly alignment check
- `risk/priors.py` + `storage/priors_store.py` — empirical probability
- `analysis/walkforward.py` — rolling train/test split
- `analysis/reliability.py` — calibration diagram
- `analysis/equity.py` — equity curves, Sharpe, DD, Sortino
- `analysis/correlation.py` — cluster correlated instruments
- `analysis/failure.py` — root-cause taxonomy
- `storage/pattern_lab.py` — add `failure_reason`, regime fields

### Phase 7 — Package properly

- `pyproject.toml` with `[project]` metadata
- Install as editable: `pip install -e .`
- Kill `sys.path` hacks in entry points
- Same package for scanner and web API

### Phase 8 — Expand test coverage

Target: 80% line coverage on `features/`, `context/`, `signals/`,
`risk/`, `execution/`. These are pure functions — easy to test.

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
