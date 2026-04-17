# aiedge-scanner — Post-migration roadmap

**For future Claude sessions.** Start here, not in the user's chat.

The big 8-phase architecture migration (2026-04-17) is done — see
`MIGRATION.md` for what landed and why. This file is the forward plan:
what to work on next, in what order, and with what guardrails.

---

## Before every session

1. **Read the last 5 commits:** `git log --oneline -5`
2. **Confirm the test suite is green:** `python -m pytest -q`
   (should be **451 passed** as of 2026-04-17)
3. **Skim this file + MIGRATION.md.** The last session's context is there.
4. **Check the vault.** `~/code/aiedge/vault/Meta/how-to-use-this-vault.md` + any
   notes touched in the last 7 days. Will keeps trading lessons there — if
   they contradict this file, trust the vault.
5. **Ask before starting a destructive or production-affecting change.**
   Refactor-style moves are fine to do autonomously; anything that changes
   scanner OUTPUT (e.g. flipping `BPA_PRIMARY_ENABLED`) needs Will's OK.

---

## Track A — "Make it pay" (highest ROI)

Flip the scanner from post-hoc BPA overlay to BPA-as-primary. The
machinery is built (`aiedge/signals/bpa_primary.py`, Phase 5), but
it's gated on empirical prior data that hasn't been loaded yet.

### Session A1: backfill the priors store

**Entry:** `aiedge/storage/priors_store.py` exists. `priors_store` SQLite
file does not exist yet (in-memory only).

**Do:**
1. Pick a persistent path: `~/code/aiedge/scanner/db/priors.db`.
2. Write `scripts/backfill_priors.py` — walks the Pattern Lab SQLite
   (~2,181 detections with outcomes), extracts
   `(setup_type, regime, htf_alignment, day_type, won)` per row, calls
   `store.record_outcome(...)`. Regime needs `features.regime.realized_vol_tercile`
   against each detection's ticker's daily closes. Alignment needs
   `context.htf.classify_htf_alignment`.
3. **Gotcha:** Pattern Lab doesn't have daily close history stored per
   detection. You'll need to pull it from Databento Historical on the
   fly, keyed by detection_date. Cache the per-ticker-per-day lookup so
   you don't refetch the same ticker 40 times.
4. Print a summary: how many strata have ≥30 samples (the threshold the
   aggregator uses).

**Exit:**
- `priors.db` exists with ≥100 filled strata.
- `python -c "from aiedge.storage.priors_store import PriorsStore; s = PriorsStore('db/priors.db'); print(s.all_by_setup('H2')[:3])"` shows real win/loss counts.
- Don't flip `BPA_PRIMARY_ENABLED` yet.

### Session A2: A/B the two aggregators on yesterday's scan

**Entry:** Session A1 done. `priors.db` populated.

**Do:**
1. Write `scripts/ab_aggregators.py` — for the most recent saved scan
   session (look in `logs/live_scanner/*.pkl`):
   - Replay the bars through the OLD path: `_determine_signal(...)` per
     ticker, capture the top-20 ranked list.
   - Replay the same bars through `bpa_primary.evaluate_setups(...)`,
     capture the top-20 ranked list.
   - Diff them: how many tickers overlap in top-20? How many NEW names
     did the primary path surface? What's the average edge?
2. Make a markdown report at `scratch/ab_report_YYYY-MM-DD.md`.
3. **Show Will the diff before flipping anything.** Get his read on
   whether the new list looks better.

**Exit:** Decision made with Will about whether to flip the flag.

### Session A3: flip the flag + ship the schema upgrade

**Entry:** Will approves the flip.

**Do:**
1. Add `failure_reason`, `regime`, `htf_alignment` columns to the Pattern
   Lab SQLite schema. Migration script in
   `aiedge/storage/migrations/001_add_failure_regime.py`. See
   `analysis/failure.classify_failure` + `features/regime.realized_vol_tercile`
   for the values to backfill.
2. Modify `aiedge/storage/pattern_lab.py::log_pattern_lab_detections`
   to write the new columns on new inserts.
3. Set `BPA_PRIMARY_ENABLED = True` in `bpa_primary.py`.
4. Update `aiedge/runners/live.py::run_scan` to:
   - Call `bpa_primary.evaluate_setups(...)` instead of (or alongside)
     the old aggregator path.
   - Feed the priors_store into `AggregatorInputs`.
5. Watch the first live session carefully. Stop the scanner if urgency/
   uncertainty distributions look wildly off.
6. Mark `_score_bpa_patterns` as deprecated in `aiedge/signals/bpa.py`
   but do NOT delete yet — give it a week of parallel running.

**Exit:** Live scanner running with the new aggregator. Pattern Lab
tagging every detection with regime + failure_reason.

---

## Track B — "Make it readable" (Phase 6 consumption)

Four small sessions. Pick any order. Each consumes one of the Phase 6
capability modules that's currently unused. All four are 2-4 hour
sessions.

### Session B1: correlation dedup on the live dashboard

**Use:** `aiedge/analysis/correlation.py`

**Do:**
- In `aiedge/runners/live.py::run_scan`, after the existing
  `_dedup_etf_families` call, add a data-driven correlation dedup:
  - Build a `correlation_matrix` from the last 30 days of daily closes
    across all surfaced tickers (cache it — it doesn't change mid-session)
  - Call `dedup_correlated(candidates, corr, threshold=0.85)` after the
    static family dedup
- Add a "+N correlated" chip on the surviving card (template in
  `aiedge/dashboard/render.py::_build_card_html` — the family_siblings
  block already does this, just extend).
- Test: on a tech-heavy day, verify SPY/QQQ/SMH collapse to one card.

**Gotcha:** Don't blow up when corr matrix has NaN (thin overlap between
two names). The `min_periods` arg on `correlation.correlation_matrix`
handles it — but the dedup code must treat NaN as "not correlated".

### Session B2: equity stats on /trades page

**Use:** `aiedge/analysis/equity.py::summary_stats`

**Do:**
- In `~/code/aiedge/site/`, wherever `/trades/page.tsx` is, add a
  summary strip at the top showing total PnL, win rate, Sharpe, Sortino,
  max drawdown, expectancy. All available from `summary_stats(pnls)`.
- Source: the trades list the scanner already syncs via
  `scripts/sync_trades.py`.
- Make it dead simple — no charts yet, just numbers. Charts can come
  in a follow-up.

### Session B3: calibration diagram on /review

**Use:** `aiedge/analysis/reliability.py`

**Do:**
- Read Pattern Lab detections that have both a predicted win-probability
  (the prior lookup's `p_win`) and a realized outcome.
- Feed them to `reliability_table(predicted, outcomes, n_bins=10)`.
- Render a stacked bar chart on aiedge.trade/review showing:
  - x-axis: bin midpoint (predicted probability)
  - y-axis: empirical hit rate
  - diagonal reference line
  - bin count annotation above each bar
- Also compute `brier_score` + `expected_calibration_error` and show them.

**Why it matters:** This is the single picture that tells you whether
the scanner's "70% confidence" actually means 70%. If bars sag below
the diagonal, the scanner is overconfident — priors need downweighting.

### Session B4: HTF alignment chip on every card

**Use:** `aiedge/context/htf.py::classify_htf_alignment`

**Do:**
- In `aiedge/runners/live.py::run_scan`, for each scored ticker,
  fetch/cache daily + weekly closes and call `classify_htf_alignment`
  against the setup direction.
- Add an `htf_alignment` field to the result dict.
- In `aiedge/dashboard/render.py::_build_card_html`, add a small chip:
  - `aligned` → green chip "↑D ↑W"
  - `mixed`   → yellow chip "↕"
  - `opposed` → red chip "✗"
  - `no_data` → hide
- Daily/weekly closes should come from `aiedge/data/databento.py` — add
  a `fetch_daily_closes(tickers, days=60)` helper if it doesn't exist.

---

## Track C — "Finish the cleanup" (lower priority)

The migration left some tech debt. Not urgent, but each item makes
future sessions cheaper.

### Session C1: shrink `shared/brooks_score.py` to a thin re-export

**Current:** 948 LOC. It still holds `score_gap` (~300 LOC) and the
`score_multiple` + `scan_universe` functions (~250 LOC). Plus a 150+
line list of re-imports.

**Do:**
- Move `score_gap` → `aiedge/signals/pipeline.py` (the "signals/pipeline.py
  façade" mentioned in MIGRATION.md).
- Move `score_multiple`, `scan_universe` → `aiedge/runners/live.py` or
  a new `aiedge/runners/batch.py`.
- Reduce `shared/brooks_score.py` to <100 LOC of re-exports. Add a
  deprecation comment pointing at the new locations.
- Run the full test suite to confirm no regressions.

### Session C2: migrate remaining `shared/*` modules

**Files:** `shared/bpa_detector.py`, `shared/pattern_lab.py`,
`shared/chart_renderer.py`, `shared/databento_client.py`,
`shared/config_loader.py`, `shared/sqlite_logger.py`, `shared/notifier.py`.

**Do:** One commit per file, moving each to its proper home per
MIGRATION.md's "file-by-file migration tracker":

- `bpa_detector.py` → `aiedge/signals/bpa_detector.py`
- `pattern_lab.py` → `aiedge/storage/pattern_lab_core.py` (rename to
  avoid colliding with the Phase 4f `aiedge/storage/pattern_lab.py`)
- `chart_renderer.py` → `aiedge/dashboard/chart_renderer.py`
- `databento_client.py` → `aiedge/data/databento_client.py`
- `config_loader.py` → `aiedge/config.py`
- `sqlite_logger.py` → `aiedge/storage/sqlite.py`
- `notifier.py` → `aiedge/dashboard/notifier_base.py`

Leave a compat shim in `shared/` for each one (import * from new
location + re-export). DO NOT delete the shims yet — `scratch/`, `tools/`,
and external notebooks import from `shared/`.

### Session C3: kill dead code

**Look for:**
- `TREND_BAR_PCT = 0.70` and `TREND_MAX_PULLBACK = 0.25` in
  `shared/brooks_score.py` — defined, never used
- `MA_WRONG_SIDE_BARS` in `aiedge/signals/components.py` — defined,
  only used inline and could be dropped
- `tests/test_spt_qc.py` — pre-existing broken test with a hardcoded
  `/sessions/stoic-sleepy-gates/...` path. Either fix it or delete it.
  Flag is in `pyproject.toml` `pytest` config.

---

## Track D — "Keep the calibration loop moving"

The vault says 4/687 figures are done in `aiedge-self-eval`. That's the
human-labor bottleneck that no refactor can fix — only Will doing the
work (or delegating to agents) can move it.

**Useful now that the code is clean:**
- Sub-agents can run `/selfeval` on batches of 20-50 figures without
  drowning in code-archaeology context usage.
- Calibration lessons in `~/code/aiedge/self-eval/lessons.md` are loaded
  at every session start via `/session-start`. Keep that file under 200
  lines — consolidate lessons that say similar things.
- The `aiedge-validation` loop is a separate validation of the Brooks
  references themselves (is the reference page correct?). Loop #1 done
  with 3 edits — Loop #2+ is pending Will's direction.

---

## Gotchas the migration taught us

Things future Claude sessions should know so they don't waste time
re-learning:

### Don't reach into `live_scanner` globals

The old code had functions calling `bars`, `instrument_map`, `daily_atrs`,
`intraday_levels` as module globals. Every carved-out function got
refactored to take those as explicit parameters, with `live_scanner.py`
keeping thin wrappers that bind the globals.

**Rule:** A new function in `aiedge/` should never name-resolve anything
in `live_scanner.py`. Always take state as parameters.

### `ET` timezone circular-import hazard

`aiedge/data/databento.py` and `aiedge/data/resample.py` both need
`ET = pytz.timezone("America/New_York")`. The canonical home is
`aiedge/data/resample.py` (databento imports it from there). If you
see a new module needing ET, import from `aiedge.data.resample` or
`aiedge.data.databento` — don't define a third copy.

### AST-based function deletion

When carving out N functions from a file, the quick way is:

```python
import ast
with open(path) as f: src = f.read(); lines = src.split('\n')
tree = ast.parse(src)
ranges = [(n.lineno, n.end_lineno, n.name) for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name in TO_REMOVE]
for start, end, name in sorted(ranges, reverse=True):
    del lines[start-1:end]
```

This is how `shared/brooks_score.py` + `live_scanner.py` both got
carved. **Gotcha:** AST doesn't see module-level constants/blocks,
only `FunctionDef` / `ClassDef`. Delete string constants (like
`_HTML_HEAD`) manually.

### pytest is the right test runner

`python -m unittest discover` works per-directory but pytest's auto-
discovery is cleaner. `pyproject.toml` has `[tool.pytest.ini_options]`
configured to ignore `tests/test_spt_qc.py` (the broken one).

**To run full suite:** `python -m pytest -q` (under 2 seconds).
**To check coverage:** `python -m pytest --cov=aiedge --cov-report=term`.
**Coverage target per MIGRATION.md:** 80% on features/context/signals/risk/
execution. Currently at 82%.

### `pip install -e .` is assumed

Entry points (`aiedge-scanner`, `aiedge-brooks-score`) live in
`pyproject.toml`. The repo is installed editable globally
(use `--break-system-packages` on macOS w/ Homebrew Python).

Running `python live_scanner.py` still works — it's a shim that
dispatches to `aiedge.runners.live:main`.

### Content pipeline vs scanner

`content/` is the YouTube video pipeline — ignore it when working on
scanner features. It shares `pyproject.toml` only because we want both
installable as one package, but the two don't share logic.

---

## What NOT to do yet

- **No ML layer.** Tempting given the priors store + walkforward module
  are right there, but the Brooks read needs to be honest first.
  Calibration loop (Track D) is the prerequisite.
- **No mass multi-asset expansion** (forex/futures) without Will's
  explicit direction. The architecture supports it (swap
  `aiedge/data/databento.py` for a new source), but priorities come
  from Will not from the codebase.
- **Don't flip `BPA_PRIMARY_ENABLED = True`** until Session A2 is done
  and Will approves. See Track A.
- **Don't delete `_score_bpa_patterns`** until Track A is fully done
  and the new aggregator has been live for a week without regressions.
- **Don't refactor working code "just because"**. The migration is
  complete. From here, each change should either (a) add trading value
  or (b) be explicitly requested cleanup.

---

## One-line session starter

Copy-paste this at the top of the next chat to give Claude the right
context:

> I'm working on aiedge-scanner. Read `ROADMAP.md` and `MIGRATION.md`
> in `~/code/aiedge/scanner/` to catch up, run `pytest -q` to confirm
> green, then let's pick a session from one of the tracks.
