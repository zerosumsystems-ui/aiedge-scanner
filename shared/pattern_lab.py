"""
Pattern Lab — BPA setup detection database.

Records every detected Brooks Price Action setup from the live scanner
and tracks outcomes over subsequent bars to build statistical evidence.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

DB_PATH = get_project_root() / "db" / "pattern_lab.sqlite"


# ── Sentinels ─────────────────────────────────────────────────────────────────

class _LiveOnly:
    """Sentinel passed as run_id=LIVE_ONLY to filter queries to live detections
    (where run_id IS NULL). Distinct from run_id=None which means 'no filter'."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "LIVE_ONLY"


LIVE_ONLY = _LiveOnly()


# ── Signal gate ───────────────────────────────────────────────────────────────
#
# The scanner emits a `signal` tag for every bar it scores. BPA setups fired
# while the scanner itself said "don't trade" (AVOID, FOG, WAIT, PASS) are not
# real trades — logging them to Pattern Lab pollutes win-rate stats.
#
# Empirically (2026-04-16): of 68,510 live detections, 52,774 (77%) had a
# veto signal. After gating, H2 bear_channel win rate jumps from 13.5% to
# something more representative.
#
# Unknown signal strings are permitted (fail-open) so that future scanner
# additions aren't silently dropped — only *known-bad* signals are blocked.

VETO_SIGNALS: frozenset[str] = frozenset({"AVOID", "FOG", "WAIT", "PASS"})

LONG_ENDORSING_SIGNALS: frozenset[str] = frozenset({
    "BUY", "BUY_PULLBACK", "BUY_PULLBACK_INTRADAY",
})
SHORT_ENDORSING_SIGNALS: frozenset[str] = frozenset({
    "SELL", "SELL_PULLBACK", "SELL_PULLBACK_INTRADAY",
})
ENDORSING_SIGNALS: frozenset[str] = LONG_ENDORSING_SIGNALS | SHORT_ENDORSING_SIGNALS

# Counter-trend phases — fire direction-mismatched setups here and Brooks
# would call it "buying a pullback in a bear" (or vice versa): 80%-fail.
BEAR_PHASES: frozenset[str] = frozenset({"bear_channel", "bear_spike"})
BULL_PHASES: frozenset[str] = frozenset({"bull_channel", "bull_spike"})

# Setups that are bar-count pullback entries (Brooks chapters on bar counting).
# In trading ranges these fight the limit orders — Brooks #54 says scalp with
# limits, don't take stop entries. Block these in trading_range specifically.
# FL1/FL2 (final flag reversals), failed_bo, and spike_channel are structural
# and can validly fire in a trading range, so they're not gated here.
BAR_COUNT_SETUPS: frozenset[str] = frozenset({"H1", "H2", "L1", "L2"})


def is_vetoed_signal(signal: Optional[str]) -> bool:
    """True if the scanner signal is an explicit veto (AVOID/FOG/WAIT/PASS)."""
    return signal is not None and signal in VETO_SIGNALS


def is_counter_trend(direction: Optional[str], cycle_phase: Optional[str]) -> bool:
    """True if the setup's direction contradicts the cycle phase."""
    if not direction or not cycle_phase:
        return False
    if direction == "long" and cycle_phase in BEAR_PHASES:
        return True
    if direction == "short" and cycle_phase in BULL_PHASES:
        return True
    return False


def is_bar_count_in_range(
    setup_type: Optional[str], cycle_phase: Optional[str]
) -> bool:
    """True if an H1/H2/L1/L2 is firing inside a trading range."""
    return (
        setup_type in BAR_COUNT_SETUPS
        and cycle_phase == "trading_range"
    )


def is_direction_signal_mismatch(
    direction: Optional[str], signal: Optional[str]
) -> bool:
    """True if the scanner signal endorses the opposite side from the setup.

    E.g. detection is L2 short but signal is BUY_PULLBACK — the scanner is
    looking for longs and we're firing a short. Unknown/None signals are
    treated as neutral (not mismatch) so the gate stays fail-open.
    """
    if not direction or not signal:
        return False
    if signal not in ENDORSING_SIGNALS:
        return False  # let the veto-signal gate handle VETO_SIGNALS separately
    if direction == "long" and signal not in LONG_ENDORSING_SIGNALS:
        return True
    if direction == "short" and signal not in SHORT_ENDORSING_SIGNALS:
        return True
    return False


def should_drop_detection(
    signal: Optional[str],
    direction: Optional[str],
    cycle_phase: Optional[str],
    setup_type: Optional[str] = None,
    ticker: Optional[str] = None,
    enforce_whitelist: bool = True,
) -> tuple[bool, Optional[str]]:
    """Return (drop?, reason). Reason codes:
       - ``veto:<signal>``                        scanner said AVOID/FOG/WAIT/PASS
       - ``mismatch:<dir>/<signal>``              signal endorses opposite side
       - ``counter-trend:<dir>/<phase>``          direction contradicts phase
       - ``bar-count-in-range:<setup>``           H1/H2/L1/L2 inside trading_range
       - ``not-whitelisted:<ticker>/<setup>``     (ticker, setup) not in whitelist
    """
    if is_vetoed_signal(signal):
        return True, f"veto:{signal}"
    if is_direction_signal_mismatch(direction, signal):
        return True, f"mismatch:{direction}/{signal}"
    if is_counter_trend(direction, cycle_phase):
        return True, f"counter-trend:{direction}/{cycle_phase}"
    if is_bar_count_in_range(setup_type, cycle_phase):
        return True, f"bar-count-in-range:{setup_type}"
    if enforce_whitelist and ticker and setup_type:
        if not _whitelist_contains(ticker, setup_type):
            return True, f"not-whitelisted:{ticker}/{setup_type}"
    return False, None


# ── Whitelist ─────────────────────────────────────────────────────────────────

# Cache invalidated on write; small enough (dozens of rows) to keep in memory.
_WHITELIST_CACHE: Optional[frozenset[tuple[str, str]]] = None
_WHITELIST_SETUP_TYPES_CACHE: Optional[frozenset[str]] = None


def _load_whitelist_caches() -> None:
    global _WHITELIST_CACHE, _WHITELIST_SETUP_TYPES_CACHE
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT ticker, setup_type FROM setup_whitelist"
        ).fetchall()
        _WHITELIST_CACHE = frozenset((r[0], r[1]) for r in rows)
        _WHITELIST_SETUP_TYPES_CACHE = frozenset(r[1] for r in rows)
    finally:
        conn.close()


def _whitelist_contains(ticker: str, setup_type: str) -> bool:
    """Data-driven whitelist gate with per-setup_type scoping:

    - If the whitelist is completely empty → allow (fail open).
    - If the whitelist has rows for this setup_type → allow only if
      (ticker, setup_type) is explicitly listed.
    - If the whitelist has rows but none for this setup_type → allow
      (we haven't backtested this setup yet, so don't filter it).
    """
    if _WHITELIST_CACHE is None or _WHITELIST_SETUP_TYPES_CACHE is None:
        _load_whitelist_caches()
    assert _WHITELIST_CACHE is not None and _WHITELIST_SETUP_TYPES_CACHE is not None
    if not _WHITELIST_CACHE:
        return True
    if setup_type not in _WHITELIST_SETUP_TYPES_CACHE:
        return True
    return (ticker, setup_type) in _WHITELIST_CACHE


def _invalidate_whitelist_cache() -> None:
    global _WHITELIST_CACHE, _WHITELIST_SETUP_TYPES_CACHE
    _WHITELIST_CACHE = None
    _WHITELIST_SETUP_TYPES_CACHE = None


def populate_whitelist_from_run(
    run_id: str,
    min_win_rate: float = 0.50,
    min_sample: int = 20,
    replace: bool = True,
) -> list[dict]:
    """Rebuild the whitelist from a backtest run.

    For each (ticker, setup_type) in the run, compute win rate and add to
    the whitelist if it meets both thresholds. Returns the list of rows
    written. When replace=True the whitelist is cleared first.
    """
    conn = _connect()
    try:
        if replace:
            conn.execute("DELETE FROM setup_whitelist")

        candidates = conn.execute(
            """SELECT ticker, setup_type,
                      COUNT(*) AS total,
                      SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
                      AVG(mfe) AS avg_mfe,
                      AVG(mae) AS avg_mae
               FROM detections
               WHERE run_id=? AND result IN ('WIN', 'LOSS')
               GROUP BY ticker, setup_type""",
            (run_id,),
        ).fetchall()

        written: list[dict] = []
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for r in candidates:
            resolved = r["wins"] + r["losses"]
            if resolved < min_sample:
                continue
            wr = r["wins"] / resolved if resolved else 0.0
            if wr < min_win_rate:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO setup_whitelist
                   (ticker, setup_type, source_run, total, wins, losses,
                    win_rate, avg_mfe, avg_mae, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (r["ticker"], r["setup_type"], run_id,
                 r["total"], r["wins"], r["losses"],
                 round(wr, 3), r["avg_mfe"], r["avg_mae"], now),
            )
            written.append({
                "ticker": r["ticker"], "setup_type": r["setup_type"],
                "total": r["total"], "wins": r["wins"], "losses": r["losses"],
                "win_rate": round(wr, 3),
                "avg_mfe": r["avg_mfe"], "avg_mae": r["avg_mae"],
            })
        conn.commit()
        _invalidate_whitelist_cache()
        return written
    finally:
        conn.close()


def list_whitelist() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM setup_whitelist ORDER BY win_rate DESC, total DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_whitelist() -> int:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM setup_whitelist")
        conn.commit()
        _invalidate_whitelist_cache()
        return cur.rowcount
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    ticker              TEXT    NOT NULL,
    setup_type          TEXT    NOT NULL,
    detected_at         TEXT    NOT NULL,
    detection_date      TEXT    NOT NULL,

    -- Bar context
    bar_index           INTEGER NOT NULL,
    bar_count_at_detect INTEGER NOT NULL,
    session_bar_number  INTEGER NOT NULL,

    -- Detector output
    entry_price         REAL,
    stop_price          REAL,
    target_price        REAL,
    entry_mode          TEXT,           -- "stop" | "limit" | "market" (Brooks entry method)
    confidence          REAL    NOT NULL,
    direction           TEXT    NOT NULL,

    -- Market context at detection
    urgency             REAL,
    uncertainty         REAL,
    always_in           TEXT,
    cycle_phase         TEXT,
    day_type            TEXT,
    signal              TEXT,
    gap_direction       TEXT,
    bpa_alignment       REAL,

    -- Price at detection
    price_at_detect     REAL    NOT NULL,

    -- Outcome status
    outcome_status      TEXT    NOT NULL DEFAULT 'pending',

    -- Checkpoints (5, 10, 20, 30 five-min bars after detection)
    ck5_high            REAL,
    ck5_low             REAL,
    ck5_close           REAL,

    ck10_high           REAL,
    ck10_low            REAL,
    ck10_close          REAL,

    ck20_high           REAL,
    ck20_low            REAL,
    ck20_close          REAL,

    ck30_high           REAL,
    ck30_low            REAL,
    ck30_close          REAL,

    -- Excursion
    mfe                 REAL,
    mae                 REAL,

    -- Final outcome
    result              TEXT,
    result_bars         INTEGER,
    hit_target_bar      INTEGER,
    hit_stop_bar        INTEGER,

    -- Chart window (JSON-encoded ChartData for site viewer)
    chart_json          TEXT,

    -- Backtest run tagging (NULL = live production detection)
    run_id              TEXT
);

CREATE INDEX IF NOT EXISTS idx_det_status ON detections(outcome_status);
CREATE INDEX IF NOT EXISTS idx_det_date   ON detections(detection_date);
CREATE INDEX IF NOT EXISTS idx_det_type   ON detections(setup_type);
CREATE INDEX IF NOT EXISTS idx_det_result ON detections(result);
CREATE INDEX IF NOT EXISTS idx_det_run_id ON detections(run_id);

-- Partial unique indexes: live dedup on (ticker, setup, date, bar); backtest
-- dedup scoped to the specific run so separate backtests can overlap.
CREATE UNIQUE INDEX IF NOT EXISTS idx_det_unique_live
    ON detections(ticker, setup_type, detection_date, bar_index)
    WHERE run_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_det_unique_backtest
    ON detections(ticker, setup_type, detection_date, bar_index, run_id)
    WHERE run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id           TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    symbols          TEXT NOT NULL,
    date_from        TEXT NOT NULL,
    date_to          TEXT NOT NULL,
    setup_filter     TEXT,
    total_detections INTEGER NOT NULL DEFAULT 0,
    args_json        TEXT NOT NULL
);

-- Per-(ticker, setup_type) whitelist. Populated from historical backtest
-- win rates. The scanner only fires a detection if the pair is whitelisted.
CREATE TABLE IF NOT EXISTS setup_whitelist (
    ticker       TEXT NOT NULL,
    setup_type   TEXT NOT NULL,
    source_run   TEXT NOT NULL,       -- run_id used to derive this row
    total        INTEGER NOT NULL,
    wins         INTEGER NOT NULL,
    losses       INTEGER NOT NULL,
    win_rate     REAL,
    avg_mfe      REAL,
    avg_mae      REAL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (ticker, setup_type)
);
"""


# ── Connection ────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ── Migration ─────────────────────────────────────────────────────────────────

_REBUILD_TABLE_SQL = """
CREATE TABLE detections_new (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    setup_type          TEXT    NOT NULL,
    detected_at         TEXT    NOT NULL,
    detection_date      TEXT    NOT NULL,
    bar_index           INTEGER NOT NULL,
    bar_count_at_detect INTEGER NOT NULL,
    session_bar_number  INTEGER NOT NULL,
    entry_price         REAL,
    stop_price          REAL,
    target_price        REAL,
    entry_mode          TEXT,
    confidence          REAL    NOT NULL,
    direction           TEXT    NOT NULL,
    urgency             REAL,
    uncertainty         REAL,
    always_in           TEXT,
    cycle_phase         TEXT,
    day_type            TEXT,
    signal              TEXT,
    gap_direction       TEXT,
    bpa_alignment       REAL,
    price_at_detect     REAL    NOT NULL,
    outcome_status      TEXT    NOT NULL DEFAULT 'pending',
    ck5_high            REAL,
    ck5_low             REAL,
    ck5_close           REAL,
    ck10_high           REAL,
    ck10_low            REAL,
    ck10_close          REAL,
    ck20_high           REAL,
    ck20_low            REAL,
    ck20_close          REAL,
    ck30_high           REAL,
    ck30_low            REAL,
    ck30_close          REAL,
    mfe                 REAL,
    mae                 REAL,
    result              TEXT,
    result_bars         INTEGER,
    hit_target_bar      INTEGER,
    hit_stop_bar        INTEGER,
    chart_json          TEXT,
    run_id              TEXT
)
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing detections table up to the current schema.

    Adds chart_json + run_id columns if missing, and replaces the old
    table-level UNIQUE (ticker, setup_type, detection_date, bar_index) with
    partial unique indexes by rebuilding the table. Idempotent.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)").fetchall()}
    if not cols:
        return  # fresh DB; SCHEMA will create it correctly

    added_any = False
    if "chart_json" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN chart_json TEXT")
        added_any = True
    if "run_id" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN run_id TEXT")
        added_any = True
    if "entry_mode" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN entry_mode TEXT")
        added_any = True
    if added_any:
        conn.commit()

    # If the old table-level UNIQUE is still in place it shows up as an
    # auto-index. Rebuild to drop it so the partial indexes can take over.
    auto_idx = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND tbl_name='detections' "
        "AND name LIKE 'sqlite_autoindex_detections_%'"
    ).fetchall()
    if auto_idx:
        _rebuild_detections_without_table_unique(conn)


def _rebuild_detections_without_table_unique(conn: sqlite3.Connection) -> None:
    """Copy detections into a new table that lacks the old table-level UNIQUE."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)").fetchall()]
    col_csv = ", ".join(cols)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(_REBUILD_TABLE_SQL)
        conn.execute(
            f"INSERT INTO detections_new ({col_csv}) SELECT {col_csv} FROM detections"
        )
        conn.execute("DROP TABLE detections")
        conn.execute("ALTER TABLE detections_new RENAME TO detections")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def init_db() -> None:
    conn = _connect()
    try:
        # Migrate first so SCHEMA's CREATE INDEX IF NOT EXISTS can safely run.
        _migrate(conn)
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# Auto-init on import
try:
    init_db()
except Exception as e:
    logger.warning(f"Pattern Lab DB init failed (non-fatal): {e}")


# ── Chart JSON builder ────────────────────────────────────────────────────────

def build_chart_json(
    df5_full,
    bar_index: int,
    entry: Optional[float],
    stop: Optional[float],
    target: Optional[float],
    direction: str,
    prior_close: Optional[float] = None,
    cycle_phase: Optional[str] = None,
    window_before: int = 30,
    window_after: int = 20,
) -> Optional[str]:
    """Slice ±window bars around the signal and emit the site's ChartData shape.

    df5_full must be a pandas DataFrame with columns:
    open/high/low/close/volume/datetime (datetime parseable by pd.Timestamp).
    Returns a compact JSON string suitable for storage in detections.chart_json,
    or None if inputs are invalid.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None

    if df5_full is None or len(df5_full) == 0:
        return None
    if bar_index < 0 or bar_index >= len(df5_full):
        return None

    start = max(0, bar_index - window_before)
    end = min(len(df5_full), bar_index + window_after + 1)
    sl = df5_full.iloc[start:end]

    bars: list[dict] = []
    signal_time: Optional[int] = None
    for i, (_, row) in enumerate(sl.iterrows()):
        try:
            ts = int(pd.Timestamp(row["datetime"]).timestamp())
        except Exception:
            continue
        if start + i == bar_index:
            signal_time = ts
        bar = {
            "t": ts,
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
        }
        vol = row.get("volume") if hasattr(row, "get") else None
        if vol is not None:
            try:
                bar["v"] = float(vol)
            except (TypeError, ValueError):
                pass
        bars.append(bar)

    if not bars:
        return None

    payload: dict = {"bars": bars, "timeframe": "5min"}

    if prior_close is not None:
        try:
            payload["keyLevels"] = {"priorClose": float(prior_close)}
        except (TypeError, ValueError):
            pass

    annotations: dict = {}
    if stop is not None:
        try:
            annotations["stopPrice"] = float(stop)
        except (TypeError, ValueError):
            pass
    if target is not None:
        try:
            annotations["targetPrice"] = float(target)
        except (TypeError, ValueError):
            pass
    if signal_time is not None and direction in ("long", "short"):
        annotations["signalBar"] = {"time": signal_time, "direction": direction}
    if cycle_phase:
        annotations["phaseLabel"] = cycle_phase
    if annotations:
        payload["annotations"] = annotations

    return json.dumps(payload, separators=(",", ":"))


# ── Detection Logging ─────────────────────────────────────────────────────────

def log_detection(
    ticker: str,
    setup_type: str,
    detected_at: str,
    detection_date: str,
    bar_index: int,
    bar_count_at_detect: int,
    session_bar_number: int,
    entry_price: Optional[float],
    stop_price: Optional[float],
    target_price: Optional[float],
    confidence: float,
    direction: str,
    price_at_detect: float,
    entry_mode: Optional[str] = None,   # "stop" | "limit" | "market"
    urgency: Optional[float] = None,
    uncertainty: Optional[float] = None,
    always_in: Optional[str] = None,
    cycle_phase: Optional[str] = None,
    day_type: Optional[str] = None,
    signal: Optional[str] = None,
    gap_direction: Optional[str] = None,
    bpa_alignment: Optional[float] = None,
    chart_json: Optional[str] = None,
    run_id: Optional[str] = None,
    force: bool = False,
    enforce_whitelist: bool = True,
) -> Optional[int]:
    """Insert a detection row. Returns row id, or None on dedup/gate rejection.

    Five gates, all bypassable with ``force=True``:
      - Signal veto               — signal ∈ {AVOID, FOG, WAIT, PASS}
      - Direction/signal mismatch — long setup with SELL* signal (or vice versa)
      - Counter-trend phase       — long in bear_channel, short in bull_channel
      - Bar-count in range        — H1/H2/L1/L2 during trading_range (Brooks #54)
      - Whitelist                 — (ticker, setup_type) not in setup_whitelist

    Pass ``force=True`` to bypass (e.g. research on rejected setups). Backtests
    that want to evaluate non-whitelisted buckets should set ``enforce_whitelist=False``
    via ``should_drop_detection`` directly.
    """
    if not force:
        drop, _reason = should_drop_detection(
            signal=signal, direction=direction,
            cycle_phase=cycle_phase, setup_type=setup_type,
            ticker=ticker, enforce_whitelist=enforce_whitelist,
        )
        if drop:
            return None

    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO detections (
                ticker, setup_type, detected_at, detection_date,
                bar_index, bar_count_at_detect, session_bar_number,
                entry_price, stop_price, target_price, entry_mode,
                confidence, direction,
                price_at_detect,
                urgency, uncertainty, always_in, cycle_phase,
                day_type, signal, gap_direction, bpa_alignment,
                chart_json, run_id
            ) VALUES (?,?,?,?, ?,?,?, ?,?,?,?, ?,?, ?, ?,?,?,?, ?,?,?,?, ?,?)""",
            (
                ticker, setup_type, detected_at, detection_date,
                bar_index, bar_count_at_detect, session_bar_number,
                entry_price, stop_price, target_price, entry_mode,
                confidence, direction,
                price_at_detect,
                urgency, uncertainty, always_in, cycle_phase,
                day_type, signal, gap_direction, bpa_alignment,
                chart_json, run_id,
            ),
        )
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None
    except Exception as e:
        logger.debug(f"Pattern Lab insert error: {e}")
        return None
    finally:
        conn.close()


def update_chart_json(detection_id: int, chart_json: str) -> None:
    """Set chart_json on an existing detection. Used by --refill-charts."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE detections SET chart_json=? WHERE id=?",
            (chart_json, detection_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Outcome Tracking ──────────────────────────────────────────────────────────

def get_pending_detections(detection_date: Optional[str] = None) -> list[dict]:
    """Return all detections needing outcome updates."""
    conn = _connect()
    try:
        if detection_date:
            rows = conn.execute(
                "SELECT * FROM detections WHERE outcome_status IN ('pending','partial') AND detection_date = ?",
                (detection_date,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detections WHERE outcome_status IN ('pending','partial')"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_checkpoint(
    detection_id: int,
    checkpoint: int,
    high: float,
    low: float,
    close: float,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
) -> None:
    """Fill in a single checkpoint (5, 10, 20, or 30)."""
    col_h = f"ck{checkpoint}_high"
    col_l = f"ck{checkpoint}_low"
    col_c = f"ck{checkpoint}_close"
    conn = _connect()
    try:
        sql = f"""UPDATE detections
                  SET {col_h}=?, {col_l}=?, {col_c}=?,
                      mfe=COALESCE(?, mfe), mae=COALESCE(?, mae),
                      outcome_status='partial'
                  WHERE id=?"""
        conn.execute(sql, (high, low, close, mfe, mae, detection_id))
        conn.commit()
    finally:
        conn.close()


def finalize_outcome(
    detection_id: int,
    result: str,
    result_bars: Optional[int] = None,
    hit_target_bar: Optional[int] = None,
    hit_stop_bar: Optional[int] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
) -> None:
    """Mark a detection as complete with its final result."""
    conn = _connect()
    try:
        conn.execute(
            """UPDATE detections
               SET outcome_status='complete', result=?, result_bars=?,
                   hit_target_bar=?, hit_stop_bar=?,
                   mfe=COALESCE(?, mfe), mae=COALESCE(?, mae)
               WHERE id=?""",
            (result, result_bars, hit_target_bar, hit_stop_bar, mfe, mae, detection_id),
        )
        conn.commit()
    finally:
        conn.close()


def expire_pending(detection_date: str) -> int:
    """Mark all remaining pending/partial for a date as INCOMPLETE."""
    conn = _connect()
    try:
        cur = conn.execute(
            """UPDATE detections
               SET outcome_status='complete', result='INCOMPLETE'
               WHERE detection_date=? AND outcome_status IN ('pending','partial')""",
            (detection_date,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ── Run filter helper ─────────────────────────────────────────────────────────

def _apply_run_filter(clauses: list, params: list, run_id) -> None:
    """Mutate clauses/params to filter detections by run_id.

    - run_id=LIVE_ONLY → only live (WHERE run_id IS NULL)
    - run_id=<str>     → only that specific run
    - run_id=None      → no filter (mixes live + all backtest runs)
    """
    if run_id is LIVE_ONLY:
        clauses.append("run_id IS NULL")
    elif run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)


# ── Query Helpers ─────────────────────────────────────────────────────────────

_CONTEXT_ALLOWLIST = {"cycle_phase", "always_in", "day_type", "gap_direction", "signal"}


def win_rate_by_setup(
    setup_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    run_id=None,
) -> dict:
    """Return {setup_type: {total, wins, losses, scratches, incomplete, win_rate, avg_mfe, avg_mae}}."""
    conn = _connect()
    try:
        clauses = ["result IS NOT NULL"]
        params: list = []
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        if date_from:
            clauses.append("detection_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("detection_date <= ?")
            params.append(date_to)
        _apply_run_filter(clauses, params, run_id)

        where = " AND ".join(clauses)
        rows = conn.execute(
            f"""SELECT setup_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                       SUM(CASE WHEN result='SCRATCH' THEN 1 ELSE 0 END) as scratches,
                       SUM(CASE WHEN result='INCOMPLETE' THEN 1 ELSE 0 END) as incomplete,
                       AVG(mfe) as avg_mfe,
                       AVG(mae) as avg_mae
                FROM detections WHERE {where}
                GROUP BY setup_type ORDER BY total DESC""",
            params,
        ).fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            resolved = d["wins"] + d["losses"]
            d["win_rate"] = round(d["wins"] / resolved, 3) if resolved > 0 else None
            out[d.pop("setup_type")] = d
        return out
    finally:
        conn.close()


def win_rate_by_context(
    context_field: str,
    setup_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    run_id=None,
) -> dict:
    """Group win rates by a context dimension (cycle_phase, always_in, day_type, etc.)."""
    if context_field not in _CONTEXT_ALLOWLIST:
        raise ValueError(f"context_field must be one of {_CONTEXT_ALLOWLIST}")

    conn = _connect()
    try:
        clauses = ["result IS NOT NULL", f"{context_field} IS NOT NULL"]
        params: list = []
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        if date_from:
            clauses.append("detection_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("detection_date <= ?")
            params.append(date_to)
        _apply_run_filter(clauses, params, run_id)

        where = " AND ".join(clauses)
        rows = conn.execute(
            f"""SELECT {context_field} as ctx, setup_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                       AVG(mfe) as avg_mfe, AVG(mae) as avg_mae
                FROM detections WHERE {where}
                GROUP BY ctx, setup_type ORDER BY ctx, total DESC""",
            params,
        ).fetchall()
        out: dict = {}
        for r in rows:
            d = dict(r)
            ctx = d.pop("ctx")
            resolved = d["wins"] + d["losses"]
            d["win_rate"] = round(d["wins"] / resolved, 3) if resolved > 0 else None
            out.setdefault(ctx, []).append(d)
        return out
    finally:
        conn.close()


def win_rate_by_time_of_day(
    setup_type: Optional[str] = None,
    bucket_size: int = 6,
    run_id=None,
) -> list[dict]:
    """Group win rates by time-of-day buckets (session_bar_number / bucket_size on 5-min bars)."""
    conn = _connect()
    try:
        clauses = ["result IS NOT NULL"]
        params: list = []
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        _apply_run_filter(clauses, params, run_id)
        params.append(bucket_size)

        where = " AND ".join(clauses)
        rows = conn.execute(
            f"""SELECT (session_bar_number / ?) * ? as bucket_start,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses
                FROM detections WHERE {where}
                GROUP BY bucket_start ORDER BY bucket_start""",
            params + [bucket_size],
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def excursion_distribution(
    setup_type: Optional[str] = None,
    metric: str = "mfe",
    run_id=None,
) -> list[float]:
    """Return sorted list of MFE or MAE values."""
    col = "mfe" if metric == "mfe" else "mae"
    conn = _connect()
    try:
        clauses = [f"{col} IS NOT NULL"]
        params: list = []
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        _apply_run_filter(clauses, params, run_id)
        where = " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT {col} FROM detections WHERE {where} ORDER BY {col}",
            params,
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def recent_detections(
    limit: int = 50,
    ticker: Optional[str] = None,
    setup_type: Optional[str] = None,
    run_id=None,
) -> list[dict]:
    """Return most recent detections."""
    conn = _connect()
    try:
        clauses: list[str] = []
        params: list = []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
        _apply_run_filter(clauses, params, run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def detection_count(
    detection_date: Optional[str] = None,
    run_id=None,
) -> int:
    """Quick count of detections."""
    conn = _connect()
    try:
        clauses: list[str] = []
        params: list = []
        if detection_date:
            clauses.append("detection_date = ?")
            params.append(detection_date)
        _apply_run_filter(clauses, params, run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        r = conn.execute(f"SELECT COUNT(*) FROM detections {where}", params).fetchone()
        return r[0]
    finally:
        conn.close()


def detections_by_day(run_id: Optional[str] = None) -> list[dict]:
    """Per-day aggregate for a run (or all live detections if run_id=None)."""
    conn = _connect()
    try:
        clauses: list[str] = []
        params: list = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        else:
            clauses.append("run_id IS NULL")
        where = "WHERE " + " AND ".join(clauses)
        rows = conn.execute(
            f"""SELECT detection_date,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN'     THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN result='LOSS'    THEN 1 ELSE 0 END) as losses,
                       SUM(CASE WHEN result='SCRATCH' THEN 1 ELSE 0 END) as scratches,
                       AVG(mfe) as avg_mfe,
                       AVG(mae) as avg_mae
                FROM detections {where}
                GROUP BY detection_date
                ORDER BY detection_date""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Backtest run registry ─────────────────────────────────────────────────────

def register_backtest_run(
    run_id: str,
    symbols: list,
    date_from: str,
    date_to: str,
    setup_filter: Optional[str],
    args_json: str,
) -> None:
    """Create (or reset) a backtest_runs row."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO backtest_runs
               (run_id, created_at, symbols, date_from, date_to, setup_filter,
                total_detections, args_json)
               VALUES (?,?,?,?,?,?,0,?)""",
            (
                run_id,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                json.dumps(symbols),
                date_from,
                date_to,
                setup_filter,
                args_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_backtest_run_total(run_id: str) -> int:
    """Recount detections for a run and store in the run row."""
    conn = _connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE backtest_runs SET total_detections=? WHERE run_id=?",
            (n, run_id),
        )
        conn.commit()
        return n
    finally:
        conn.close()


def get_backtest_run(run_id: str) -> Optional[dict]:
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def list_backtest_runs(limit: int = 50) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_backtest_run(run_id: str) -> int:
    """Delete all detections for a run plus its metadata row. Returns count deleted."""
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM detections WHERE run_id=?", (run_id,))
        n = cur.rowcount
        conn.execute("DELETE FROM backtest_runs WHERE run_id=?", (run_id,))
        conn.commit()
        return n
    finally:
        conn.close()


def delete_run_detections_outside_setups(
    run_id: str,
    keep_setups: list[str],
) -> int:
    """Post-filter: drop detections in this run whose setup_type isn't in keep_setups."""
    if not keep_setups:
        return 0
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(keep_setups))
        cur = conn.execute(
            f"DELETE FROM detections WHERE run_id=? AND setup_type NOT IN ({placeholders})",
            [run_id, *keep_setups],
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ── Purge (retroactive gate application) ──────────────────────────────────────

def purge_polluted_detections(
    scope: str = "live",
    run_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Retroactively apply both gates to existing detections.

    scope:
      - "live"  : only rows where run_id IS NULL (default)
      - "run"   : only rows matching run_id (requires run_id=...)
      - "all"   : every row regardless of run_id

    Returns ``{"scanned": N, "vetoed": X, "counter_trend": Y, "deleted": Z,
    "remaining": R}``. When ``dry_run=True`` nothing is deleted but all
    counts are computed.
    """
    conn = _connect()
    try:
        clauses: list[str] = []
        params: list = []
        if scope == "live":
            clauses.append("run_id IS NULL")
        elif scope == "run":
            if not run_id:
                raise ValueError("scope='run' requires run_id")
            clauses.append("run_id = ?")
            params.append(run_id)
        elif scope != "all":
            raise ValueError(f"invalid scope: {scope!r}")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT id, ticker, setup_type, signal, direction, cycle_phase FROM detections {where}",
            params,
        ).fetchall()

        to_delete: list[int] = []
        buckets = {"vetoed": 0, "counter_trend": 0, "mismatch": 0,
                   "range_barcount": 0, "not_whitelisted": 0}
        for r in rows:
            drop, reason = should_drop_detection(
                signal=r["signal"], direction=r["direction"],
                cycle_phase=r["cycle_phase"], setup_type=r["setup_type"],
                ticker=r["ticker"],
            )
            if drop:
                to_delete.append(r["id"])
                if not reason:
                    continue
                if reason.startswith("veto:"):
                    buckets["vetoed"] += 1
                elif reason.startswith("counter-trend:"):
                    buckets["counter_trend"] += 1
                elif reason.startswith("mismatch:"):
                    buckets["mismatch"] += 1
                elif reason.startswith("bar-count-in-range:"):
                    buckets["range_barcount"] += 1
                elif reason.startswith("not-whitelisted:"):
                    buckets["not_whitelisted"] += 1

        deleted = 0
        if to_delete and not dry_run:
            # Delete in chunks to avoid massive IN clauses
            CHUNK = 900
            for i in range(0, len(to_delete), CHUNK):
                chunk = to_delete[i:i + CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cur = conn.execute(
                    f"DELETE FROM detections WHERE id IN ({placeholders})",
                    chunk,
                )
                deleted += cur.rowcount
            conn.commit()

        remaining = conn.execute(
            f"SELECT COUNT(*) FROM detections {where}", params
        ).fetchone()[0]

        return {
            "scope": scope,
            "run_id": run_id,
            "scanned": len(rows),
            **buckets,
            "to_delete": len(to_delete),
            "deleted": deleted,
            "remaining": remaining,
            "dry_run": dry_run,
        }
    finally:
        conn.close()
