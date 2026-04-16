"""
Pattern Lab — BPA setup detection database.

Records every detected Brooks Price Action setup from the live scanner
and tracks outcomes over subsequent bars to build statistical evidence.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.config_loader import get_project_root

logger = logging.getLogger(__name__)

DB_PATH = get_project_root() / "db" / "pattern_lab.sqlite"

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

    UNIQUE(ticker, setup_type, detection_date, bar_index)
);

CREATE INDEX IF NOT EXISTS idx_det_status ON detections(outcome_status);
CREATE INDEX IF NOT EXISTS idx_det_date   ON detections(detection_date);
CREATE INDEX IF NOT EXISTS idx_det_type   ON detections(setup_type);
CREATE INDEX IF NOT EXISTS idx_det_result ON detections(result);
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


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


# Auto-init on import
try:
    init_db()
except Exception as e:
    logger.warning(f"Pattern Lab DB init failed (non-fatal): {e}")


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
    urgency: Optional[float] = None,
    uncertainty: Optional[float] = None,
    always_in: Optional[str] = None,
    cycle_phase: Optional[str] = None,
    day_type: Optional[str] = None,
    signal: Optional[str] = None,
    gap_direction: Optional[str] = None,
    bpa_alignment: Optional[float] = None,
) -> Optional[int]:
    """Insert a detection row. Returns row id, or None on dedup conflict."""
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO detections (
                ticker, setup_type, detected_at, detection_date,
                bar_index, bar_count_at_detect, session_bar_number,
                entry_price, stop_price, target_price, confidence, direction,
                price_at_detect,
                urgency, uncertainty, always_in, cycle_phase,
                day_type, signal, gap_direction, bpa_alignment
            ) VALUES (?,?,?,?, ?,?,?, ?,?,?,?,?, ?, ?,?,?,?, ?,?,?,?)""",
            (
                ticker, setup_type, detected_at, detection_date,
                bar_index, bar_count_at_detect, session_bar_number,
                entry_price, stop_price, target_price, confidence, direction,
                price_at_detect,
                urgency, uncertainty, always_in, cycle_phase,
                day_type, signal, gap_direction, bpa_alignment,
            ),
        )
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None
    except Exception as e:
        logger.debug(f"Pattern Lab insert error: {e}")
        return None
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


# ── Query Helpers ─────────────────────────────────────────────────────────────

_CONTEXT_ALLOWLIST = {"cycle_phase", "always_in", "day_type", "gap_direction", "signal"}


def win_rate_by_setup(
    setup_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
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
) -> list[dict]:
    """Group win rates by time-of-day buckets (session_bar_number / bucket_size on 5-min bars)."""
    conn = _connect()
    try:
        clauses = ["result IS NOT NULL"]
        params: list = []
        if setup_type:
            clauses.append("setup_type = ?")
            params.append(setup_type)
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
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM detections {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def detection_count(detection_date: Optional[str] = None) -> int:
    """Quick count of detections."""
    conn = _connect()
    try:
        if detection_date:
            r = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE detection_date=?",
                (detection_date,),
            ).fetchone()
        else:
            r = conn.execute("SELECT COUNT(*) FROM detections").fetchone()
        return r[0]
    finally:
        conn.close()
