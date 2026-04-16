"""
Shared SQLite logger. All pipeline runs log to a database,
rows tagged by pipeline_name.
Falls back to JSON file if SQLite fails (e.g., FUSE filesystem issues).
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.config_loader import get_project_root


DB_PATH = get_project_root() / "db" / "pipeline.sqlite"
JSON_LOG_PATH = get_project_root() / "db" / "pipeline_log.json"

# Check for environment variable to force JSON mode
FORCE_JSON_MODE = os.getenv("PIPELINE_USE_JSON_LOG", "").lower() in ("1", "true", "yes")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    date TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    screener_result_count INTEGER,
    script_segment_count INTEGER,
    total_duration_seconds REAL,
    output_path TEXT,
    youtube_video_id TEXT,
    youtube_url TEXT,
    newsletter_status TEXT,
    newsletter_post_id TEXT,
    newsletter_url TEXT,
    estimated_cost_usd REAL DEFAULT 0.0,
    error_message TEXT,
    error_stage TEXT,
    state_json TEXT,
    UNIQUE(run_id)
);

CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    service TEXT NOT NULL,
    endpoint TEXT,
    tokens_used INTEGER,
    characters_used INTEGER,
    cost_usd REAL,
    status TEXT,
    error TEXT,
    retry_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS daily_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    total_cost_usd REAL DEFAULT 0.0,
    claude_cost REAL DEFAULT 0.0,
    elevenlabs_cost REAL DEFAULT 0.0,
    kling_cost REAL DEFAULT 0.0,
    databento_cost REAL DEFAULT 0.0,
    youtube_quota_used INTEGER DEFAULT 0,
    UNIQUE(date, pipeline_name)
);

CREATE INDEX IF NOT EXISTS idx_runs_pipeline ON runs(pipeline_name);
CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(date);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_api_calls_run ON api_calls(run_id);
"""


def _load_json_log():
    """Load JSON log file."""
    if JSON_LOG_PATH.exists():
        with open(JSON_LOG_PATH, 'r') as f:
            return json.load(f)
    return {"runs": [], "api_calls": [], "daily_costs": []}

def _save_json_log(data):
    """Save JSON log file."""
    JSON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_LOG_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def init_db():
    """Initialize the database and create tables if needed."""
    if FORCE_JSON_MODE:
        JSON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        return
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=2)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.warning(f"SQLite init failed: {e}. Falling back to JSON mode.")
        _load_json_log()  # Ensure JSON log exists


def _connect() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(str(DB_PATH), timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def create_run(run_id: str, pipeline_name: str) -> None:
    """Create a new run record."""
    if FORCE_JSON_MODE:
        data = _load_json_log()
        data["runs"].append({
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "started_at": datetime.now().isoformat(),
            "status": "running"
        })
        _save_json_log(data)
        return

    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO runs (run_id, pipeline_name, date, started_at, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, pipeline_name, datetime.now().strftime("%Y-%m-%d"),
             datetime.now().isoformat(), "running"),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.warning(f"SQLite write failed: {e}. Using JSON fallback.")
        data = _load_json_log()
        data["runs"].append({
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "started_at": datetime.now().isoformat(),
            "status": "running"
        })
        _save_json_log(data)


def update_run(run_id: str, **kwargs) -> None:
    """Update fields on a run record."""
    if FORCE_JSON_MODE:
        data = _load_json_log()
        for run in data["runs"]:
            if run["run_id"] == run_id:
                run.update(kwargs)
                break
        _save_json_log(data)
        return

    try:
        conn = _connect()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [run_id]
        conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", vals)
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.warning(f"SQLite update failed: {e}. Using JSON fallback.")
        data = _load_json_log()
        for run in data["runs"]:
            if run["run_id"] == run_id:
                run.update(kwargs)
                break
        _save_json_log(data)


def complete_run(run_id: str, status: str = "success", **kwargs) -> None:
    """Mark a run as completed."""
    kwargs["status"] = status
    kwargs["completed_at"] = datetime.now().isoformat()
    update_run(run_id, **kwargs)


def fail_run(run_id: str, error_message: str, error_stage: str) -> None:
    """Mark a run as failed."""
    complete_run(run_id, status="failed", error_message=error_message, error_stage=error_stage)


def log_api_call(
    run_id: str,
    pipeline_name: str,
    service: str,
    endpoint: str = None,
    tokens_used: int = None,
    characters_used: int = None,
    cost_usd: float = None,
    status: str = "success",
    error: str = None,
    retry_count: int = 0,
) -> None:
    """Log an individual API call."""
    if FORCE_JSON_MODE:
        data = _load_json_log()
        data["api_calls"].append({
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "endpoint": endpoint,
            "tokens_used": tokens_used,
            "characters_used": characters_used,
            "cost_usd": cost_usd,
            "status": status,
            "error": error,
            "retry_count": retry_count
        })
        _save_json_log(data)
        return

    try:
        conn = _connect()
        conn.execute(
            """INSERT INTO api_calls
               (run_id, pipeline_name, timestamp, service, endpoint,
                tokens_used, characters_used, cost_usd, status, error, retry_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, pipeline_name, datetime.now().isoformat(), service, endpoint,
             tokens_used, characters_used, cost_usd, status, error, retry_count),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.warning(f"SQLite API call log failed: {e}. Using JSON fallback.")
        data = _load_json_log()
        data["api_calls"].append({
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "endpoint": endpoint,
            "tokens_used": tokens_used,
            "characters_used": characters_used,
            "cost_usd": cost_usd,
            "status": status,
            "error": error,
            "retry_count": retry_count
        })
        _save_json_log(data)


def get_daily_cost(date: str = None) -> float:
    """Get total estimated cost for a date (default today)."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if FORCE_JSON_MODE:
        data = _load_json_log()
        total = sum(r.get("estimated_cost_usd", 0) for r in data["runs"] if r.get("date") == date)
        return total

    try:
        conn = _connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM runs WHERE date = ?",
            (date,),
        ).fetchone()
        conn.close()
        return row[0]
    except Exception as e:
        import logging
        logging.warning(f"SQLite read failed: {e}. Using JSON fallback.")
        data = _load_json_log()
        total = sum(r.get("estimated_cost_usd", 0) for r in data["runs"] if r.get("date") == date)
        return total


def get_daily_youtube_quota(date: str = None) -> int:
    """Get YouTube quota units used today."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if FORCE_JSON_MODE:
        return 0  # No quota tracking in JSON mode

    try:
        conn = _connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(youtube_quota_used), 0) FROM daily_costs WHERE date = ?",
            (date,),
        ).fetchone()
        conn.close()
        return row[0]
    except Exception:
        return 0


def update_daily_cost(date: str, pipeline_name: str, **kwargs) -> None:
    """Upsert daily cost tracking."""
    if FORCE_JSON_MODE:
        data = _load_json_log()
        existing = None
        for cost in data["daily_costs"]:
            if cost.get("date") == date and cost.get("pipeline_name") == pipeline_name:
                existing = cost
                break
        if existing:
            for k, v in kwargs.items():
                existing[k] = existing.get(k, 0) + v
        else:
            data["daily_costs"].append({"date": date, "pipeline_name": pipeline_name, **kwargs})
        _save_json_log(data)
        return

    try:
        conn = _connect()
        existing = conn.execute(
            "SELECT id FROM daily_costs WHERE date = ? AND pipeline_name = ?",
            (date, pipeline_name),
        ).fetchone()

        if existing:
            sets = ", ".join(f"{k} = {k} + ?" for k in kwargs)
            vals = list(kwargs.values()) + [date, pipeline_name]
            conn.execute(
                f"UPDATE daily_costs SET {sets} WHERE date = ? AND pipeline_name = ?", vals
            )
        else:
            cols = "date, pipeline_name, " + ", ".join(kwargs.keys())
            placeholders = "?, ?, " + ", ".join("?" for _ in kwargs)
            vals = [date, pipeline_name] + list(kwargs.values())
            conn.execute(f"INSERT INTO daily_costs ({cols}) VALUES ({placeholders})", vals)

        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.warning(f"SQLite daily cost update failed: {e}. Using JSON fallback.")
        data = _load_json_log()
        existing = None
        for cost in data["daily_costs"]:
            if cost.get("date") == date and cost.get("pipeline_name") == pipeline_name:
                existing = cost
                break
        if existing:
            for k, v in kwargs.items():
                existing[k] = existing.get(k, 0) + v
        else:
            data["daily_costs"].append({"date": date, "pipeline_name": pipeline_name, **kwargs})
        _save_json_log(data)


def save_run_state(run_id: str, state: dict) -> None:
    """Save checkpoint state for a run (for resume support)."""
    update_run(run_id, state_json=json.dumps(state))


def get_run_state(run_id: str) -> Optional[dict]:
    """Load checkpoint state for a run."""
    if FORCE_JSON_MODE:
        data = _load_json_log()
        for run in data["runs"]:
            if run["run_id"] == run_id and run.get("state_json"):
                return json.loads(run["state_json"])
        return None

    try:
        conn = _connect()
        row = conn.execute("SELECT state_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        conn.close()
        if row and row["state_json"]:
            return json.loads(row["state_json"])
        return None
    except Exception:
        data = _load_json_log()
        for run in data["runs"]:
            if run["run_id"] == run_id and run.get("state_json"):
                return json.loads(run["state_json"])
        return None


def get_run_api_cost(run_id: str) -> float:
    """Sum all API call costs for a run."""
    conn = _connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    return row[0]


def get_runs_for_date(date: str = None, pipeline_name: str = None) -> list[dict]:
    """Get all runs for a date, optionally filtered by pipeline."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    query = "SELECT * FROM runs WHERE date = ?"
    params = [date]
    if pipeline_name:
        query += " AND pipeline_name = ?"
        params.append(pipeline_name)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
