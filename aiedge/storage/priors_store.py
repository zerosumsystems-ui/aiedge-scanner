"""SQLite-backed empirical-prior storage for setup win-rates.

Tracks per-stratum win rates so the aggregator can condition on
(setup_type, regime, htf_alignment, day_type). Keys are composite;
values are running (wins, losses, last_updated).

Schema:

    CREATE TABLE setup_priors (
        setup_type     TEXT NOT NULL,
        regime         TEXT NOT NULL,      -- "low" | "mid" | "high" vol tercile
        htf_alignment  TEXT NOT NULL,      -- "aligned" | "mixed" | "opposed" | "no_data"
        day_type       TEXT NOT NULL,
        wins           INTEGER NOT NULL DEFAULT 0,
        losses         INTEGER NOT NULL DEFAULT 0,
        last_updated   TEXT NOT NULL,
        PRIMARY KEY (setup_type, regime, htf_alignment, day_type)
    );

Read access is cached in-memory for one call-site's lifetime; writes
are immediate. Callers pass a `db_path` so tests use :memory:.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS setup_priors (
    setup_type     TEXT NOT NULL,
    regime         TEXT NOT NULL,
    htf_alignment  TEXT NOT NULL,
    day_type       TEXT NOT NULL,
    wins           INTEGER NOT NULL DEFAULT 0,
    losses         INTEGER NOT NULL DEFAULT 0,
    last_updated   TEXT NOT NULL,
    PRIMARY KEY (setup_type, regime, htf_alignment, day_type)
);
"""


class PriorsStore:
    """Thin SQLite wrapper around the setup_priors table."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # ── Writes ──────────────────────────────────────────────────────

    def record_outcome(
        self,
        setup_type: str,
        regime: str,
        htf_alignment: str,
        day_type: str,
        won: bool,
    ) -> None:
        """Increment wins or losses for this stratum."""
        now = datetime.now(timezone.utc).isoformat()
        col = "wins" if won else "losses"
        cur = self._conn.cursor()
        cur.execute(
            f"""
            INSERT INTO setup_priors
                (setup_type, regime, htf_alignment, day_type, {col}, last_updated)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(setup_type, regime, htf_alignment, day_type) DO UPDATE SET
                {col} = {col} + 1,
                last_updated = excluded.last_updated
            """,
            (setup_type, regime, htf_alignment, day_type, now),
        )
        self._conn.commit()

    # ── Reads ───────────────────────────────────────────────────────

    def get(
        self,
        setup_type: str,
        regime: str,
        htf_alignment: str,
        day_type: str,
    ) -> tuple[int, int]:
        """Return (wins, losses) for the exact key; (0, 0) if absent."""
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT wins, losses FROM setup_priors
            WHERE setup_type = ? AND regime = ? AND htf_alignment = ? AND day_type = ?
            """,
            (setup_type, regime, htf_alignment, day_type),
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def all_by_setup(self, setup_type: str) -> list[dict]:
        """Return every recorded stratum for one setup type."""
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT regime, htf_alignment, day_type, wins, losses, last_updated
            FROM setup_priors
            WHERE setup_type = ?
            ORDER BY (wins + losses) DESC
            """,
            (setup_type,),
        ).fetchall()
        return [
            {
                "setup_type": setup_type,
                "regime": r[0], "htf_alignment": r[1], "day_type": r[2],
                "wins": r[3], "losses": r[4],
                "last_updated": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PriorsStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
