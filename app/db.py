"""SQLite layer for the polevision dashboard.

Schema is intentionally tiny — captures, runs, measurements, ref_taps —
and lives behind plain `sqlite3` calls (no ORM). The DB path is taken
from the `POLEVISION_DB` env var so tests can swap a temp file in.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

DEFAULT_DB_PATH = "polevision.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    folder_path TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    n_photos INTEGER,
    gps_centroid_lat REAL,
    gps_centroid_lon REAL,
    gps_baseline_m REAL,
    notes TEXT,
    deleted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at DATETIME,
    finished_at DATETIME,
    log_path TEXT,
    exit_code INTEGER,
    artifact_path TEXT,
    params TEXT
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    value REAL NOT NULL,
    sigma REAL,
    source_run_id INTEGER REFERENCES runs(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ref_taps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    top_x REAL NOT NULL,
    top_y REAL NOT NULL,
    bot_x REAL NOT NULL,
    bot_y REAL NOT NULL,
    length_m REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_capture ON runs(capture_id, stage);
CREATE INDEX IF NOT EXISTS idx_measurements_capture
    ON measurements(capture_id, kind);
"""


def db_path() -> str:
    return os.environ.get("POLEVISION_DB", DEFAULT_DB_PATH)


def init_schema(path: str | None = None) -> None:
    """Create the tables idempotently."""
    p = path or db_path()
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect():
    """Yield a connection with row-factory set to dict-like rows."""
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
