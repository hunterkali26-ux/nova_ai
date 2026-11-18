"""SQLite persistence layer for NOVA AI.

Handles connection management, schema initialization, versioning, and basic
migrations. All databases live in the project ``databases/`` directory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from ..core.config import DATABASES_DIR
from ..core.logging import get_logger

logger = get_logger("database")

SCHEMA_VERSION = 1


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False allows the connection to be used by FastAPI's
    # worker threads (the app creates the connection at startup).
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


class DatabaseManager:
    """Manages a single SQLite database with schema versioning."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = _connect(self.db_path)
            self._ensure_meta_table()
            self.migrate()
        return self.conn

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_meta_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def set_version(self, version: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(version),),
        )
        self.conn.commit()

    def migrate(self) -> None:
        """Apply migrations up to SCHEMA_VERSION."""
        current = self.get_version()
        if current >= SCHEMA_VERSION:
            return
        logger.info("Migrating %s from schema v%d to v%d", self.db_path, current, SCHEMA_VERSION)
        # v0 -> v1: handled by the individual store modules creating their tables.
        # This hook exists for future additive migrations.
        self.set_version(SCHEMA_VERSION)

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        return self.connect().executemany(sql, params_seq)

    def commit(self) -> None:
        self.connect().commit()
