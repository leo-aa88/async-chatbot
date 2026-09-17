"""SQLite connection management and transactions.

The reducer is the single writer, so a single serialized connection is sufficient (DESIGN
8.1). WAL mode plus ``FULL`` synchronous gives durability across crash/power-loss (DESIGN
23.5). Transactions are explicit: durable acceptance and reduction happen in one commit so an
ACKed event cannot be lost (invariant 14).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import SCHEMA, SCHEMA_VERSION


class Database:
    """A thin wrapper over one SQLite connection with an explicit transaction helper."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, isolation_level=None, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._apply_schema()

    def _configure(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    def _apply_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    # Additive column migrations that CREATE TABLE IF NOT EXISTS can't apply to an existing table.
    _COLUMN_MIGRATIONS = (
        ("topics", "source_memory_id", "TEXT"),
        ("outbound_messages", "candidate_kind", "TEXT"),
        ("outbound_messages", "candidate_id", "TEXT"),
    )

    def _migrate(self) -> None:
        """Apply additive column migrations idempotently.

        A column is added only if PRAGMA table_info shows it's missing, so a genuine migration
        failure (locked DB, bad DDL) surfaces instead of being swallowed as "already applied".
        """
        for table, column, coltype in self._COLUMN_MIGRATIONS:
            if not self._column_exists(table, column):
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    def _column_exists(self, table: str, column: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        cur.close()
        return row

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A single atomic unit. Commits on success, rolls back on any exception."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()
