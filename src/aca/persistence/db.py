"""SQLite connection management and transactions.

The reducer is the single writer, so a single serialized connection is sufficient (DESIGN
8.1). WAL mode plus ``FULL`` synchronous gives durability across crash/power-loss (DESIGN
23.5). Transactions are explicit: durable acceptance and reduction happen in one commit so an
ACKed event cannot be lost (invariant 14).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

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
