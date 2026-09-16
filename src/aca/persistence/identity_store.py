"""Agent identity and runtime-session persistence (DESIGN 23.6).

The logical ``agent_id`` is created exactly once and never re-minted on restart (invariant 29).
Each process execution is a distinct runtime-session row.
"""

from __future__ import annotations

import sqlite3

from ..domain.enums import ExitKind, LifecycleState
from ..domain.runtime import AgentIdentity, RuntimeSession
from .db import Database
from .mapping import dt, txt


class IdentityStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def load_identity(self) -> AgentIdentity | None:
        row = self._db.query_one("SELECT * FROM agent_identity LIMIT 1")
        return None if row is None else self._to_identity(row)

    def insert_identity(self, identity: AgentIdentity) -> None:
        self._db.execute(
            """INSERT INTO agent_identity (
                agent_id, created_at, lifecycle_state, current_runtime_session_id,
                last_started_at, last_active_at, last_clean_suspend_at, last_resume_at,
                last_heartbeat_at, total_active_seconds, last_runtime_exit_kind
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identity.agent_id,
                txt(identity.created_at),
                identity.lifecycle_state.value,
                identity.current_runtime_session_id,
                txt(identity.last_started_at),
                txt(identity.last_active_at),
                txt(identity.last_clean_suspend_at),
                txt(identity.last_resume_at),
                txt(identity.last_heartbeat_at),
                identity.total_active_seconds,
                identity.last_runtime_exit_kind.value if identity.last_runtime_exit_kind else None,
            ),
        )

    def update_identity(self, identity: AgentIdentity) -> None:
        self._db.execute(
            """UPDATE agent_identity SET
                lifecycle_state=?, current_runtime_session_id=?, last_started_at=?,
                last_active_at=?, last_clean_suspend_at=?, last_resume_at=?, last_heartbeat_at=?,
                total_active_seconds=?, last_runtime_exit_kind=?
            WHERE agent_id=?""",
            (
                identity.lifecycle_state.value,
                identity.current_runtime_session_id,
                txt(identity.last_started_at),
                txt(identity.last_active_at),
                txt(identity.last_clean_suspend_at),
                txt(identity.last_resume_at),
                txt(identity.last_heartbeat_at),
                identity.total_active_seconds,
                identity.last_runtime_exit_kind.value if identity.last_runtime_exit_kind else None,
                identity.agent_id,
            ),
        )

    def insert_session(self, session: RuntimeSession) -> None:
        self._db.execute(
            """INSERT INTO runtime_sessions (
                id, agent_id, started_at, scheduler_generation, closed_at, exit_kind
            ) VALUES (?,?,?,?,?,?)""",
            (
                session.id,
                session.agent_id,
                txt(session.started_at),
                session.scheduler_generation,
                txt(session.closed_at),
                session.exit_kind.value if session.exit_kind else None,
            ),
        )

    def close_session(self, session_id: str, closed_at, exit_kind: ExitKind) -> None:
        self._db.execute(
            "UPDATE runtime_sessions SET closed_at=?, exit_kind=? WHERE id=?",
            (txt(closed_at), exit_kind.value, session_id),
        )

    def update_session_generation(self, session_id: str, generation: int) -> None:
        self._db.execute(
            "UPDATE runtime_sessions SET scheduler_generation=? WHERE id=?",
            (generation, session_id),
        )

    def latest_session(self) -> RuntimeSession | None:
        row = self._db.query_one(
            "SELECT * FROM runtime_sessions ORDER BY started_at DESC LIMIT 1"
        )
        return None if row is None else self._to_session(row)

    @staticmethod
    def _to_identity(row: sqlite3.Row) -> AgentIdentity:
        exit_kind = row["last_runtime_exit_kind"]
        return AgentIdentity(
            agent_id=row["agent_id"],
            created_at=dt(row["created_at"]),
            lifecycle_state=LifecycleState(row["lifecycle_state"]),
            current_runtime_session_id=row["current_runtime_session_id"],
            last_started_at=dt(row["last_started_at"]),
            last_active_at=dt(row["last_active_at"]),
            last_clean_suspend_at=dt(row["last_clean_suspend_at"]),
            last_resume_at=dt(row["last_resume_at"]),
            last_heartbeat_at=dt(row["last_heartbeat_at"]),
            total_active_seconds=row["total_active_seconds"],
            last_runtime_exit_kind=ExitKind(exit_kind) if exit_kind else None,
        )

    @staticmethod
    def _to_session(row: sqlite3.Row) -> RuntimeSession:
        exit_kind = row["exit_kind"]
        return RuntimeSession(
            id=row["id"],
            agent_id=row["agent_id"],
            started_at=dt(row["started_at"]),
            scheduler_generation=row["scheduler_generation"],
            closed_at=dt(row["closed_at"]),
            exit_kind=ExitKind(exit_kind) if exit_kind else None,
        )
