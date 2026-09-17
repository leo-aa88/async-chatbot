"""Outbound-message, delivery-attempt, and conversation-turn persistence (DESIGN 6.1, 23.7).

Deciding to speak is not delivery (invariant 37): an outbound item is persisted PENDING and
only becomes a user-visible conversation turn once DELIVERED. ``delivery_key`` is UNIQUE so
retried transport deduplicates (invariant 38).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ..domain.enums import OutboundKind, OutboundStatus
from ..domain.runtime import OutboundMessage
from .db import Database
from .mapping import dt, txt


class OutboxStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def insert_message(self, m: OutboundMessage) -> None:
        self._db.execute(
            """INSERT INTO outbound_messages (
                message_id, delivery_key, action_id, kind, channel, payload, status,
                created_at, expires_at, delivered_at, superseded_by_id, last_delivery_error,
                candidate_kind, candidate_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m.message_id, m.delivery_key, m.action_id, m.kind.value, m.channel, m.payload,
                m.status.value, txt(m.created_at), txt(m.expires_at), txt(m.delivered_at),
                m.superseded_by_id, m.last_delivery_error, m.candidate_kind, m.candidate_id,
            ),
        )

    def get_message(self, message_id: str) -> OutboundMessage | None:
        row = self._db.query_one(
            "SELECT * FROM outbound_messages WHERE message_id=?", (message_id,)
        )
        return None if row is None else self._to_message(row)

    def set_status(
        self,
        message_id: str,
        status: OutboundStatus,
        *,
        delivered_at: datetime | None = None,
        error: str | None = None,
        superseded_by_id: str | None = None,
    ) -> None:
        self._db.execute(
            """UPDATE outbound_messages SET status=?, delivered_at=COALESCE(?, delivered_at),
                last_delivery_error=?, superseded_by_id=COALESCE(?, superseded_by_id)
            WHERE message_id=?""",
            (status.value, txt(delivered_at), error, superseded_by_id, message_id),
        )

    def pending_by_kind(self, kind: OutboundKind) -> list[OutboundMessage]:
        rows = self._db.query_all(
            "SELECT * FROM outbound_messages WHERE kind=? AND status IN (?, ?) "
            "ORDER BY created_at ASC",
            (kind.value, OutboundStatus.PENDING_DELIVERY.value, OutboundStatus.DELIVERING.value),
        )
        return [self._to_message(r) for r in rows]

    def in_flight_proactive_candidates(self) -> set[tuple[str, str]]:
        """(kind, id) of candidates with a still-undelivered proactive item (in-flight dedup).

        Naturally released when the item leaves PENDING/DELIVERING (delivered, expired, failed,
        superseded), so an item that never reached the user does not suppress its candidate.
        """
        rows = self._db.query_all(
            "SELECT candidate_kind, candidate_id FROM outbound_messages "
            "WHERE kind=? AND status IN (?, ?) AND candidate_id IS NOT NULL",
            (
                OutboundKind.PROACTIVE.value,
                OutboundStatus.PENDING_DELIVERY.value,
                OutboundStatus.DELIVERING.value,
            ),
        )
        return {(r["candidate_kind"], r["candidate_id"]) for r in rows}

    def deliverable(self) -> list[OutboundMessage]:
        rows = self._db.query_all(
            "SELECT * FROM outbound_messages WHERE status=? ORDER BY created_at ASC",
            (OutboundStatus.PENDING_DELIVERY.value,),
        )
        return [self._to_message(r) for r in rows]

    def record_attempt(
        self, message_id: str, delivery_key: str, at: datetime, delivered: bool, error: str | None
    ) -> None:
        self._db.execute(
            """INSERT INTO delivery_attempts (
                message_id, delivery_key, attempted_at, delivered, error
            ) VALUES (?,?,?,?,?)""",
            (message_id, delivery_key, txt(at), int(delivered), error),
        )

    # --- conversation turns --------------------------------------------------------------
    def add_human_turn(self, text: str, channel: str, event_id: str, at: datetime) -> None:
        self._db.execute(
            """INSERT INTO conversation_turns (role, text, channel, event_id, created_at)
            VALUES ('human', ?, ?, ?, ?)""",
            (text, channel, event_id, txt(at)),
        )

    def add_agent_turn(self, text: str, channel: str, message_id: str, at: datetime) -> None:
        self._db.execute(
            """INSERT INTO conversation_turns (role, text, channel, message_id, created_at)
            VALUES ('agent', ?, ?, ?, ?)""",
            (text, channel, message_id, txt(at)),
        )

    def recent_turns(self, limit: int = 10) -> list[dict]:
        rows = self._db.query_all(
            "SELECT role, text, channel, created_at FROM conversation_turns "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        turns = [
            {"role": r["role"], "text": r["text"], "channel": r["channel"], "at": r["created_at"]}
            for r in rows
        ]
        turns.reverse()  # chronological order for prompt/display
        return turns

    @staticmethod
    def _to_message(row: sqlite3.Row) -> OutboundMessage:
        return OutboundMessage(
            message_id=row["message_id"], delivery_key=row["delivery_key"],
            action_id=row["action_id"], kind=OutboundKind(row["kind"]), channel=row["channel"],
            payload=row["payload"], status=OutboundStatus(row["status"]),
            created_at=dt(row["created_at"]), expires_at=dt(row["expires_at"]),
            delivered_at=dt(row["delivered_at"]), superseded_by_id=row["superseded_by_id"],
            last_delivery_error=row["last_delivery_error"],
            candidate_kind=row["candidate_kind"], candidate_id=row["candidate_id"],
        )
