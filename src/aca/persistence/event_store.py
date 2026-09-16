"""Durable inbound event log / inbox (DESIGN 8, 23.8).

``accept`` is idempotent by ``event_id``: re-sending the same id returns the existing row
rather than creating a second turn (invariant 14). The service ACKs only after ``accept``
commits. ``unreduced`` events are recovered on startup and reduced effectively once.
"""

from __future__ import annotations

from datetime import datetime

from ..domain.events import Event, deserialize, serialize
from .db import Database
from .mapping import dt, dumps, loads, txt


class EventStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def accept(self, event: Event, accepted_at: datetime) -> bool:
        """Durably insert an event. Returns ``True`` if newly accepted, ``False`` if duplicate."""
        envelope = serialize(event)
        cur = self._db.execute(
            """INSERT OR IGNORE INTO events (
                event_id, type, source, payload, timestamp, accepted_at, reduced_at
            ) VALUES (?,?,?,?,?,?,NULL)""",
            (
                envelope["event_id"],
                envelope["type"],
                envelope["source"],
                dumps(envelope["payload"]),
                envelope["timestamp"],
                txt(accepted_at),
            ),
        )
        return cur.rowcount == 1

    def is_accepted(self, event_id: str) -> bool:
        return self._db.query_one("SELECT 1 FROM events WHERE event_id=?", (event_id,)) is not None

    def is_reduced(self, event_id: str) -> bool:
        row = self._db.query_one(
            "SELECT reduced_at FROM events WHERE event_id=?", (event_id,)
        )
        return row is not None and row["reduced_at"] is not None

    def mark_reduced(self, event_id: str, reduced_at: datetime) -> None:
        self._db.execute(
            "UPDATE events SET reduced_at=? WHERE event_id=?", (txt(reduced_at), event_id)
        )

    def unreduced(self) -> list[Event]:
        """Accepted-but-not-yet-reduced events, oldest first (DESIGN 23.8 recovery order)."""
        rows = self._db.query_all(
            "SELECT * FROM events WHERE reduced_at IS NULL ORDER BY accepted_at ASC"
        )
        return [self._to_event(row) for row in rows]

    @staticmethod
    def _to_event(row) -> Event:
        return deserialize(
            {
                "event_id": row["event_id"],
                "type": row["type"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "payload": loads(row["payload"], {}),
            }
        )

    def get_timestamp(self, event_id: str) -> datetime | None:
        row = self._db.query_one("SELECT timestamp FROM events WHERE event_id=?", (event_id,))
        return None if row is None else dt(row["timestamp"])
