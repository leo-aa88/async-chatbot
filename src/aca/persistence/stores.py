"""Store aggregate: bundles the database and typed stores behind one handle.

Grouping the stores keeps the reducer's dependencies explicit (they are passed in, not
imported ad hoc) while letting each store stay small and single-responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .db import Database
from .event_store import EventStore
from .identity_store import IdentityStore
from .memory_store import MemoryStore
from .outbox_store import OutboxStore
from .state_store import StateStore
from .work_store import WorkStore


@dataclass(slots=True)
class Stores:
    """A handle to the database and all typed stores."""

    db: Database
    identity: IdentityStore
    events: EventStore
    state: StateStore
    memory: MemoryStore
    work: WorkStore
    outbox: OutboxStore

    @staticmethod
    def open(path: str | Path) -> Stores:
        db = Database(path)
        return Stores(
            db=db,
            identity=IdentityStore(db),
            events=EventStore(db),
            state=StateStore(db),
            memory=MemoryStore(db),
            work=WorkStore(db),
            outbox=OutboxStore(db),
        )

    def close(self) -> None:
        self.db.close()
