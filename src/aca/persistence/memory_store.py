"""Memory persistence: provisional memories, embeddings, topics, deferred intents (DESIGN 12, 16).

Stored ``activation``/``last_activated_at`` are the materialization point for lazy decay
(DESIGN 12.4); this store never applies decay itself. Retrieval here is cheap and lexical/id
based; semantic ranking is applied by the caller.
"""

from __future__ import annotations

import sqlite3

from ..domain.enums import EnrichmentStatus
from ..domain.state import DeferredIntent, ProvisionalMemory, Topic
from .db import Database
from .mapping import dt, dumps, loads, txt


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- provisional memories ------------------------------------------------------------
    def insert_memory(self, m: ProvisionalMemory) -> None:
        self._db.execute(
            """INSERT INTO provisional_memories (
                id, event_id, text, keywords, activation, salience, decay_rate_per_hour,
                embedding_id, enrichment_status, enrichment_attempts, next_enrichment_after,
                last_enrichment_error, created_at, last_activated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m.id, m.event_id, m.text, dumps(list(m.keywords)), m.activation, m.salience,
                m.decay_rate_per_hour, m.embedding_id, m.enrichment_status.value,
                m.enrichment_attempts, txt(m.next_enrichment_after), m.last_enrichment_error,
                txt(m.created_at), txt(m.last_activated_at),
            ),
        )

    def update_memory(self, m: ProvisionalMemory) -> None:
        self._db.execute(
            """UPDATE provisional_memories SET
                text=?, keywords=?, activation=?, salience=?, decay_rate_per_hour=?,
                embedding_id=?, enrichment_status=?, enrichment_attempts=?,
                next_enrichment_after=?, last_enrichment_error=?, last_activated_at=?
            WHERE id=?""",
            (
                m.text, dumps(list(m.keywords)), m.activation, m.salience, m.decay_rate_per_hour,
                m.embedding_id, m.enrichment_status.value, m.enrichment_attempts,
                txt(m.next_enrichment_after), m.last_enrichment_error, txt(m.last_activated_at),
                m.id,
            ),
        )

    def get_memory(self, memory_id: str) -> ProvisionalMemory | None:
        row = self._db.query_one(
            "SELECT * FROM provisional_memories WHERE id=?", (memory_id,)
        )
        return None if row is None else self._to_memory(row)

    def recent_memories(self, limit: int = 25) -> list[ProvisionalMemory]:
        rows = self._db.query_all(
            "SELECT * FROM provisional_memories ORDER BY last_activated_at DESC LIMIT ?",
            (limit,),
        )
        return [self._to_memory(r) for r in rows]

    def search_memories(self, keyword: str, limit: int = 10) -> list[ProvisionalMemory]:
        rows = self._db.query_all(
            "SELECT * FROM provisional_memories WHERE text LIKE ? "
            "ORDER BY last_activated_at DESC LIMIT ?",
            (f"%{keyword}%", limit),
        )
        return [self._to_memory(r) for r in rows]

    # --- embeddings ----------------------------------------------------------------------
    def insert_embedding(
        self, embedding_id: str, memory_id: str, model_version: str, vector: list[float], at
    ) -> None:
        self._db.execute(
            """INSERT OR IGNORE INTO embeddings (
                id, provisional_memory_id, model_version, vector, created_at
            ) VALUES (?,?,?,?,?)""",
            (embedding_id, memory_id, model_version, dumps(vector), txt(at)),
        )

    def get_embedding_vector(self, embedding_id: str) -> list[float] | None:
        row = self._db.query_one("SELECT vector FROM embeddings WHERE id=?", (embedding_id,))
        return None if row is None else loads(row["vector"], [])

    def embeddings_for_memories(self, memory_ids: list[str]) -> list[tuple[str, list[float]]]:
        """Return ``(model_version, vector)`` for each given memory that has an embedding.

        Ids without an embedding (e.g. topic candidates, or a memory not yet embedded) are simply
        absent. Callers must group by model_version before comparing — cosine across models is
        meaningless (drift).
        """
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self._db.query_all(
            f"SELECT model_version, vector FROM embeddings WHERE provisional_memory_id IN ({placeholders})",
            tuple(memory_ids),
        )
        return [(r["model_version"], loads(r["vector"], [])) for r in rows]

    # --- topics --------------------------------------------------------------------------
    def insert_topic(self, t: Topic) -> None:
        self._db.execute(
            """INSERT INTO topics (
                id, summary, tags, activation, importance, unfinished, decay_rate_per_hour,
                source, source_memory_id, evidence_count, created_at, last_activated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                t.id, t.summary, dumps(list(t.tags)), t.activation, t.importance,
                int(t.unfinished), t.decay_rate_per_hour, t.source, t.source_memory_id,
                t.evidence_count, txt(t.created_at), txt(t.last_activated_at),
            ),
        )

    def update_topic(self, t: Topic) -> None:
        self._db.execute(
            """UPDATE topics SET summary=?, tags=?, activation=?, importance=?, unfinished=?,
                decay_rate_per_hour=?, evidence_count=?, last_activated_at=? WHERE id=?""",
            (
                t.summary, dumps(list(t.tags)), t.activation, t.importance, int(t.unfinished),
                t.decay_rate_per_hour, t.evidence_count, txt(t.last_activated_at), t.id,
            ),
        )

    def get_topic(self, topic_id: str) -> Topic | None:
        row = self._db.query_one("SELECT * FROM topics WHERE id=?", (topic_id,))
        return None if row is None else self._to_topic(row)

    def all_topics(self, limit: int = 50) -> list[Topic]:
        rows = self._db.query_all(
            "SELECT * FROM topics ORDER BY last_activated_at DESC LIMIT ?", (limit,)
        )
        return [self._to_topic(r) for r in rows]

    def count_unfinished_topics(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS n FROM topics WHERE unfinished=1")
        return int(row["n"]) if row else 0

    # --- deferred intents ----------------------------------------------------------------
    def insert_intent(self, i: DeferredIntent) -> None:
        self._db.execute(
            """INSERT INTO deferred_intents (
                id, topic_id, provisional_memory_id, intent, activation, decay_rate_per_hour,
                created_at, last_activated_at, expires_at, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                i.id, i.topic_id, i.provisional_memory_id, i.intent, i.activation,
                i.decay_rate_per_hour, txt(i.created_at), txt(i.last_activated_at),
                txt(i.expires_at), i.status,
            ),
        )

    def update_intent(self, i: DeferredIntent) -> None:
        self._db.execute(
            """UPDATE deferred_intents SET intent=?, activation=?, decay_rate_per_hour=?,
                last_activated_at=?, expires_at=?, status=? WHERE id=?""",
            (
                i.intent, i.activation, i.decay_rate_per_hour, txt(i.last_activated_at),
                txt(i.expires_at), i.status, i.id,
            ),
        )

    def get_intent(self, intent_id: str) -> DeferredIntent | None:
        row = self._db.query_one("SELECT * FROM deferred_intents WHERE id=?", (intent_id,))
        return None if row is None else self._to_intent(row)

    def pending_intents(self, limit: int = 50) -> list[DeferredIntent]:
        rows = self._db.query_all(
            "SELECT * FROM deferred_intents WHERE status='pending' "
            "ORDER BY last_activated_at DESC LIMIT ?",
            (limit,),
        )
        return [self._to_intent(r) for r in rows]

    def has_pending_intent_for_topic(self, topic_id: str, *, exclude_intent_id: str | None = None) -> bool:
        """Whether any pending deferred intent still points at this topic (an open thread)."""
        row = self._db.query_one(
            "SELECT 1 FROM deferred_intents WHERE status='pending' AND topic_id=? AND id != ? LIMIT 1",
            (topic_id, exclude_intent_id or ""),
        )
        return row is not None

    # --- expression suppression ----------------------------------------------------------
    def record_expression(self, kind: str, candidate_id: str, at) -> None:
        """Record that a candidate was just proactively expressed (upsert, latest wins)."""
        self._db.execute(
            """INSERT INTO expressions (candidate_kind, candidate_id, expressed_at)
            VALUES (?,?,?)
            ON CONFLICT(candidate_kind, candidate_id) DO UPDATE SET expressed_at=excluded.expressed_at""",
            (kind, candidate_id, txt(at)),
        )

    def last_expressed_at(self, kind: str, candidate_id: str):
        row = self._db.query_one(
            "SELECT expressed_at FROM expressions WHERE candidate_kind=? AND candidate_id=?",
            (kind, candidate_id),
        )
        return None if row is None else dt(row["expressed_at"])

    # --- row mappers ---------------------------------------------------------------------
    @staticmethod
    def _to_memory(row: sqlite3.Row) -> ProvisionalMemory:
        return ProvisionalMemory(
            id=row["id"], event_id=row["event_id"], text=row["text"],
            keywords=tuple(loads(row["keywords"], [])), activation=row["activation"],
            salience=row["salience"], decay_rate_per_hour=row["decay_rate_per_hour"],
            embedding_id=row["embedding_id"],
            enrichment_status=EnrichmentStatus(row["enrichment_status"]),
            enrichment_attempts=row["enrichment_attempts"],
            next_enrichment_after=dt(row["next_enrichment_after"]),
            last_enrichment_error=row["last_enrichment_error"],
            created_at=dt(row["created_at"]), last_activated_at=dt(row["last_activated_at"]),
        )

    @staticmethod
    def _to_topic(row: sqlite3.Row) -> Topic:
        return Topic(
            id=row["id"], summary=row["summary"], tags=tuple(loads(row["tags"], [])),
            activation=row["activation"], importance=row["importance"],
            unfinished=bool(row["unfinished"]), decay_rate_per_hour=row["decay_rate_per_hour"],
            source=row["source"], source_memory_id=row["source_memory_id"],
            evidence_count=row["evidence_count"] if "evidence_count" in row.keys() else 1,
            created_at=dt(row["created_at"]), last_activated_at=dt(row["last_activated_at"]),
        )

    @staticmethod
    def _to_intent(row: sqlite3.Row) -> DeferredIntent:
        return DeferredIntent(
            id=row["id"], topic_id=row["topic_id"],
            provisional_memory_id=row["provisional_memory_id"], intent=row["intent"],
            activation=row["activation"], decay_rate_per_hour=row["decay_rate_per_hour"],
            created_at=dt(row["created_at"]), last_activated_at=dt(row["last_activated_at"]),
            expires_at=dt(row["expires_at"]), status=row["status"],
        )
