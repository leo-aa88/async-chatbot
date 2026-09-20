"""Topic-summary embedding pipeline (embedding arc, stage 1 — populate only).

Enrichment enqueues an async job to embed the topic *summary* (not its source memory's raw text),
and an EmbeddingResult carrying a topic_id stores it in the topic_embeddings table. Nothing consumes
these vectors yet (dedup/dominance still use the old path) — this stage just populates them.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, WorkKind
from aca.domain.events import EmbeddingResult
from aca.domain.proposals import parse_decision
from aca.domain.state import ProvisionalMemory, Topic
from aca.reducer.apply_proposals import apply_proposals


def test_embedding_result_with_topic_id_stores_topic_embedding(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id="topic_1", summary="a topic", activation=0.8, importance=0.8,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, source_memory_id="m1",
        ))
    h.reduce(EmbeddingResult(
        event_id=ids.new_id(ids.EVENT), timestamp=now, source="w", work_id="w1",
        topic_id="topic_1", embedding_id=ids.new_id(ids.EMBEDDING),
        model_version="m", vector=[0.1, 0.2, 0.3],
    ))
    got = h.stores.memory.topic_embeddings_by_id(["topic_1"])
    assert got["topic_1"] == ("m", [0.1, 0.2, 0.3])
    # It must NOT leak into the memory embeddings table.
    assert h.stores.memory.embeddings_by_memory(["topic_1"]) == {}
    h.close()


def test_enrichment_enqueues_and_embeds_the_summary(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    now = h.clock.now_utc()
    mid = ids.new_id(ids.PROVISIONAL_MEMORY)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e1", text="raw fragment", activation=0.9, salience=0.9,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))
    decision = parse_decision({"action": "silence", "proposals": [
        {"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": mid,
         "topic_summary": "A clean summary of the thought"}]})
    with h.stores.db.transaction():
        apply_proposals(h.ctx, decision.proposals, now)

    topic = h.stores.memory.all_topics()[0]
    # A pending embedding job for the summary was enqueued.
    embedding_jobs = [w for w in h.stores.work.pending() if w.kind is WorkKind.EMBEDDING]
    assert any(w.snapshot.get("topic_id") == topic.id and w.snapshot.get("text") == topic.summary
               for w in embedding_jobs)

    h.run_all_pending()  # runs the fake embedding worker + reduces the result
    got = h.stores.memory.topic_embeddings_by_id([topic.id])
    assert topic.id in got and len(got[topic.id][1]) > 0  # summary embedding now stored
    h.close()
