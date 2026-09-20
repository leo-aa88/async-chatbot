"""Semantic topic de-duplication on summary vectors (embedding arc, stage 2).

Paraphrase dedup runs POST-embedding: when a topic's summary vector lands, it merges into a
near-duplicate existing topic (summary cosine >= topic_dedup_cosine), vetoing polarity flips and
comparing only same-model vectors. The enrich-time lexical path still catches near-identical
wording synchronously.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.events import EmbeddingResult
from aca.domain.proposals import parse_decision
from aca.domain.state import ProvisionalMemory, Topic
from aca.reducer.apply_proposals import apply_proposals

_RATE = half_life_to_rate_per_hour(24.0)


def _topic(h: Harness, tid: str, summary: str) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary=summary, activation=0.8, importance=0.8, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, source_memory_id=None,
        ))


def _land_embedding(h: Harness, tid: str, vector: list[float], model: str = "m") -> None:
    """Simulate the topic's summary embedding arriving (triggers the post-embedding merge)."""
    h.reduce(EmbeddingResult(
        event_id=ids.new_id(ids.EVENT), timestamp=h.clock.now_utc(), source="w",
        work_id=ids.new_id(ids.WORK_ITEM), topic_id=tid,
        embedding_id=ids.new_id(ids.EMBEDDING), model_version=model, vector=vector,
    ))


def test_paraphrase_merges_on_summary_vectors(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t1", "Plan to put the runtime on a robot")
    _topic(h, "t2", "Deploy the agent onto physical hardware")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0])
    _land_embedding(h, "t2", [1.0, 0.0, 0.0])  # cosine 1.0 >= 0.94 -> t2 merges into t1
    topics = h.stores.memory.all_topics()
    assert [t.id for t in topics] == ["t1"]
    assert topics[0].evidence_count == 2
    assert h.stores.memory.get_topic("t2") is None  # duplicate deleted
    h.close()


def test_related_but_distinct_do_not_merge(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t1", "Robot embodiment plans")
    _topic(h, "t2", "Robot safety constraints")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0])
    _land_embedding(h, "t2", [0.8, 0.6, 0.0])  # cosine 0.8 < 0.94 -> stay separate
    assert len(h.stores.memory.all_topics()) == 2
    h.close()


def test_polarity_flip_never_merges_even_at_cosine_one(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t1", "User wants to add dark mode")
    _topic(h, "t2", "User wants to remove dark mode")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0])
    _land_embedding(h, "t2", [1.0, 0.0, 0.0])  # identical vectors, opposite meaning
    assert len(h.stores.memory.all_topics()) == 2
    h.close()


def test_different_embedding_model_not_compared(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t1", "Runtime on a robot")
    _topic(h, "t2", "Agent on hardware")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0], model="m1")
    _land_embedding(h, "t2", [1.0, 0.0, 0.0], model="m2")  # same vector, different model -> no merge
    assert len(h.stores.memory.all_topics()) == 2
    h.close()


def test_merged_topic_absorbs_deferred_intents(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t1", "Ship the runtime to a robot")
    _topic(h, "t2", "Move the agent onto hardware")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0])
    # An intent points at t2 before it merges away; it must be repointed to t1.
    decision = parse_decision({"action": "silence", "proposals": [
        {"type": "CREATE_DEFERRED_INTENT", "intent": "follow up", "topic_id": "t2"}]})
    with h.stores.db.transaction():
        apply_proposals(h.ctx, decision.proposals, h.clock.now_utc())
    _land_embedding(h, "t2", [1.0, 0.0, 0.0])  # t2 merges into t1
    assert h.stores.memory.get_topic("t2") is None
    intents = h.stores.memory.pending_intents()
    assert intents and all(i.topic_id == "t1" for i in intents)  # repointed
    assert h.stores.memory.get_topic("t1").unfinished is True  # open thread carried over
    h.close()


def test_topic_dedup_cosine_zero_disables_post_embedding_merge(tmp_path, clock):
    # The semantic merge is off when the threshold is 0, even for identical summary vectors.
    h = Harness(tmp_path, Config.from_mapping(
        {"rng_seed": 1, "memory": {"topic_dedup_cosine": 0}}), clock)
    _topic(h, "t1", "Plan to put the runtime on a robot")
    _topic(h, "t2", "Deploy the agent onto physical hardware")
    _land_embedding(h, "t1", [1.0, 0.0, 0.0])
    _land_embedding(h, "t2", [1.0, 0.0, 0.0])
    assert len(h.stores.memory.all_topics()) == 2  # merging disabled
    h.close()


def test_enrich_time_lexical_path_still_merges(tmp_path, clock):
    # Near-identical wording still merges synchronously at enrichment, no embedding needed.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)

    def enrich(summary: str) -> None:
        now = h.clock.now_utc()
        mid = ids.new_id(ids.PROVISIONAL_MEMORY)
        with h.stores.db.transaction():
            h.stores.memory.insert_memory(ProvisionalMemory(
                id=mid, event_id="e", text="t", activation=0.9, salience=0.9,
                decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
                enrichment_status=EnrichmentStatus.RAW,
            ))
        decision = parse_decision({"action": "silence", "proposals": [
            {"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": mid,
             "topic_summary": summary}]})
        with h.stores.db.transaction():
            apply_proposals(h.ctx, decision.proposals, now)

    enrich("Robot safety constraints and limits")
    enrich("Robot safety constraints and limits")
    assert len(h.stores.memory.all_topics()) == 1
    h.close()
