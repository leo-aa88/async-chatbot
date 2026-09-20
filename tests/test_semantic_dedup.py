"""Semantic topic de-duplication at enrichment (embedding arc, PR 3).

Catches paraphrases the lexical check (#25) misses, via cosine between the new memory's embedding
and an existing topic's inherited (source-memory) embedding — while keeping the polarity veto and a
very high threshold so related-but-distinct topics do NOT merge.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.proposals import parse_decision
from aca.domain.state import ProvisionalMemory
from aca.reducer.apply_proposals import apply_proposals


def _seed_memory(h: Harness, text: str, vector: list[float], *, model="m", salience=0.9) -> str:
    now = h.clock.now_utc()
    mid = ids.new_id(ids.PROVISIONAL_MEMORY)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e", text=text, activation=0.9, salience=salience,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))
        h.stores.memory.insert_embedding(ids.new_id(ids.EMBEDDING), mid, model, vector, now)
    return mid


def _enrich(h: Harness, mid: str, summary: str) -> None:
    decision = parse_decision({"action": "silence", "proposals": [
        {"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": mid, "topic_summary": summary}]})
    with h.stores.db.transaction():
        apply_proposals(h.ctx, decision.proposals, h.clock.now_utc())


def _cfg():
    # Disable the lexical path so these tests isolate the semantic one.
    return Config.from_mapping({"rng_seed": 1, "memory": {"topic_merge_similarity": 0}})


def test_paraphrase_merges_semantically(tmp_path, clock):
    h = Harness(tmp_path, _cfg(), clock)
    # Same embedding (cosine 1.0 >= 0.94), lexically distinct summaries -> merge, not duplicate.
    _enrich(h, _seed_memory(h, "a", [1.0, 0.0, 0.0]), "Plan to put the runtime on a robot")
    _enrich(h, _seed_memory(h, "b", [1.0, 0.0, 0.0]), "Deploy the agent onto physical hardware")
    topics = h.stores.memory.all_topics()
    assert len(topics) == 1
    assert topics[0].evidence_count == 2
    h.close()


def test_related_but_distinct_do_not_merge(tmp_path, clock):
    h = Harness(tmp_path, _cfg(), clock)
    # cosine ~0.80 (related neighborhood) is below the 0.94 dedup threshold -> stay separate.
    _enrich(h, _seed_memory(h, "a", [1.0, 0.0, 0.0]), "Robot embodiment plans")
    _enrich(h, _seed_memory(h, "b", [0.8, 0.6, 0.0]), "Robot safety constraints")
    assert len(h.stores.memory.all_topics()) == 2
    h.close()


def test_polarity_flip_never_merges_even_at_cosine_one(tmp_path, clock):
    h = Harness(tmp_path, _cfg(), clock)
    _enrich(h, _seed_memory(h, "a", [1.0, 0.0, 0.0]), "User wants to add dark mode")
    _enrich(h, _seed_memory(h, "b", [1.0, 0.0, 0.0]), "User wants to remove dark mode")
    assert len(h.stores.memory.all_topics()) == 2  # identical vectors, but opposite meaning
    h.close()


def test_different_embedding_model_is_not_compared(tmp_path, clock):
    h = Harness(tmp_path, _cfg(), clock)
    # Same vector but different model_version -> not comparable (drift), so no semantic merge.
    _enrich(h, _seed_memory(h, "a", [1.0, 0.0, 0.0], model="m1"), "Runtime on a robot")
    _enrich(h, _seed_memory(h, "b", [1.0, 0.0, 0.0], model="m2"), "Agent on hardware")
    assert len(h.stores.memory.all_topics()) == 2
    h.close()
