"""Lexical topic de-duplication at enrichment time (DESIGN 12.3).

A near-duplicate summary reinforces the existing topic (and counts the evidence) instead of
spawning a parallel topic; distinct summaries still create new topics; the merge can be disabled.
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


def _seed_memory(h: Harness, salience: float = 0.9) -> str:
    now = h.clock.now_utc()
    mid = ids.new_id(ids.PROVISIONAL_MEMORY)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e", text="t", activation=0.9, salience=salience,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))
    return mid


def _enrich(h: Harness, mid: str, summary: str) -> None:
    decision = parse_decision({"action": "silence", "proposals": [
        {"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": mid, "topic_summary": summary}]})
    with h.stores.db.transaction():
        apply_proposals(h.ctx, decision.proposals, h.clock.now_utc())


def test_near_duplicate_summary_reinforces_instead_of_duplicating(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _enrich(h, _seed_memory(h), "AI chatbot experiment logs with silence scenarios")
    _enrich(h, _seed_memory(h), "AI chatbot experiment logs with silence and scenarios")
    topics = h.stores.memory.all_topics()
    assert len(topics) == 1  # merged, not duplicated
    assert topics[0].evidence_count == 2
    h.close()


def test_distinct_summaries_create_separate_topics(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _enrich(h, _seed_memory(h), "Future robot embodiment and unexpected text-to-speech")
    _enrich(h, _seed_memory(h), "Reducing sycophancy through non-mandatory responses")
    assert len(h.stores.memory.all_topics()) == 2
    h.close()


def test_opposite_meaning_summaries_are_not_merged(tmp_path, clock):
    # High lexical similarity but reversed polarity ("add" vs "remove") must stay separate topics,
    # or a change-of-mind would be folded in as confirming evidence for the original.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _enrich(h, _seed_memory(h), "User wants to add dark mode to the settings page")
    _enrich(h, _seed_memory(h), "User wants to remove dark mode from the settings page")
    topics = h.stores.memory.all_topics()
    assert len(topics) == 2
    assert all(t.evidence_count == 1 for t in topics)
    h.close()


def test_merge_can_be_disabled(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping(
        {"rng_seed": 1, "memory": {"topic_merge_similarity": 0}}), clock)
    _enrich(h, _seed_memory(h), "identical summary text")
    _enrich(h, _seed_memory(h), "identical summary text")
    assert len(h.stores.memory.all_topics()) == 2  # merging off -> duplicates allowed
    h.close()
