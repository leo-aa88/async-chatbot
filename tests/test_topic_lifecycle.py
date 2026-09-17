"""Topic open-thread (``unfinished``) lifecycle (DESIGN 12.7, 16).

A topic is "unfinished" only while a deferred intent keeps a thread open on it — not by default.
This keeps unfinishedness (which boosts activation) from being a permanent "remember this" bit.
"""

from __future__ import annotations

from conftest import Harness

from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.proposals import parse_decision
from aca.domain.state import Topic
from aca.reducer.apply_proposals import apply_proposals


def _seed_topic(h: Harness, tid: str = "topic_1") -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary="ACA architecture", activation=0.8, importance=0.8,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, unfinished=False,
        ))


def _apply(h: Harness, proposals: list[dict]) -> None:
    decision = parse_decision({"action": "silence", "proposals": proposals})
    with h.stores.db.transaction():
        apply_proposals(h.ctx, decision.proposals, h.clock.now_utc())


def test_intent_opens_topic_and_resolve_closes_it(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _seed_topic(h)
    assert h.stores.memory.get_topic("topic_1").unfinished is False  # a topic is a fact by default
    assert h.stores.memory.count_unfinished_topics() == 0

    _apply(h, [{"type": "CREATE_DEFERRED_INTENT", "intent": "follow up", "topic_id": "topic_1"}])
    assert h.stores.memory.get_topic("topic_1").unfinished is True  # an open thread now
    assert h.stores.memory.count_unfinished_topics() == 1

    intent_id = h.stores.memory.pending_intents()[0].id
    _apply(h, [{"type": "RESOLVE_DEFERRED_INTENT", "intent_id": intent_id}])
    assert h.stores.memory.get_topic("topic_1").unfinished is False  # thread closed
    assert h.stores.memory.count_unfinished_topics() == 0
    h.close()


def test_topic_stays_unfinished_while_another_intent_is_pending(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _seed_topic(h)
    _apply(h, [{"type": "CREATE_DEFERRED_INTENT", "intent": "thread A", "topic_id": "topic_1"}])
    _apply(h, [{"type": "CREATE_DEFERRED_INTENT", "intent": "thread B", "topic_id": "topic_1"}])
    assert h.stores.memory.count_unfinished_topics() == 1

    first = h.stores.memory.pending_intents()[-1].id  # resolve one of the two
    _apply(h, [{"type": "RESOLVE_DEFERRED_INTENT", "intent_id": first}])
    # A second thread is still open, so the topic remains unfinished.
    assert h.stores.memory.get_topic("topic_1").unfinished is True
    h.close()


def test_intent_without_topic_does_not_crash(tmp_path, clock):
    # A memory-only deferred intent (no topic_id) must be a no-op for the topic flag.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _apply(h, [{"type": "CREATE_DEFERRED_INTENT", "intent": "orphan", "provisional_memory_id": "m9"}])
    assert h.stores.memory.count_unfinished_topics() == 0
    h.close()
