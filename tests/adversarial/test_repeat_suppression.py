"""Adversarial: the agent doesn't nag — but suppression is anchored to DELIVERY, not the decision.

Repeat-suppression must behave like proactive cooldown (DESIGN 6.1, 29.7): only a message the
user actually received suppresses its candidate. An item that was decided-but-never-delivered
(expired/failed) must not make the agent act as if it already said something. While an item is
still in-flight (pending delivery) its candidate is excluded to avoid double-sending, and that
exclusion is released the moment the item leaves PENDING.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import ProvisionalMemory, Topic


def _config(repeat="1h"):
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"proactive_cooldown": "0s"},
        "memory": {"repeat_suppression": repeat},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _proactive_count(h: Harness) -> int:
    return h.stores.db.query_one(
        "SELECT COUNT(*) AS n FROM outbound_messages WHERE kind=?", (OutboundKind.PROACTIVE.value,)
    )["n"]


def _latest_proactive_id(h: Harness) -> str:
    return h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)[-1].message_id


def _seed_topic(h: Harness):
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=ids.new_id(ids.TOPIC), summary="returning to fluid mechanics", activation=0.9,
            importance=0.8, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, unfinished=True,
        ))


def _seed_memory(h: Harness):
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=ids.new_id(ids.PROVISIONAL_MEMORY), event_id="seed", text="the machine telos idea",
            activation=0.95, salience=0.95, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))


def test_in_flight_proactive_is_not_duplicated(tmp_path, clock):
    # While a proactive item is pending delivery, its candidate is excluded (no double-send)...
    h = Harness(tmp_path, _config("1h"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.close()


def test_delivered_proactive_is_suppressed_within_window(tmp_path, clock):
    # ...and once delivered, it stays suppressed for the repeat-suppression window.
    h = Harness(tmp_path, _config("1h"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    h.deliver(_latest_proactive_id(h), delivered=True)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.close()


def test_expired_proactive_does_not_suppress(tmp_path, clock):
    # The review's repro: an item decided-but-never-delivered must NOT suppress its candidate.
    h = Harness(tmp_path, _config("6h"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    # Simulate the item expiring without ever reaching a client (the expired:* delivery path).
    h.deliver(_latest_proactive_id(h), delivered=False, error="expired:ttl")
    # The topic never reached the user, so it must be selectable again immediately.
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 2
    h.close()


def test_memory_and_its_enriched_topic_are_not_both_spoken(tmp_path, clock):
    h = Harness(tmp_path, _config("1h"), clock)
    _seed_memory(h)
    h.wake(); h.run_all_pending()  # speaks about the memory (in-flight), enriches into a topic
    assert _proactive_count(h) == 1
    # Enriched memory is excluded structurally; its topic inherits the in-flight suppression.
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.close()


def test_suppression_expires_after_window(tmp_path, clock):
    h = Harness(tmp_path, _config("30s"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    h.deliver(_latest_proactive_id(h), delivered=True)
    # Past the repeat-suppression window the thought may resurface (revisit later, DESIGN 13.5).
    clock.advance(60)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 2
    h.close()
