"""Adversarial: the agent doesn't nag — a just-expressed thought isn't resurfaced (DESIGN 6, 11.3, 12.7).

Fast-mode testing surfaced the agent repeating itself: it spoke about the same memory (and the
topic enriched from it, same text) on back-to-back wakes. This locks in the fix: a proactively
expressed candidate is suppressed from re-selection for `memory.repeat_suppression`, an enriched
memory is represented only by its topic, and a topic inherits its source memory's suppression.
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


def test_same_topic_is_not_resurfaced_within_window(tmp_path, clock):
    h = Harness(tmp_path, _config("1h"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    # A second wake in-window must not speak about the same topic again.
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.close()


def test_memory_and_its_enriched_topic_are_not_both_spoken(tmp_path, clock):
    h = Harness(tmp_path, _config("1h"), clock)
    _seed_memory(h)
    h.wake(); h.run_all_pending()  # speaks about the memory, enriches it into a topic
    assert _proactive_count(h) == 1
    # The enriched memory is excluded, and its topic inherits the memory's suppression.
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    h.close()


def test_suppression_expires_after_window(tmp_path, clock):
    h = Harness(tmp_path, _config("30s"), clock)
    _seed_topic(h)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 1
    # Past the repeat-suppression window the thought may resurface (revisit later, DESIGN 13.5).
    clock.advance(60)
    h.wake(); h.run_all_pending()
    assert _proactive_count(h) == 2
    h.close()
