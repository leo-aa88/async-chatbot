"""Adversarial: autonomous enrichment is bounded and never duplicated (invariant 23, DESIGN 12.8).

Reproduces the runaway-enrichment bug (many wakes enriching the same RAW memory before the first
result commits, each creating a duplicate topic) and locks in the fix: enrichment is claimed
(RAW -> PENDING_ENRICHMENT) at dispatch, applied idempotently by memory identity, and released
to RAW / DO_NOT_ENRICH (after the attempt cap) if a cycle doesn't actually enrich.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.state import ProvisionalMemory
from aca.workers.base import LLMOutput


class SilentLLM:
    """A worker that always declines — never speaks, never enriches."""

    async def run(self, snapshot):
        return LLMOutput(result={"action": "silence"}, tokens_in=1, tokens_out=1)


def _config():
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000, "max_enrichment_attempts": 3},
    })


def _seed_raw(h: Harness, text="machine telos and purpose"):
    now = h.clock.now_utc()
    mid = ids.new_id(ids.PROVISIONAL_MEMORY)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="seed", text=text, activation=0.95, salience=0.95,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))
    return mid


def test_concurrent_wakes_enrich_a_memory_at_most_once(tmp_path, clock):
    h = Harness(tmp_path, _config(), clock)  # DORMANT (no human turns) -> proactive allowed
    _seed_raw(h)
    # Two wakes fire before either result is processed (the concurrency that caused duplicates).
    h.wake()
    h.wake()
    assert len([w for w in h.pending_work() if w.kind.value != "EMBEDDING"]) == 2
    h.run_all_pending()
    topics = [t for t in h.stores.memory.all_topics() if t.summary]
    assert len(topics) == 1  # exactly one topic, not one per wake (invariant 23)
    h.close()


def test_declined_enrichment_is_released_then_capped(tmp_path, clock):
    h = Harness(tmp_path, _config(), clock, llm=SilentLLM())
    mid = _seed_raw(h)
    # Each cycle claims PENDING then, because the worker declines, releases back toward RAW.
    for _ in range(2):
        h.wake()
        h.run_all_pending()
        assert h.stores.memory.get_memory(mid).enrichment_status is EnrichmentStatus.RAW
    # The third release hits the attempt cap and parks it as DO_NOT_ENRICH (no token leak).
    h.wake()
    h.run_all_pending()
    memory = h.stores.memory.get_memory(mid)
    assert memory.enrichment_status is EnrichmentStatus.DO_NOT_ENRICH
    assert memory.enrichment_attempts == 3
    h.close()
