"""Adversarial: proactive/optional results are revalidated against current state before speaking.

Covers the review's finding #1 and invariants 8/9 (DESIGN §16.2, §22.2): a proactive SPEAK is
dropped — never delivered — when a newer human message arrived while the call was in flight, or
when the specific candidate it was about was resolved/removed. The generic budget/mode/quiet
gates are not sufficient; the *candidate* must still make sense.
"""

from __future__ import annotations

from datetime import timedelta

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import DeferredIntent, ProvisionalMemory
from aca.workers.base import LLMOutput


class AlwaysSpeak:
    """A worker that always wants to speak about the candidate (the dangerous case)."""

    async def run(self, snapshot):
        candidate = (snapshot.context.get("source") or {}).get("candidate") or {}
        subject = candidate.get("summary") or candidate.get("intent") or candidate.get("text", "x")
        return LLMOutput(result={"action": "speak", "message": f"About: {subject}"}, tokens_in=10, tokens_out=10)


def _proactive_config():
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 5, "proactive_llm_calls_per_hour": 5,
                    "proactive_llm_calls_per_day": 50},
    })


def _seed_memory(h: Harness, text="return to fluid mechanics"):
    now = h.clock.now_utc()
    mid = ids.new_id(ids.PROVISIONAL_MEMORY)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="seed", text=text, activation=0.95, salience=0.95,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))
    return mid


def test_proactive_dropped_when_newer_human_message_arrives_mid_flight(tmp_path, clock):
    h = Harness(tmp_path, _proactive_config(), clock, llm=AlwaysSpeak())
    _seed_memory(h)
    h.wake()  # selects the memory, dispatches a proactive LLM call
    assert len(h.pending_work()) == 1

    # A human speaks while the worker is "in flight" — the conversation has moved on.
    clock.advance(1)
    h.send_human("hey, different subject entirely")

    # Now the proactive result comes back wanting to SPEAK. It must be dropped, not delivered.
    proactive_work = next(w for w in h.stores.work.pending() if w.kind.value != "EMBEDDING")
    h.run_work(proactive_work.work_id)
    proactive = [m for m in h.stores.outbox.deliverable() if m.kind is OutboundKind.PROACTIVE]
    assert proactive == []  # superseded by the newer human turn (invariant 8/9)


def test_proactive_dropped_when_candidate_intent_resolved(tmp_path, clock):
    h = Harness(tmp_path, _proactive_config(), clock, llm=AlwaysSpeak())
    now = clock.now_utc()
    intent_id = ids.new_id(ids.DEFERRED_INTENT)
    with h.stores.db.transaction():
        h.stores.memory.insert_intent(DeferredIntent(
            id=intent_id, intent="revisit the telos question", activation=0.95,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0), created_at=now,
            last_activated_at=now, expires_at=now + timedelta(hours=48), status="pending",
        ))
    h.wake()
    proactive_work = next(w for w in h.stores.work.pending() if w.kind.value != "EMBEDDING")

    # The user resolves the intent's topic before the result lands.
    with h.stores.db.transaction():
        intent = h.stores.memory.get_intent(intent_id)
        from dataclasses import replace
        h.stores.memory.update_intent(replace(intent, status="resolved"))

    h.run_work(proactive_work.work_id)
    proactive = [m for m in h.stores.outbox.deliverable() if m.kind is OutboundKind.PROACTIVE]
    assert proactive == []  # candidate no longer valid -> dropped, never regenerated inline


def test_proactive_still_speaks_when_nothing_superseded(tmp_path, clock):
    # Control: with no invalidation, the same worker DOES speak — the drop above is real, not blanket.
    h = Harness(tmp_path, _proactive_config(), clock, llm=AlwaysSpeak())
    _seed_memory(h)
    h.wake()
    h.run_all_pending()
    proactive = [m for m in h.stores.outbox.deliverable() if m.kind is OutboundKind.PROACTIVE]
    assert len(proactive) == 1
