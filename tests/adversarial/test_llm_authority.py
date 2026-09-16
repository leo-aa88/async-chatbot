"""Adversarial: LLM output is data, never authority (invariants 5, 6, 25).

A hostile or malformed worker result must not: apply an out-of-bounds state delta, execute a
non-whitelisted action, satisfy a mandatory obligation with silence, or be hidden as intentional
silence on parse failure.
"""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, ObligationStatus
from aca.domain.state import ProvisionalMemory, Topic
from aca.workers.base import LLMOutput


class ScriptedLLM:
    def __init__(self, result: dict):
        self._result = result

    async def run(self, snapshot):
        return LLMOutput(result=self._result, tokens_in=10, tokens_out=10)


def _proactive_config():
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
    })


def _seed_memory(h: Harness):
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=ids.new_id(ids.PROVISIONAL_MEMORY), event_id="seed", text="telos of machines",
            activation=0.9, salience=0.9, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))


def test_out_of_bounds_delta_is_clamped(tmp_path, clock):
    worker = ScriptedLLM({
        "action": "speak", "message": "hello",
        "proposals": [{"type": "ADJUST_TOPIC_ACTIVATION", "topic_id": "t1", "delta": 999}],
    })
    h = Harness(tmp_path, _proactive_config(), clock, llm=worker)
    now = clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id="t1", summary="s", activation=0.2, importance=0.5,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0), created_at=now, last_activated_at=now,
        ))
    _seed_memory(h)
    h.wake(); h.run_all_pending()
    # Delta clamped to <= 0.5, so activation cannot jump to ~1000.
    assert h.stores.memory.get_topic("t1").activation <= 0.2 + 0.5 + 1e-9
    h.close()


def test_non_whitelisted_action_has_no_effect(tmp_path, clock):
    worker = ScriptedLLM({"action": "wipe_database"})  # invalid action
    h = Harness(tmp_path, _proactive_config(), clock, llm=worker)
    _seed_memory(h)
    h.wake()
    h.run_all_pending()
    # Parse failed -> no outbound at all (proactive parse failure is silent, DESIGN 22.3).
    assert h.stores.outbox.deliverable() == []
    h.close()


def _mandatory_work_id(h: Harness) -> str:
    return next(w.work_id for w in h.pending_work() if w.kind.value == "LLM_COGNITION")


def test_mandatory_silence_becomes_failure_not_silence(tmp_path, clock):
    worker = ScriptedLLM({"action": "silence"})  # model tries to silently ignore a task
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock, llm=worker)
    h.send_human("Fix this bug now.")
    work_id = _mandatory_work_id(h)
    h.run_all_pending()
    obligation = h.stores.work.obligation_by_work(work_id)
    # Explicitly FAILED, never SATISFIED-by-silence and never left dangling pending.
    assert obligation.status is ObligationStatus.FAILED
    assert all(t["role"] != "agent" for t in h.stores.outbox.recent_turns())
    h.close()


def test_mandatory_parse_failure_marks_obligation_failed(tmp_path, clock):
    worker = ScriptedLLM({"garbage": True})  # no action field -> ValidationError
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock, llm=worker)
    h.send_human("Explain the crash.")
    work_id = _mandatory_work_id(h)
    h.run_all_pending()
    assert h.stores.work.obligation_by_work(work_id).status is ObligationStatus.FAILED
    assert all(t["role"] != "agent" for t in h.stores.outbox.recent_turns())
    h.close()
