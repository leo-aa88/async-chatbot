"""Adversarial: explicit tasks are never intentionally silenced (invariants 11, 12, 13).

A conservative classifier, an exhausted proactive budget, and stochastic social behavior must
never cause a real task to be dropped.
"""

from __future__ import annotations

from aca.config import Config
from aca.domain.enums import OutboundKind
from conftest import Harness


def test_task_creates_obligation_even_with_zero_proactive_budget(tmp_path, clock):
    cfg = Config.from_mapping({
        "rng_seed": 1,
        "budgets": {"proactive_llm_calls_per_day": 0, "proactive_llm_calls_per_hour": 0,
                    "proactive_messages_per_hour": 0},
    })
    h = Harness(tmp_path, cfg, clock)
    h.send_human("Explain this stack trace.")
    llm_work = [w for w in h.pending_work() if w.kind.value == "LLM_COGNITION"]
    assert len(llm_work) == 1  # proactive budget exhaustion does not block a task (inv 11)
    h.run_all_pending()
    msgs = h.stores.outbox.deliverable()
    assert len(msgs) == 1 and msgs[0].kind is OutboundKind.MANDATORY
    assert h.stores.work.pending_obligations() == []  # satisfied
    h.close()


def test_reprompt_after_silence_forces_response(harness: Harness):
    # First an optional turn the agent may stay silent on...
    harness.send_human("lol")
    # ...then a re-prompt. It must take the response-required path (a mandatory work item).
    harness.send_human("?")
    llm_work = [w for w in harness.pending_work() if w.kind.value == "LLM_COGNITION"]
    assert len(llm_work) >= 1
    obligation_events = [
        o for o in [harness.stores.work.obligation_by_work(w.work_id) for w in llm_work]
        if o is not None
    ]
    assert obligation_events, "re-prompt must create a response obligation"


def test_ambiguous_task_like_question_gets_obligation(harness: Harness):
    harness.send_human("did the migration finish running?")
    llm_work = [w for w in harness.pending_work() if w.kind.value == "LLM_COGNITION"]
    assert len(llm_work) == 1
    assert harness.stores.work.obligation_by_work(llm_work[0].work_id) is not None
