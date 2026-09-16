"""Adversarial: durable idempotent ingress and no unbounded recursive cognition.

Covers invariants 14 (durable ingress, effectively-once reduction) and 3/7 (one generative call
per cycle; a result never re-triggers generative cognition).
"""

from __future__ import annotations

from datetime import timezone

from aca import ids
from aca.domain.events import HumanMessage
from conftest import Harness

UTC = timezone.utc


def test_duplicate_event_id_is_not_a_second_turn(harness: Harness):
    eid = ids.new_id(ids.EVENT)
    r1 = harness.send_human("Explain the traceback.", event_id=eid)
    # Re-send the SAME event_id (client retry after a lost ACK).
    ev = HumanMessage(event_id=eid, timestamp=harness.clock.now_utc(), text="Explain the traceback.")
    with harness.stores.db.transaction():
        newly = harness.stores.events.accept(ev, harness.clock.now_utc())
    assert newly is False  # ingress deduplicated
    human_turns = [t for t in harness.stores.outbox.recent_turns(limit=50) if t["role"] == "human"]
    assert len(human_turns) == 1


def test_one_cycle_dispatches_at_most_one_generative_call(harness: Harness):
    harness.send_human("Rewrite this function please.")
    llm_work = [w for w in harness.pending_work() if w.kind.value == "LLM_COGNITION"]
    assert len(llm_work) == 1  # invariant 3


def test_llm_result_does_not_spawn_more_generative_work(harness: Harness):
    harness.send_human("Fix the failing test.")
    # Run embedding + LLM; capture generative work created strictly by result processing.
    for w in list(harness.pending_work()):
        before = _generative_ids(harness)
        harness.run_work(w.work_id)
        after = _generative_ids(harness)
        # Processing a result must not create NEW generative work (invariant 7).
        assert after <= before | {w.work_id}


def _generative_ids(h: Harness) -> set[str]:
    return {
        w.work_id
        for w in h.stores.work.pending()
        if w.kind.value in ("LLM_COGNITION", "LLM_ENRICHMENT")
    }
