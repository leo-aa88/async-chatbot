"""Integration tests for the reducer over the full mandatory/optional/proactive paths."""

from __future__ import annotations

import pytest
from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind, OutboundStatus
from aca.domain.state import ProvisionalMemory


def _insert_strong_memory(h: Harness, text="return to fluid mechanics", salience=0.9):
    now = h.clock.now_utc()
    mem = ProvisionalMemory(
        id=ids.new_id(ids.PROVISIONAL_MEMORY), event_id="seed", text=text,
        activation=0.9, salience=salience,
        decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
        created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
    )
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(mem)
    return mem.id


def test_mandatory_task_creates_obligation_and_reply(harness: Harness):
    harness.send_human("Explain this stack trace.")
    # embedding + one LLM cognition work item
    kinds = sorted(w.kind.value for w in harness.pending_work())
    assert "LLM_COGNITION" in kinds and "EMBEDDING" in kinds
    harness.run_all_pending()
    obligations = harness.stores.work.pending_obligations()
    assert obligations == []  # satisfied
    pending_msgs = harness.stores.outbox.deliverable()
    assert len(pending_msgs) == 1 and pending_msgs[0].kind is OutboundKind.MANDATORY


def test_trace_records_speak_action_for_mandatory(harness: Harness):
    # `aca logs` must answer "did it speak?": mandatory/reactive traces record the outcome,
    # not just the dispatch (DESIGN 26). Previously `action` stayed null for these cycles.
    harness.send_human("Explain this stack trace.")
    harness.run_all_pending()
    traces = harness.stores.work.recent_traces()
    mandatory = next(t for t in traces if t.trigger == "HumanMessage" and t.llm_called)
    assert mandatory.action == "speak"


def test_trace_records_silence_action_when_silenced_before_any_llm_call(harness: Harness):
    # The common case: the fast loop silences a turn without an LLM call (DESIGN 9.1, 19). Its
    # trace must still say "silence", not null — otherwise `aca logs` can't tell an intentional
    # silence apart from an unfinalized trace (DESIGN 26).
    harness.send_human("lol")  # low-substance -> silenced by the reducer, no LLM dispatched
    trace = harness.stores.work.recent_traces()[0]
    assert trace.llm_called is False
    assert trace.action == "silence"
    assert trace.notes == "silence_low_substance"


def test_delivery_makes_turn_visible(harness: Harness):
    harness.send_human("Explain this error.")
    harness.run_all_pending()
    message = harness.stores.outbox.deliverable()[0]
    assert message.status is OutboundStatus.PENDING_DELIVERY
    harness.deliver(message.message_id, delivered=True)
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.DELIVERED
    roles = [t["role"] for t in harness.stores.outbox.recent_turns()]
    assert roles == ["human", "agent"]


def test_trivial_message_skips_embedding_but_is_persisted(harness: Harness):
    harness.send_human("lol")
    assert harness.pending_work() == []  # no embedding work for trivial content
    assert harness.stores.outbox.recent_turns()[-1]["text"] == "lol"  # still a durable turn


@pytest.mark.parametrize("text", ["t", "e", "ok", "hi"])
def test_low_substance_turn_is_persisted_without_a_reply(harness: Harness, text):
    # Stray tokens / acknowledgements must not earn an earnest reactive reply (DESIGN 13.6, 15).
    harness.send_human(text)
    assert [w for w in harness.pending_work() if w.kind.value == "LLM_COGNITION"] == []
    harness.run_all_pending()  # runs any embedding work; must not produce an agent turn
    assert all(t["role"] != "agent" for t in harness.stores.outbox.recent_turns())
    assert harness.stores.outbox.recent_turns()[-1]["text"] == text  # human turn persisted


@pytest.mark.parametrize("text", ["n", "y", "5"])
def test_single_char_answer_stays_semantically_retrievable(harness: Harness, text):
    # A lone meaningful answer ("n"/"y"/"5") is silent but must keep a provisional memory so it
    # stays retrievable — no cold-retrieval failure (DESIGN 12.2). Only the ack allowlist skips it.
    harness.send_human(text)
    assert any(w.kind.value == "EMBEDDING" for w in harness.pending_work())
    memories = [m for m in harness.stores.memory.recent_memories() if m.text == text]
    assert len(memories) == 1  # provisional memory created (keyword/FTS + embedding addressable)


def test_proactive_speak_path(tmp_path, clock):
    # Config that makes a single strong candidate dominate NOTHING deterministically.
    cfg = Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
    })
    h = Harness(tmp_path, cfg, clock)
    _insert_strong_memory(h)
    h.wake()
    dispatched = h.pending_work()
    assert len(dispatched) == 1  # exactly one proactive generative call (invariant 3)
    h.run_all_pending()
    proactive = [m for m in h.stores.outbox.deliverable() if m.kind is OutboundKind.PROACTIVE]
    assert len(proactive) == 1
    h.deliver(proactive[0].message_id, delivered=True)
    model = h.stores.state.load_self_model()
    assert model.recent_proactive_messages == 1
    assert model.last_delivered_proactive_at is not None  # cooldown anchored to delivery
    h.close()


def test_proactive_supersedes_older_pending(tmp_path, clock):
    cfg = Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 5, "proactive_llm_calls_per_hour": 5},
    })
    h = Harness(tmp_path, cfg, clock)
    _insert_strong_memory(h)
    h.wake(); h.run_all_pending()
    h.wake(); h.run_all_pending()
    proactive = h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)
    # At most one still-pending proactive item per channel (DESIGN 6.1).
    assert len(proactive) <= 1
    h.close()
