"""Discourse-focus gate (DESIGN §34, v0.7): proactive speech must fit the conversation *now*.

While mode is IDLE, a proactive candidate off the current focus subject (ORPHAN) is suppressed;
CONTINUE/BRIDGE speak. DORMANT is left open so autonomous resurfacing survives. The focus subject
is set by substantive turns, held through acknowledgement trains, and retired after dormancy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, MessageClass
from aca.domain.state import ProvisionalMemory, Topic
from aca.reducer.discourse import is_focus_setting

_RATE = half_life_to_rate_per_hour(24.0)


def _config():
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        # Short ACTIVE window so a 60s-old turn is IDLE (the orphan zone); long IDLE window.
        "conversation": {"active_within": "10s", "idle_within": "1h"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _harness(tmp_path):
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)))


def _set_focus(h: Harness, vector: list[float], *, gap_seconds: float) -> None:
    """Install a focus memory (with an embedding) and set cadence so mode is inferred from `gap`."""
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id="focus_mem", event_id="e", text="the current subject", activation=0.8, salience=0.8,
            decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
            enrichment_status=EnrichmentStatus.ENRICHED,  # keep it out of the candidate pool
        ))
        h.stores.memory.insert_embedding("focus_emb", "focus_mem", "m", vector, now)
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(
            conv.__class__(
                last_human_message_at=now - timedelta(seconds=gap_seconds),
                focus_memory_id="focus_mem",
            )
        )


def _candidate_topic(h: Harness, tid: str, vector: list[float]) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary=tid, activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None,
        ))
        h.stores.memory.insert_topic_embedding(tid, "m", vector, now)


def _wake_note(h: Harness) -> str:
    h.wake()
    return h.stores.work.recent_traces(limit=1)[0].notes


# --- affinity gate (wake-time) -----------------------------------------------------------------

def test_orphan_suppressed_while_idle(tmp_path):
    h = _harness(tmp_path)
    _set_focus(h, [1.0, 0.0, 0.0], gap_seconds=60)  # IDLE (10s < 60s < 1h)
    _candidate_topic(h, "t_off", [0.0, 1.0, 0.0])   # cosine 0 with focus -> ORPHAN
    assert _wake_note(h) == "discourse_orphan"
    h.close()


def test_continue_speaks_while_idle(tmp_path):
    h = _harness(tmp_path)
    _set_focus(h, [1.0, 0.0, 0.0], gap_seconds=60)
    _candidate_topic(h, "t_on", [1.0, 0.0, 0.0])    # cosine 1.0 -> CONTINUE
    assert _wake_note(h) == "proactive_dispatch"
    h.close()


def test_bridge_band_is_independent_of_the_observational_cut(tmp_path):
    # Affinity ~0.6 is below the observational neighbor cut (0.78) but >= discourse_bridge_cosine
    # (0.55): a BRIDGE, which speaks. Proves ORPHAN != the advance-rate "switch" class.
    h = _harness(tmp_path)
    _set_focus(h, [1.0, 0.0, 0.0], gap_seconds=60)
    _candidate_topic(h, "t_bridge", [0.6, 0.8, 0.0])  # cosine 0.6
    assert _wake_note(h) == "proactive_dispatch"
    h.close()


def test_dormant_resurfacing_is_not_gated(tmp_path):
    h = _harness(tmp_path)
    _set_focus(h, [1.0, 0.0, 0.0], gap_seconds=7200)  # 2h -> DORMANT
    _candidate_topic(h, "t_off", [0.0, 1.0, 0.0])     # unrelated, but gate is inactive
    assert _wake_note(h) == "proactive_dispatch"
    h.close()


def test_no_focus_vector_fails_open(tmp_path):
    # IDLE with a focus id whose embedding hasn't landed yet -> gate inactive, candidate speaks.
    h = _harness(tmp_path)
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(seconds=60), focus_memory_id="ghost",
        ))
    _candidate_topic(h, "t_off", [0.0, 1.0, 0.0])
    assert _wake_note(h) == "proactive_dispatch"
    h.close()


def test_pre_outbox_recheck_drops_a_candidate_that_became_orphan(tmp_path):
    # Fail-open at wake (focus id set, vector not landed) -> dispatched output-eligible. The focus
    # embedding then lands unrelated to the candidate; at LLMResult it is ORPHAN -> dropped pre-outbox.
    from aca.domain.enums import OutboundKind
    h = _harness(tmp_path)
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(seconds=60), focus_memory_id="focus_mem"))
        h.stores.memory.insert_memory(ProvisionalMemory(
            id="focus_mem", event_id="e", text="subject", activation=0.8, salience=0.8,
            decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
            enrichment_status=EnrichmentStatus.ENRICHED))
    _candidate_topic(h, "t_off", [0.0, 1.0, 0.0])
    assert _wake_note(h) == "proactive_dispatch"  # fail-open: no focus vector yet -> dispatched

    with h.stores.db.transaction():  # focus vector lands, unrelated to the candidate
        h.stores.memory.insert_embedding("focus_emb", "focus_mem", "m", [1.0, 0.0, 0.0], now)
    h.run_all_pending()
    assert h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE) == []  # not voiced
    assert h.stores.work.recent_traces(limit=1)[0].notes == "pre_outbox:discourse_orphan"
    h.close()


# --- focus lifecycle (via the human handler) ---------------------------------------------------

def test_substantive_turn_sets_focus(tmp_path):
    h = _harness(tmp_path)
    h.send_human("Let's design the robot's balance controller on the ESP32.")
    assert h.stores.state.load_conversation().focus_memory_id is not None
    h.close()


def test_noted_is_a_backchannel_not_a_subject(tmp_path):
    h = _harness(tmp_path)
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(seconds=60), focus_memory_id="subject_X",
        ))  # pre-turn IDLE
    h.send_human("noted")
    assert h.stores.state.load_conversation().focus_memory_id == "subject_X"  # kept, not replaced
    h.close()


def test_backchannel_after_dormancy_retires_the_subject(tmp_path):
    h = _harness(tmp_path)
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(hours=2), focus_memory_id="stale_X",
        ))  # pre-turn DORMANT
    h.send_human("ok")
    assert h.stores.state.load_conversation().focus_memory_id is None  # cleared
    h.close()


# --- the focus predicate -----------------------------------------------------------------------

def test_focus_predicate_separates_backchannels_from_subjects():
    assert is_focus_setting("what about a coma?", MessageClass.SOCIAL_QUESTION) is True
    assert is_focus_setting("Deploy the agent onto the robot.", MessageClass.STATEMENT) is True
    # Backchannels, including STATEMENT-classified ones, do not set a subject.
    assert is_focus_setting("noted", MessageClass.STATEMENT) is False
    assert is_focus_setting("alright", MessageClass.STATEMENT) is False
    assert is_focus_setting("ok", MessageClass.ACKNOWLEDGEMENT) is False
    assert is_focus_setting("cool.", MessageClass.STATEMENT) is False
