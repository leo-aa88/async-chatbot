"""Continuity gate: suppress re-voicing a just-said thought in reworded form (DESIGN 6, 11.3).

A proactive speak is suppressed when its candidate is a semantic near-paraphrase (cosine >=
topic_dedup_cosine) of something voiced within the repeat window — but an *advance* (same subject,
new content, below that threshold) still speaks. Only ever suppresses; never forces speech.
"""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind, WorkKind
from aca.domain.state import ProvisionalMemory, Topic

_RATE = half_life_to_rate_per_hour(24.0)


def _config():
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _harness(tmp_path):
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)))


def _seed_candidate_topic(h: Harness, tid: str, vector: list[float]) -> None:
    """A topic that will win selection, with a summary embedding."""
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary=tid, activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None,
        ))
        h.stores.memory.insert_topic_embedding(tid, "m", vector, now)


def _record_recent_voiced(h: Harness, tid: str, vector: list[float]) -> None:
    """A different topic that was voiced just now (embedding + expression record)."""
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic_embedding(tid, "m", vector, now)
        h.stores.memory.record_expression("TOPIC", tid, now)


def _seed_raw_memory(h: Harness, mid: str, vector: list[float]) -> None:
    """A salient RAW memory (speak- and enrichment-eligible) with an embedding."""
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e", text=f"content {mid}", activation=0.9, salience=0.9,
            decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
            enrichment_status=EnrichmentStatus.RAW,
        ))
        h.stores.memory.insert_embedding(f"emb_{mid}", mid, "m", vector, now)


def _proactive_outbound(h: Harness) -> list:
    return h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)


def _wake_note(h: Harness) -> str:
    h.wake()
    return h.stores.work.recent_traces(limit=1)[0].notes


def test_near_paraphrase_of_recent_message_is_suppressed(tmp_path):
    h = _harness(tmp_path)
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])
    _seed_candidate_topic(h, "t_now", [1.0, 0.0, 0.0])  # cosine 1.0 with t_said -> repeat
    assert _wake_note(h) == "continuity_repeat"
    assert [w for w in h.pending_work() if w.kind.value != "EMBEDDING"] == []  # nothing dispatched
    h.close()


def test_advance_of_recent_message_still_speaks(tmp_path):
    h = _harness(tmp_path)
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])
    _seed_candidate_topic(h, "t_now", [0.85, 0.527, 0.0])  # cosine ~0.85: same subject, new content
    assert _wake_note(h) == "proactive_dispatch"  # an advance is not suppressed
    h.close()


def test_no_recent_expression_speaks(tmp_path):
    h = _harness(tmp_path)
    _seed_candidate_topic(h, "t_now", [1.0, 0.0, 0.0])  # nothing voiced recently
    assert _wake_note(h) == "proactive_dispatch"
    h.close()


def test_enrichment_only_degrade_never_voices(tmp_path):
    # The blocking bug: a RAW paraphrase memory dispatches enrichment-only (output_eligible False),
    # but FakeLLM speaks anyway (salience >= 0.6). The reducer — not the prompt — must drop the speak.
    h = _harness(tmp_path)
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])
    _seed_raw_memory(h, "m_raw", [1.0, 0.0, 0.0])  # cosine 1.0 with t_said -> near-repeat
    assert _wake_note(h) == "proactive_dispatch"  # dispatched (for enrichment), not silenced at wake
    # The dispatched work is enrichment-only, and its source says so.
    work = [w for w in h.pending_work() if w.kind is not WorkKind.EMBEDDING][0]
    assert work.kind is WorkKind.LLM_ENRICHMENT
    assert work.snapshot["context"]["source"]["output_eligible"] is False

    h.run_all_pending()  # worker returns speak; reducer must refuse to voice it
    assert _proactive_outbound(h) == []  # invariant 2: LLM output is data, not authority
    assert h.stores.memory.get_memory("m_raw").enrichment_status is EnrichmentStatus.ENRICHED
    assert h.stores.work.recent_traces(limit=1)[0].notes == "enrichment_only"
    h.close()


def test_advance_dispatches_cognition_and_delivers(tmp_path):
    # The positive control: an advance is output-eligible and, when the worker speaks, is voiced.
    h = _harness(tmp_path)
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])
    _seed_candidate_topic(h, "t_now", [0.85, 0.527, 0.0])  # cosine ~0.85 < dedup: an advance
    h.wake()
    work = [w for w in h.pending_work() if w.kind is not WorkKind.EMBEDDING][0]
    assert work.kind is WorkKind.LLM_COGNITION
    assert work.snapshot["context"]["source"]["output_eligible"] is True
    h.run_all_pending()
    assert len(_proactive_outbound(h)) == 1  # a genuine advance reaches the outbox
    h.close()


def test_repeat_arriving_after_dispatch_is_dropped_pre_outbox(tmp_path):
    # Current-state checkpoint: nothing was voiced at wake, so t_now dispatches output-eligible. A
    # paraphrase is expressed *before* the result is processed; the pre-outbox re-check must drop it.
    h = _harness(tmp_path)
    _seed_candidate_topic(h, "t_now", [1.0, 0.0, 0.0])
    h.wake()
    assert h.stores.work.recent_traces(limit=1)[0].notes == "proactive_dispatch"
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])  # a near-repeat lands in the interim
    h.run_all_pending()
    assert _proactive_outbound(h) == []
    assert h.stores.work.recent_traces(limit=1)[0].notes == "pre_outbox:continuity_repeat"
    h.close()


def test_mandatory_reply_is_untouched_by_continuity(tmp_path):
    # Scope guard (invariants 12/13): the gate is proactive-only. A task still gets answered even
    # when its content would look like a recent expression.
    h = _harness(tmp_path)
    _record_recent_voiced(h, "t_said", [1.0, 0.0, 0.0])
    h.send_human("Explain this stack trace.")
    h.run_all_pending()
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.MANDATORY)) == 1
    h.close()
