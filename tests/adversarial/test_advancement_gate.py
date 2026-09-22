"""Advancement relation gate (DESIGN §35.4): a proactive message must *move* the thread.

The proactive worker classifies its own drafted message against the current thread; the reducer
mutes a self-assessed ORPHAN/REPEAT at pre-outbox. Suppress-only (invariant 42a): a forward move or
a missing/unknown relation leaves the v0.7 (§34) decision untouched, and the label can never force a
speak the deterministic gates already block.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import OutboundKind
from aca.domain.proposals import parse_decision
from aca.domain.state import Topic
from aca.reducer.discourse import advancement_suppresses
from aca.workers.base import LLMOutput

_RATE = half_life_to_rate_per_hour(24.0)


class _RelationLLM:
    """A proactive worker that speaks a fixed message with a chosen advancement relation (or none)."""

    def __init__(self, relation: str | None) -> None:
        self._relation = relation

    async def run(self, snapshot) -> LLMOutput:
        result = {"action": "speak", "message": "I've been thinking about the robot thread."}
        if self._relation is not None:
            result["relation"] = self._relation
        return LLMOutput(result=result, tokens_in=10, tokens_out=6)


def _config():
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        # Short ACTIVE window so a 60s-old turn is IDLE (a live thread); long IDLE window.
        "conversation": {"active_within": "10s", "idle_within": "1h"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _make_idle(h: Harness, focus: str | None = "t1") -> None:
    """Put the conversation in IDLE (a live window). With ``focus`` set (default) a focus subject
    exists, so the advancement gate sees a live *thread*; the discourse-focus gate still fails open
    at wake (no focus *vector* stored), letting the candidate reach the advancement gate. Pass
    ``focus=None`` for the reachable IDLE-but-no-focus state (only check-ins exchanged, so no turn
    ever set the focus) where there is no thread to be orphaned from."""
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(seconds=60), focus_memory_id=focus))


def _harness(tmp_path, relation: str | None) -> Harness:
    # Default conversation is DORMANT (no human turn), so the discourse-focus gate is inactive and
    # the candidate reaches the advancement gate.
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                   llm=_RelationLLM(relation))


def _candidate_topic(h: Harness, tid: str, vector: list[float] | None = None) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary=tid, activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None,
        ))
        if vector is not None:
            h.stores.memory.insert_topic_embedding(tid, "m", vector, now)


def _note(h: Harness) -> str:
    return h.stores.work.recent_traces(limit=1)[0].notes


def _run_proactive(tmp_path, relation: str | None, *, idle: bool = False,
                   focus: str | None = "t1") -> Harness:
    h = _harness(tmp_path, relation)
    _candidate_topic(h, "t1")
    if idle:
        _make_idle(h, focus)
    h.wake()
    assert _note(h) == "proactive_dispatch"  # dispatched (gate fails open at wake)
    h.run_all_pending()
    return h


def test_advancement_orphan_is_suppressed_while_idle(tmp_path):
    # ORPHAN mutes only when a live thread exists: IDLE window AND a focus subject set.
    h = _run_proactive(tmp_path, "ORPHAN", idle=True)  # default focus="t1"
    assert h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE) == []
    assert _note(h) == "pre_outbox:advancement_orphan"
    h.close()


def test_orphan_does_not_suppress_idle_without_focus(tmp_path):
    # IDLE but no focus_memory_id (only check-ins were ever exchanged, so no turn set the focus):
    # there is no thread to be off, so an ORPHAN label is a legitimate resurfacing and must NOT be
    # suppressed — mode alone (gate_active) is not enough, focus existence is required.
    h = _run_proactive(tmp_path, "ORPHAN", idle=True, focus=None)
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)) == 1
    h.close()


def test_orphan_does_not_suppress_dormant_resurfacing(tmp_path):
    # In DORMANT there is no current thread: an ORPHAN label is a legitimate resurfacing and must
    # NOT be suppressed (§34.7 preserves dormant resurfacing) — the load-bearing guard.
    h = _run_proactive(tmp_path, "ORPHAN", idle=False)
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)) == 1
    h.close()


def test_advancement_repeat_is_suppressed_in_any_mode(tmp_path):
    # REPEAT (never restate) mutes regardless of mode — here DORMANT.
    h = _run_proactive(tmp_path, "REPEAT", idle=False)
    assert h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE) == []
    assert _note(h) == "pre_outbox:advancement_repeat"
    h.close()


def test_advancement_forward_move_speaks(tmp_path):
    h = _run_proactive(tmp_path, "ADVANCE")
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)) == 1
    h.close()


def test_missing_relation_is_v07_behavior(tmp_path):
    # The current v0.7 worker emits no relation -> gated by §34 alone (here DORMANT -> speaks).
    h = _run_proactive(tmp_path, None)
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)) == 1
    h.close()


def test_unknown_relation_fails_open(tmp_path):
    # An unrecognized label parses to None (not a parse failure, §35.9 case 2) -> speaks.
    h = _run_proactive(tmp_path, "FLARGLE")
    assert len(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)) == 1
    h.close()


def test_forward_label_cannot_force_past_a_deterministic_block(tmp_path):
    # A self-assessed ADVANCE must not override a deterministic suppression: the candidate becomes a
    # semantic near-repeat between dispatch and result, and continuity drops it pre-outbox — the
    # ADVANCE label does not save it (the advancement gate is reached only after the §34 gates pass).
    h = _harness(tmp_path, "ADVANCE")
    _candidate_topic(h, "t1", vector=[1.0, 0.0, 0.0])
    h.wake()
    assert _note(h) == "proactive_dispatch"  # no repeat at wake yet
    now = h.clock.now_utc()
    with h.stores.db.transaction():  # a paraphrase is voiced before the result is processed
        h.stores.memory.insert_topic_embedding("t_said", "m", [1.0, 0.0, 0.0], now)
        h.stores.memory.record_expression("TOPIC", "t_said", now)
    h.run_all_pending()
    assert h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE) == []
    assert _note(h) == "pre_outbox:continuity_repeat"  # blocked by continuity, not advancement
    h.close()


# --- pure contract -----------------------------------------------------------------------------

def test_parse_decision_relation_whitelist():
    assert parse_decision({"action": "speak", "message": "x", "relation": "advance"}).relation == "ADVANCE"
    assert parse_decision({"action": "speak", "message": "x", "relation": "ORPHAN"}).relation == "ORPHAN"
    assert parse_decision({"action": "speak", "message": "x", "relation": "nope"}).relation is None
    assert parse_decision({"action": "speak", "message": "x"}).relation is None  # absent


def test_advancement_suppresses_policy(tmp_path):
    h = _harness(tmp_path, None)
    now = h.clock.now_utc()
    # REPEAT mutes in any mode; forward moves / None never.
    assert advancement_suppresses(h.ctx, "REPEAT", now) is True         # DORMANT default
    for forward in ("ADVANCE", "EVIDENCE", "REVISE", "CLOSE", "REOPEN"):
        assert advancement_suppresses(h.ctx, forward, now) is False
    assert advancement_suppresses(h.ctx, None, now) is False            # missing -> fail open
    # ORPHAN mutes only against a live thread: IDLE window AND a focus subject set.
    assert advancement_suppresses(h.ctx, "ORPHAN", now) is False        # DORMANT (no window)
    _make_idle(h, focus=None)                                           # IDLE but no focus set
    assert advancement_suppresses(h.ctx, "ORPHAN", now) is False        # no thread to be off
    _make_idle(h)                                                       # IDLE + focus (live thread)
    assert advancement_suppresses(h.ctx, "ORPHAN", now) is True
    assert advancement_suppresses(h.ctx, "REPEAT", now) is True         # REPEAT: any state
    h.close()
