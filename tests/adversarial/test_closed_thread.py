"""Closed-subject state (DESIGN §34.11, v0.8): a deliberately closed thread keeps its *identity*.

``focus_memory_id is None`` had three meanings — never had a focus, closed on purpose, lapsed — and
the discourse gate failed open on all of them, so a just-closed thread got renagged in IDLE (the gap
the continuity corpus surfaced). ``closed_focus_memory_id`` records *which* subject a §35.3 ``CLEAR``
closed, so the gate can **invert** against it: while IDLE with no active focus, a candidate *related*
to the closed subject is muted (don't reopen a resolved thread) while a genuinely *unrelated*
worthwhile thought still speaks. These tests pin that inversion and the marker's lifecycle.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.state import ProvisionalMemory, Topic
from aca.reducer.discourse import is_discourse_orphan
from aca.workers.base import LLMOutput

_RATE = half_life_to_rate_per_hour(24.0)
_SUBJECT_VEC = [1.0, 0.0, 0.0]
_RELATED_VEC = [0.96, 0.28, 0.0]   # cosine ~0.96 to the closed subject -> renag
_UNRELATED_VEC = [0.0, 1.0, 0.0]   # cosine 0 -> a genuinely different thought


def _config() -> Config:
    return Config.from_mapping({
        "rng_seed": 4,
        # High initiative so an optional-reactive declarative reliably dispatches (for the reactive-
        # silence case); irrelevant to the mandatory/ack/gate cases.
        "temperament": {"initiative": 0.95, "inhibition": 0.05},
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        "conversation": {"active_within": "10s", "idle_within": "1h"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


class _FocusLLM:
    """A reply worker with a settable focus transition (KEEP/REPLACE/CLEAR) and speak/silence, so a
    turn can close, re-open, or restore the subject with or without a verbal reply."""

    def __init__(self, focus: str | None = None, *, speak: bool = True) -> None:
        self.focus = focus  # None | "KEEP" | "CLEAR" | "REPLACE"
        self.speak = speak

    async def run(self, snapshot) -> LLMOutput:
        result: dict = {"action": "speak", "message": "Noted."} if self.speak else {"action": "silence"}
        if self.focus in ("KEEP", "CLEAR", "REPLACE"):
            result["focus"] = self.focus
            if self.focus == "REPLACE":
                result["focus_memory_id"] = snapshot.context.get("source", {}).get("turn_memory_id")
        return LLMOutput(result=result, tokens_in=8, tokens_out=4)


def _harness(tmp_path, focus: str | None = None, *, speak: bool = True) -> Harness:
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                   llm=_FocusLLM(focus, speak=speak))


def _set(h: Harness, *, gap_seconds: float, focus_memory_id, closed_focus_memory_id) -> None:
    now = h.clock.now_utc()
    conv = h.stores.state.load_conversation()
    h.stores.state.save_conversation(replace(
        conv, last_human_message_at=now - timedelta(seconds=gap_seconds),
        focus_memory_id=focus_memory_id, closed_focus_memory_id=closed_focus_memory_id))


def _install_subject(h: Harness, mid: str, vec: list[float]) -> None:
    now = h.clock.now_utc()
    h.stores.memory.insert_memory(ProvisionalMemory(
        id=mid, event_id="e", text="a closed subject", activation=0.8, salience=0.8,
        decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
        enrichment_status=EnrichmentStatus.ENRICHED))
    h.stores.memory.insert_embedding(f"{mid}_emb", mid, "m", vec, now)


def _install_candidate(h: Harness, tid: str, vec: list[float]) -> None:
    now = h.clock.now_utc()
    h.stores.memory.insert_topic(Topic(
        id=tid, summary=tid, activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
        created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None))
    h.stores.memory.insert_topic_embedding(tid, "m", vec, now)


def _closed(h: Harness):
    return h.stores.state.load_conversation().closed_focus_memory_id


def _focus(h: Harness):
    return h.stores.state.load_conversation().focus_memory_id


def _orphan(h: Harness, tid: str) -> bool:
    return is_discourse_orphan(h.ctx, "TOPIC", tid, h.clock.now_utc())


# --- the inverted gate (IDLE) -------------------------------------------------------------------

def test_closed_thread_renags_a_related_candidate(tmp_path):
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_subject(h, "closed", _SUBJECT_VEC)
        _install_candidate(h, "same_thread", _RELATED_VEC)
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="closed")  # IDLE
    assert _orphan(h, "same_thread") is True   # renagging the resolved thread -> muted
    h.close()


def test_closed_thread_allows_an_unrelated_candidate(tmp_path):
    # The load-bearing guard against over-suppression: closing subject X must NOT gag a genuinely
    # different worthwhile thought while IDLE.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_subject(h, "closed", _SUBJECT_VEC)
        _install_candidate(h, "different", _UNRELATED_VEC)
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="closed")  # IDLE
    assert _orphan(h, "different") is False     # unrelated -> speaks
    h.close()


def test_nothing_closed_fails_open(tmp_path):
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_candidate(h, "anything", _UNRELATED_VEC)
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id=None)
    assert _orphan(h, "anything") is False
    h.close()


def test_closed_thread_not_enforced_in_dormant(tmp_path):
    # A lull leaves the gate inactive, so even a candidate related to the closed subject resurfaces.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_subject(h, "closed", _SUBJECT_VEC)
        _install_candidate(h, "same_thread", _RELATED_VEC)
    _set(h, gap_seconds=2 * 3600, focus_memory_id=None, closed_focus_memory_id="closed")  # DORMANT
    assert _orphan(h, "same_thread") is False
    h.close()


# --- the marker's lifecycle ---------------------------------------------------------------------

def test_clear_records_the_closed_subject_identity(tmp_path):
    h = _harness(tmp_path, focus="CLEAR")
    _set(h, gap_seconds=60, focus_memory_id="prior_subject", closed_focus_memory_id=None)
    h.send_human("Anyway, that's settled — let's drop it.")
    h.run_all_pending()
    assert _focus(h) is None
    assert _closed(h) == "prior_subject"   # the closed thread keeps its identity
    h.close()


def test_a_new_subject_reopens_the_floor(tmp_path):
    h = _harness(tmp_path)  # deterministic focus (worker emits no transition)
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="old")
    h.send_human("How does the gyroscope stabilize it?")  # a question -> focus-setting
    h.run_all_pending()
    assert _focus(h) is not None
    assert _closed(h) is None
    h.close()


def test_a_bare_ack_keeps_it_closed(tmp_path):
    h = _harness(tmp_path)
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="old")
    h.send_human("ok")  # ACKNOWLEDGEMENT: no call, no new subject
    assert _closed(h) == "old"
    h.close()


def test_a_lull_ends_the_closed_window(tmp_path):
    h = _harness(tmp_path)
    _set(h, gap_seconds=2 * 3600, focus_memory_id=None, closed_focus_memory_id="old")  # then a lull
    h.send_human("ok")  # arrives after DORMANT -> the closed window is over
    assert _closed(h) is None
    h.close()


# --- lifecycle holes the review caught (must restore the PRE-turn closed state) -----------------

def test_keep_after_a_provisional_focus_preserves_the_closed_subject(tmp_path):
    # closed=A, then a check-in question the deterministic handler provisionally treats as focus-setting
    # (focus=B, clearing closed). The model correctly returns KEEP -> restore the PRE-turn state:
    # focus None, closed A. (Regression: KEEP read closed from the already-mutated conversation and
    # lost A, silently re-opening the thread.)
    h = _harness(tmp_path, focus="KEEP")
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="A")
    h.send_human("Was that clear?")  # TASK_QUESTION -> mandatory, provisionally focus-setting
    h.run_all_pending()
    assert _focus(h) is None
    assert _closed(h) == "A"
    h.close()


def test_a_second_clear_keeps_the_closed_identity(tmp_path):
    # Already closed (focus None, closed A); another CLEAR must not erase A just because there is no
    # current focus to close — it falls back to the prior closed subject.
    h = _harness(tmp_path, focus="CLEAR")
    _set(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="A")
    h.send_human("Should we forget all of that?")  # TASK_QUESTION -> mandatory
    h.run_all_pending()
    assert _focus(h) is None
    assert _closed(h) == "A"
    h.close()


def test_a_reactive_silence_can_close_the_subject(tmp_path):
    # The focus transition belongs to the human turn, not the reply (§35.3): a reactive result that
    # stays SILENT but returns CLEAR still closes the thread. ("Let's drop it" — nothing worth saying,
    # yet the subject is resolved.) Regression: _finish_reactive returned on silence before applying it.
    h = _harness(tmp_path, focus="CLEAR", speak=False)
    with h.stores.db.transaction():
        _install_candidate(h, "some_topic", _UNRELATED_VEC)  # gives the optional path a candidate
    _set(h, gap_seconds=60, focus_memory_id="X", closed_focus_memory_id=None)
    h.send_human("The gyroscope keeps drifting over time.")  # HIGH_INFORMATION -> reactive-optional
    assert h.stores.work.recent_traces(limit=1)[0].notes == "reactive_optional"  # it dispatched
    h.run_all_pending()
    assert _focus(h) is None
    assert _closed(h) == "X"   # closed = the pre-turn focus
    h.close()
