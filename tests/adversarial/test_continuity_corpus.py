"""Continuity eval corpus (DESIGN §27): does the agent continue when it should and abandon when it
should — across the six conversational dynamics?

Each case seeds a deterministic end-state (a focus subject + candidate vectors that encode the
semantic relationship, exactly as the §34 discourse tests do), fires one proactive wake, and observes
whether the agent committed a message. The hand-authored ``should_speak`` label is the ground truth;
the score is two-sided (persistence AND restraint), so neither chattiness nor thread-death can inflate
it. The focus-transition MACHINERY that produces these end-states is tested separately
(``test_focus_transition.py``); this corpus tests the proactive *decision* the end-state should yield.

Label rationale is in each case. The corpus itself is **observational** — `test_continuity_corpus_report`
prints the scored report and asserts only that it is well-formed, not any particular rate, so an
*improvement* never breaks CI. Only a small set of load-bearing invariants is hard-pinned as
regressions (near-repeat must not speak, an off-focus interruption must not intrude, dormant
resurfacing must remain possible, a backchannel must preserve focus, and a closed thread must renag
*only itself* — its own candidate stays silent while an unrelated worthwhile thought still speaks).
The `shift` gap this corpus originally surfaced — a closed thread was still spoken because a cleared
focus fail-opened the gate — is now fixed by the §34.11 closed-subject *identity*, so it is a pair of
hard regressions rather than the non-strict xfail it began as.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import ProvisionalMemory, Topic
from aca.eval.continuity import (
    ContinuityCategory,
    DecisionOutcome,
    format_continuity,
    score,
)
from aca.workers.base import LLMOutput

_RATE = half_life_to_rate_per_hour(24.0)
_ON_THREAD = [0.96, 0.28, 0.0]   # cosine ~0.96 to the focus -> CONTINUE
_OFF_THREAD = [0.0, 1.0, 0.0]    # cosine 0 to the focus     -> ORPHAN
_FOCUS_VEC = [1.0, 0.0, 0.0]


class _SpeakLLM:
    """Always speaks a fixed line, so whether a proactive message appears is decided by the gates,
    not by model restraint — the corpus scores the machinery, with the model held constant."""

    async def run(self, snapshot) -> LLMOutput:
        return LLMOutput(result={"action": "speak", "message": "Following that thread —"},
                         tokens_in=8, tokens_out=5)


def _config() -> Config:
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        "conversation": {"active_within": "10s", "idle_within": "1h"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _harness(tmp_path) -> Harness:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                   llm=_SpeakLLM())


def _set_conversation(h: Harness, *, gap_seconds: float, focus_memory_id: str | None,
                      closed_focus_memory_id: str | None = None) -> None:
    now = h.clock.now_utc()
    conv = h.stores.state.load_conversation()
    h.stores.state.save_conversation(conv.__class__(
        last_human_message_at=now - timedelta(seconds=gap_seconds), focus_memory_id=focus_memory_id,
        closed_focus_memory_id=closed_focus_memory_id))


def _install_focus(h: Harness, vec: list[float]) -> None:
    now = h.clock.now_utc()
    h.stores.memory.insert_memory(ProvisionalMemory(
        id="focus_mem", event_id="e", text="the current subject", activation=0.8, salience=0.8,
        decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
        enrichment_status=EnrichmentStatus.ENRICHED))  # ENRICHED -> out of the candidate pool
    h.stores.memory.insert_embedding("focus_emb", "focus_mem", "m", vec, now)


def _install_candidate(h: Harness, tid: str, vec: list[float]) -> None:
    now = h.clock.now_utc()
    h.stores.memory.insert_topic(Topic(
        id=tid, summary=tid, activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
        created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None))
    h.stores.memory.insert_topic_embedding(tid, "m", vec, now)


def _did_speak(h: Harness) -> bool:
    h.wake()
    h.run_all_pending()
    return bool(h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE))


def _outcome(case_id, category, candidate_id, should_speak, did_speak) -> DecisionOutcome:
    return DecisionOutcome(case_id, category, should_speak=should_speak, did_speak=did_speak,
                           candidate_id=candidate_id, detail="spoke" if did_speak else "silent")


# --- the corpus ---------------------------------------------------------------------------------
# Each builder seeds a scenario, fires one wake, and returns a labelled DecisionOutcome.

def case_continuation_advance(tmp_path) -> DecisionOutcome:
    # A live thread (IDLE, focus set) with a genuine same-thread advance on the floor. The right move
    # is to continue it -> SPEAK. Failing = the thread dies.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)
        _install_candidate(h, "advance", _ON_THREAD)
    _set_conversation(h, gap_seconds=60, focus_memory_id="focus_mem")
    return _outcome("continuation_advance", ContinuityCategory.CONTINUATION, "advance", True, _did_speak(h))


def case_continuation_near_repeat(tmp_path) -> DecisionOutcome:
    # Same live thread, but the candidate merely restates something just expressed (a semantic
    # near-repeat). Continuing would be nagging -> stay SILENT (§16.2). Failing = repetition.
    h = _harness(tmp_path)
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)
        _install_candidate(h, "restate", _ON_THREAD)
        # An already-voiced topic with the same vector -> the candidate is a near-repeat of it.
        h.stores.memory.insert_topic_embedding("said", "m", _ON_THREAD, now)
        h.stores.memory.record_expression("TOPIC", "said", now)
    _set_conversation(h, gap_seconds=60, focus_memory_id="focus_mem")
    return _outcome("continuation_near_repeat", ContinuityCategory.CONTINUATION, "restate", False, _did_speak(h))


def case_interruption(tmp_path) -> DecisionOutcome:
    # A new subject is on the floor (the focus is the new subject); a candidate from the prior thread
    # is off it (ORPHAN). Resurfacing the old thread mid-new-subject is an intrusion -> stay SILENT.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)                 # the new subject holds the focus
        _install_candidate(h, "old_thread", _OFF_THREAD)
    _set_conversation(h, gap_seconds=60, focus_memory_id="focus_mem")
    return _outcome("interruption", ContinuityCategory.INTERRUPTION, "old_thread", False, _did_speak(h))


def case_resumption(tmp_path) -> DecisionOutcome:
    # The human has returned to the earlier subject, so the focus is back on it and the once-orphaned
    # candidate is on-thread again -> SPEAK. (Given a correctly-resumed focus; the transition itself
    # is tested in test_focus_transition.py.) Failing = a stale focus keeps it wrongly suppressed.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)                 # focus resumed onto the earlier subject
        _install_candidate(h, "resumed", _ON_THREAD)
    _set_conversation(h, gap_seconds=60, focus_memory_id="focus_mem")
    return _outcome("resumption", ContinuityCategory.RESUMPTION, "resumed", True, _did_speak(h))


def case_shift_renag_closed_thread(tmp_path) -> DecisionOutcome:
    # The human closed a subject (a §35.3 CLEAR: focus None, the closed subject's identity retained,
    # still a live/IDLE conversation). A candidate from that CLOSED thread should NOT be resurfaced —
    # nagging a finished topic -> stay SILENT. §34.11 renags only the closed subject; DORMANT still
    # resurfaces.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)                    # the (now closed) subject, with an embedding
        _install_candidate(h, "closed_thread", _ON_THREAD)  # related to it -> renag
    _set_conversation(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="focus_mem")
    return _outcome("shift_renag", ContinuityCategory.SHIFT, "closed_thread", False, _did_speak(h))


def case_shift_unrelated_thought(tmp_path) -> DecisionOutcome:
    # Same close, but a genuinely UNRELATED worthwhile thought. Closing subject X must not gag every
    # initiative for the rest of the IDLE window -> SPEAK. This is the anti-over-suppression guard:
    # the closed marker renags only the closed thread, not the whole conversation.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)                    # the closed subject
        _install_candidate(h, "unrelated", _OFF_THREAD)  # orthogonal to it -> allowed
    _set_conversation(h, gap_seconds=60, focus_memory_id=None, closed_focus_memory_id="focus_mem")
    return _outcome("shift_unrelated", ContinuityCategory.SHIFT, "unrelated", True, _did_speak(h))


def case_backchannel(tmp_path) -> DecisionOutcome:
    # A bare acknowledgement ("makes sense") carries no new subject: the focus must stay on the live
    # thread, so an on-thread thought still surfaces -> SPEAK. Failing (the backchannel mis-anchors
    # the focus onto itself) would wrongly orphan the on-thread candidate.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)
        _install_candidate(h, "on_thread", _ON_THREAD)
    _set_conversation(h, gap_seconds=60, focus_memory_id="focus_mem")
    h.send_human("makes sense")  # a backchannel: ACKNOWLEDGEMENT, no call, focus kept (not re-anchored)
    h.clock.advance(60)          # let the conversation settle back to IDLE (a fresh turn made it ACTIVE)
    return _outcome("backchannel", ContinuityCategory.BACKCHANNEL_RETENTION, "on_thread", True, _did_speak(h))


def case_dormant_resurfacing(tmp_path) -> DecisionOutcome:
    # The conversation has gone quiet (DORMANT); a worthwhile older thought resurfaces, unrelated to
    # the last subject. This is the distinctive autonomous behavior (§34.7) -> SPEAK. Failing = the
    # agent never resurfaces anything after a lull.
    h = _harness(tmp_path)
    with h.stores.db.transaction():
        _install_focus(h, _FOCUS_VEC)                 # a stale prior subject
        _install_candidate(h, "resurfaced", _OFF_THREAD)  # unrelated to it — but worthwhile
    _set_conversation(h, gap_seconds=2 * 3600, focus_memory_id="focus_mem")  # 2h gap -> DORMANT
    return _outcome("dormant_resurfacing", ContinuityCategory.DORMANT_RESURFACING, "resurfaced", True, _did_speak(h))


CASES = (
    case_continuation_advance,
    case_continuation_near_repeat,
    case_interruption,
    case_resumption,
    case_shift_renag_closed_thread,
    case_shift_unrelated_thought,
    case_backchannel,
    case_dormant_resurfacing,
)


# --- hard regressions: only load-bearing invariants ---------------------------------------------
# A small set of guarantees that must never silently break. Everything else in the corpus is
# scored-only (see the report below) so that an IMPROVEMENT never fails CI — the corpus is a
# measurement, not the truth table.

def test_near_repeat_must_not_speak(tmp_path):
    # §16.2: never re-voice, reworded, a thought just expressed.
    assert case_continuation_near_repeat(tmp_path).kind == "correct_abandonment"


def test_interruption_must_not_intrude(tmp_path):
    # §34: while a new subject holds the focus, an off-focus candidate stays muted.
    assert case_interruption(tmp_path).kind == "correct_abandonment"


def test_dormant_resurfacing_must_remain_possible(tmp_path):
    # §34.7: the distinctive autonomous behavior — a worthwhile older thought may surface after a lull.
    assert case_dormant_resurfacing(tmp_path).kind == "correct_continuation"


def test_backchannel_must_preserve_focus(tmp_path):
    # A bare acknowledgement must not re-anchor the focus, so the live thread stays speak-eligible.
    assert case_backchannel(tmp_path).kind == "correct_continuation"


def test_shift_must_abandon_the_closed_thread(tmp_path):
    # Once a §34.11 CLOSE retains the closed subject, the gate declines to renag it in IDLE. This was
    # a known gap (a cleared focus used to fail-open); the closed marker fixed it — closing a thread
    # must keep it closed.
    assert case_shift_renag_closed_thread(tmp_path).kind == "correct_abandonment"


def test_shift_must_not_gag_an_unrelated_thought(tmp_path):
    # The paired guarantee: closing a subject must renag ONLY that thread, not mute the whole IDLE
    # window. A genuinely unrelated worthwhile thought still speaks (guards against over-suppression).
    assert case_shift_unrelated_thought(tmp_path).kind == "correct_continuation"


# --- the scored report (observational) ----------------------------------------------------------

def test_continuity_corpus_report(tmp_path, capsys):
    # A measurement, not a gate: print the report and assert only that it is well-formed, so a rate
    # moving in EITHER direction (a fix or a regression on a scored-only case) never fails here — the
    # hard guarantees above are what protect against silent backsliding.
    outcomes = [case(tmp_path / f"c{i}") for i, case in enumerate(CASES)]
    sc = score(outcomes)
    print("\n" + "\n".join(format_continuity(sc)))

    s = sc.summary()
    assert s["decisions"] == len(CASES)
    assert s["balanced_accuracy"] is not None   # both sides of the corpus were exercised
