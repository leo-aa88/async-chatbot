"""Focus transition (DESIGN §35.3): a human-turn result may override the conversation's *subject*.

The §34.4 deterministic predicate writes a *provisional* focus at reduction (so a wake before the
result is scored against something); when the human-turn generative result carries a ``focus``
transition, the ``llm_result`` handler **overrides** it — a validated state write, not a suppression
(invariant 42b). ``KEEP`` reverts to the pre-turn subject (fixing the check-in residual), ``REPLACE``
re-points to a validated provisional-memory id, ``CLEAR`` empties it; a missing/unknown transition
applies no override, so the current worker reproduces v0.7 exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import Harness

from aca.clock import ManualClock
from aca.config import Config
from aca.domain.enums import WorkKind
from aca.domain.proposals import parse_decision
from aca.reducer.discourse import resolve_focus_transition
from aca.workers.base import LLMOutput


class _FocusLLM:
    """A reply worker that speaks and returns a chosen focus transition. ``mode`` is mutable so one
    harness can establish a subject on an early turn and transition it on a later one."""

    def __init__(self, mode: str = "none") -> None:
        self.mode = mode
        self.replace_id: str | None = None

    async def run(self, snapshot) -> LLMOutput:
        result = {"action": "speak", "message": "Here is a direct answer."}
        source = snapshot.context.get("source", {})
        if self.mode == "keep":
            result["focus"] = "KEEP"
        elif self.mode == "clear":
            result["focus"] = "CLEAR"
        elif self.mode == "replace_turn":
            result["focus"] = "REPLACE"
            result["focus_memory_id"] = source.get("turn_memory_id")
        elif self.mode == "replace_id":
            result["focus"] = "REPLACE"
            result["focus_memory_id"] = self.replace_id
        elif self.mode == "replace_invalid":
            result["focus"] = "REPLACE"
            result["focus_memory_id"] = "not_a_provisional_memory"
        # mode == "none": no focus field -> no override (v0.7 behavior)
        return LLMOutput(result=result, tokens_in=8, tokens_out=4)


def _config() -> Config:
    return Config.from_mapping({
        "rng_seed": 7,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "conversation": {"active_within": "60s", "idle_within": "1h"},
    })


def _harness(tmp_path, mode: str = "none") -> Harness:
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                   llm=_FocusLLM(mode))


def _focus(h: Harness) -> str | None:
    return h.stores.state.load_conversation().focus_memory_id


def _recent_memory_id(h: Harness) -> str:
    return h.stores.memory.recent_memories(limit=1)[0].id


def _mandatory_turn(h: Harness, text: str) -> str:
    """Send a mandatory (response-required) human turn and return this turn's provisional memory id.
    Advances the clock 2s first so successive turns get distinct timestamps (stable recency)."""
    h.clock.advance(2)
    h.send_human(text)
    return _recent_memory_id(h)


def _latest_llm_work_id(h: Harness) -> str:
    """The most recently created pending cognition (LLM) work item — the turn currently in flight."""
    llm = [w for w in h.pending_work() if w.kind is WorkKind.LLM_COGNITION]
    return max(llm, key=lambda w: w.created_at).work_id


# --- override semantics -------------------------------------------------------------------------

def test_replace_returns_focus_to_an_earlier_memory(tmp_path):
    # Establish subject A deterministically (worker emits no focus). Then a new-subject turn B: the
    # deterministic predicate anchors the provisional focus to B, and the worker's REPLACE(A) — a
    # deliberate return to the earlier memory — overrides it. Proves the override *and* case 7
    # (a pre-result read sees the provisional, not the proposal).
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a  # deterministic focus (v0.7 floor)

    h.llm.mode = "replace_id"
    h.llm.replace_id = mem_a
    mem_b = _mandatory_turn(h, "What about the sensor budget?")
    assert _focus(h) == mem_b  # provisional anchored to B *before* the result (case 7)
    h.run_all_pending()
    assert _focus(h) == mem_a  # REPLACE overrode the provisional: back to A
    h.close()


def test_keep_reverts_the_unlisted_checkin_anchor(tmp_path):
    # The check-in residual (§34.4): an *unlisted* confirmation phrasing ("Was that understandable?")
    # is a TASK_QUESTION, so is_focus_setting wrongly anchors the provisional focus to it. KEEP is
    # the fix: it reverts to the pre-turn subject rather than leaving the wrong anchor.
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a

    h.llm.mode = "keep"
    mem_b = _mandatory_turn(h, "Was that understandable?")
    assert _focus(h) == mem_b  # the residual: provisional anchored the check-in
    h.run_all_pending()
    assert _focus(h) == mem_a  # KEEP reverted to the real subject, not the check-in
    h.close()


def test_clear_empties_the_focus(tmp_path):
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a

    h.llm.mode = "clear"
    _mandatory_turn(h, "Anyway, what time is it?")
    h.run_all_pending()
    assert _focus(h) is None  # nothing on the floor; the gate now fails open (widening)
    h.close()


def test_replace_invalid_id_is_treated_as_keep(tmp_path):
    # §35.9 case 6: a REPLACE naming a non-provisional-memory id (a topic_id or unknown) is not
    # adopted — it is treated as KEEP (reverts to the pre-turn subject), never an error.
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a

    h.llm.mode = "replace_invalid"
    mem_b = _mandatory_turn(h, "What about the sensor budget?")
    h.run_all_pending()
    assert _focus(h) == mem_a  # not adopted; treated as KEEP -> prior subject
    assert _focus(h) != mem_b
    assert _focus(h) != "not_a_provisional_memory"
    h.close()


def test_missing_focus_is_v07_deterministic_behavior(tmp_path):
    # §35.9 cases 3 & 8: the current worker emits no focus -> no override -> the deterministic
    # provisional focus stands. A focus-setting turn anchors to its own memory, exactly as v0.7.
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a
    h.close()


def test_no_call_turn_keeps_the_focus(tmp_path):
    # §35.9 case 8: a backchannel ("ok") makes no generative call, so no result carries a transition
    # and the deterministic focus stands — a live acknowledgement keeps the known-good subject.
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    h.run_all_pending()
    assert _focus(h) == mem_a

    h.clock.advance(2)
    h.send_human("ok")  # ACKNOWLEDGEMENT: no call, no work to run
    assert _focus(h) == mem_a  # unchanged
    h.close()


def test_stale_mandatory_result_does_not_clobber_a_newer_turn(tmp_path):
    # Two mandatory turns in flight at once. A is dispatched; then B is reduced first, deterministically
    # setting the focus to mem_b; then A's now-stale result arrives with REPLACE(mem_a). The mandatory
    # reply is always delivered (invariant 25), but the focus write must be DROPPED — a stale result
    # must not drag the focus back to A's subject. (Reachable in normal async operation; the reactive
    # branch is guarded by is_superseded, the mandatory branch is not — so the guard lives in the
    # focus write itself.)
    h = _harness(tmp_path, "none")
    mem_a = _mandatory_turn(h, "How does the robot keep its balance?")
    a_llm = _latest_llm_work_id(h)          # A's cognition work, still pending
    mem_b = _mandatory_turn(h, "What about the sensor budget?")
    b_llm = _latest_llm_work_id(h)
    assert _focus(h) == mem_b               # B's deterministic provisional focus

    h.llm.mode = "replace_id"               # A's stale result wants REPLACE(mem_a)...
    h.llm.replace_id = mem_a
    h.run_work(a_llm)
    assert _focus(h) == mem_b               # ...dropped: B still owns the subject

    h.llm.mode = "keep"                     # B is the current turn -> not stale -> its write applies
    h.run_work(b_llm)
    assert _focus(h) == mem_a               # KEEP reverts to B's pre-turn subject (A)
    h.close()


# --- pure contract ------------------------------------------------------------------------------

def test_parse_decision_focus_whitelist():
    d = parse_decision({"action": "speak", "message": "x", "focus": "replace", "focus_memory_id": "m1"})
    assert d.focus_transition == "REPLACE" and d.focus_memory_id == "m1"
    assert parse_decision({"action": "speak", "message": "x", "focus": "KEEP"}).focus_transition == "KEEP"
    assert parse_decision({"action": "speak", "message": "x", "focus": "clear"}).focus_transition == "CLEAR"
    assert parse_decision({"action": "speak", "message": "x", "focus": "nope"}).focus_transition is None
    assert parse_decision({"action": "speak", "message": "x"}).focus_transition is None  # absent
    assert parse_decision({"action": "speak", "message": "x", "focus_memory_id": ""}).focus_memory_id is None


def test_resolve_focus_transition_policy(tmp_path):
    h = _harness(tmp_path, "none")
    mem = _mandatory_turn(h, "How does the robot keep its balance?")  # a real provisional memory
    prior = "prior_mem"
    # Absent/unknown -> no override (v0.7 floor).
    assert resolve_focus_transition(h.ctx, None, None, prior) == (False, None)
    # KEEP reverts to the pre-turn subject; CLEAR empties.
    assert resolve_focus_transition(h.ctx, "KEEP", None, prior) == (True, prior)
    assert resolve_focus_transition(h.ctx, "CLEAR", None, prior) == (True, None)
    # REPLACE adopts only a validated existing provisional memory; anything else is treated as KEEP.
    assert resolve_focus_transition(h.ctx, "REPLACE", mem, prior) == (True, mem)
    assert resolve_focus_transition(h.ctx, "REPLACE", "ghost", prior) == (True, prior)
    assert resolve_focus_transition(h.ctx, "REPLACE", None, prior) == (True, prior)
    h.close()
