"""Closed-subject state (DESIGN §34.11, v0.8): a deliberately closed thread is not "no focus yet".

``focus_memory_id is None`` had three meanings — never had a focus, closed on purpose, lapsed — and
the discourse gate failed open on all of them, so a just-closed thread got renagged in IDLE (the gap
the continuity corpus surfaced). A ``subject_closed`` marker set by a §35.3 ``CLEAR`` disambiguates
them: the gate declines to renag a closed thread while the human is present (IDLE), without touching
dormant resurfacing (DORMANT, gate inactive). These tests pin the gate rule and the marker's
lifecycle (set on CLEAR; cleared by a new subject or a lull; held across a bare acknowledgement).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.config import Config
from aca.reducer.discourse import is_discourse_orphan
from aca.workers.base import LLMOutput


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


class _FocusLLM:
    """A reply worker with a settable focus transition, so a turn can close or re-open the subject."""

    def __init__(self, focus: str | None = None) -> None:
        self.focus = focus  # None | "CLEAR" | "REPLACE"

    async def run(self, snapshot) -> LLMOutput:
        result = {"action": "speak", "message": "Noted."}
        if self.focus == "CLEAR":
            result["focus"] = "CLEAR"
        elif self.focus == "REPLACE":
            result["focus"] = "REPLACE"
            result["focus_memory_id"] = snapshot.context.get("source", {}).get("turn_memory_id")
        return LLMOutput(result=result, tokens_in=8, tokens_out=4)


def _harness(tmp_path, focus: str | None = None) -> Harness:
    return Harness(tmp_path, _config(), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                   llm=_FocusLLM(focus))


def _set(h: Harness, *, gap_seconds: float, focus_memory_id, subject_closed: bool) -> None:
    now = h.clock.now_utc()
    conv = h.stores.state.load_conversation()
    h.stores.state.save_conversation(replace(
        conv, last_human_message_at=now - timedelta(seconds=gap_seconds),
        focus_memory_id=focus_memory_id, subject_closed=subject_closed))


def _closed(h: Harness) -> bool:
    return h.stores.state.load_conversation().subject_closed


def _focus(h: Harness):
    return h.stores.state.load_conversation().focus_memory_id


# --- the gate rule (IDLE-scoped) ----------------------------------------------------------------

def test_closed_subject_suppresses_in_idle(tmp_path):
    h = _harness(tmp_path)
    _set(h, gap_seconds=60, focus_memory_id=None, subject_closed=True)   # IDLE, closed
    assert is_discourse_orphan(h.ctx, "TOPIC", "anything", h.clock.now_utc()) is True
    h.close()


def test_no_focus_not_closed_still_fails_open(tmp_path):
    # The declarative residual / a fresh conversation: no focus but nothing was closed -> fail open.
    h = _harness(tmp_path)
    _set(h, gap_seconds=60, focus_memory_id=None, subject_closed=False)
    assert is_discourse_orphan(h.ctx, "TOPIC", "anything", h.clock.now_utc()) is False
    h.close()


def test_closed_subject_does_not_suppress_in_dormant(tmp_path):
    # A lull leaves the gate inactive, so dormant resurfacing survives even with the closed marker set.
    h = _harness(tmp_path)
    _set(h, gap_seconds=2 * 3600, focus_memory_id=None, subject_closed=True)  # DORMANT
    assert is_discourse_orphan(h.ctx, "TOPIC", "anything", h.clock.now_utc()) is False
    h.close()


# --- the marker's lifecycle ---------------------------------------------------------------------

def test_clear_transition_marks_the_subject_closed(tmp_path):
    # A human turn whose result CLEARs the focus closes the subject: focus None AND subject_closed.
    h = _harness(tmp_path, focus="CLEAR")
    _set(h, gap_seconds=60, focus_memory_id="prior_subject", subject_closed=False)
    h.send_human("Anyway, that's settled — let's drop it.")
    h.run_all_pending()
    assert _focus(h) is None
    assert _closed(h) is True
    h.close()


def test_a_new_subject_reopens_the_floor(tmp_path):
    # After a close, a focus-setting turn establishes a subject -> the closed marker clears.
    h = _harness(tmp_path)  # deterministic focus (worker emits no transition)
    _set(h, gap_seconds=60, focus_memory_id=None, subject_closed=True)
    h.send_human("How does the gyroscope stabilize it?")  # a question -> focus-setting
    h.run_all_pending()
    assert _focus(h) is not None
    assert _closed(h) is False
    h.close()


def test_a_bare_ack_keeps_it_closed(tmp_path):
    # An acknowledgement after a close must not quietly re-open the resolved thread.
    h = _harness(tmp_path)
    _set(h, gap_seconds=60, focus_memory_id=None, subject_closed=True)
    h.send_human("ok")  # ACKNOWLEDGEMENT: no call, no new subject
    assert _closed(h) is True
    h.close()


def test_a_lull_ends_the_closed_window(tmp_path):
    # Once the conversation has gone DORMANT, the just-closed window is over: the next turn clears the
    # marker, so later resurfacing is unconstrained again.
    h = _harness(tmp_path)
    _set(h, gap_seconds=2 * 3600, focus_memory_id=None, subject_closed=True)  # closed, then a long lull
    h.send_human("ok")  # arrives after DORMANT -> pre_mode DORMANT ends the closed window
    assert _closed(h) is False
    h.close()
