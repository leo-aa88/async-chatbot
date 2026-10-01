"""The model sees what became of each unanswered human message, and the human's local time.

An unanswered message used to look the same whether the agent chose silence, a reply failed, or a newer
message overtook it, so the agent apologised for deliberate silences and could claim a failure as its
own choice (DESIGN 15).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import make_harness

from aca.clock import ManualClock
from aca.config import Config
from aca.domain.cycles import CYCLE_MANDATORY
from aca.reducer.conversation_notes import (
    CHOSE_SILENCE,
    FAILED,
    SUPERSEDED,
    human_local_time,
    outcome_note,
)
from aca.workers.base import LLMOutput


@pytest.mark.parametrize(("trace", "expected"), [
    (("speak", None, False), None),                       # answered
    ((None, None, False), None),                          # still in flight
    (("silence", "silence_stochastic", False), CHOSE_SILENCE),
    (("silence", "focus_set", False), CHOSE_SILENCE),     # the model chose silence
    (("silence", "worker_failure", False), FAILED),       # recorded as silence, but a failure
    (("silence", "parse_failure", False), FAILED),
    (("failed", "non_speak_for_mandatory", False), FAILED),
    (("silence", "newer_human_event", True), SUPERSEDED),
])
def test_a_trace_becomes_an_outcome_note(trace, expected):
    assert outcome_note(*trace) == expected


def test_the_local_time_names_the_day_and_the_zone():
    clock = ManualClock(datetime(2026, 10, 1, 17, 5, tzinfo=UTC), local_timezone="America/Sao_Paulo")
    assert human_local_time(clock.now_local()) == "Thursday 1 October 2026, 14:05 (America/Sao_Paulo)"


class _Scripted:
    """Returns the scripted results in order, one per model call."""

    def __init__(self, *results: dict) -> None:
        self.results = list(results)

    async def run(self, snapshot):
        return LLMOutput(result=dict(self.results.pop(0)), tokens_in=1, tokens_out=1)


def _context_of_the_next_question(tmp_path, first: str, *results: dict) -> dict:
    clock = ManualClock(datetime(2026, 10, 1, 17, 5, tzinfo=UTC), local_timezone="America/Sao_Paulo")
    config = Config.from_mapping({"rng_seed": 1234, "local_timezone": "America/Sao_Paulo"})
    h = make_harness(tmp_path, config, clock, llm=_Scripted(*results))
    try:
        h.send_human(first)
        h.run_all_pending()
        for message in h.stores.outbox.deliverable():
            h.deliver(message.message_id)  # a reply joins the conversation when it is delivered
        clock.advance(60)
        h.send_human("Explain this stack trace.")
        [work] = [w for w in h.pending_work()
                  if (w.snapshot.get("context") or {}).get("source", {}).get("cycle_type") == CYCLE_MANDATORY]
        return work.snapshot["context"]
    finally:
        h.close()


def test_a_message_left_unanswered_is_marked_as_a_choice(tmp_path):
    # Whether the runtime gated it or the model chose silence, it was the agent's choice.
    context = _context_of_the_next_question(
        tmp_path, "the deploy finally went through and the dashboards look calm today", {"action": "silence"})
    first, current = context["recent_conversation"]
    assert first["outcome"] == CHOSE_SILENCE
    assert "outcome" not in current  # the message being answered now
    assert context["agent_state"]["human_local_time"] == "Thursday 1 October 2026, 14:06 (America/Sao_Paulo)"


def test_a_failed_reply_is_marked_as_a_failure_never_a_choice(tmp_path):
    # A question whose reply came back as silence: the obligation failed (invariant 25).
    context = _context_of_the_next_question(tmp_path, "How do I read this core dump?", {"action": "silence"})
    assert context["recent_conversation"][0]["outcome"] == FAILED


def test_an_answered_message_has_no_note_and_no_ids_reach_the_prompt(tmp_path):
    context = _context_of_the_next_question(
        tmp_path, "How do I read this core dump?", {"action": "speak", "message": "With gdb."})
    assert [t["role"] for t in context["recent_conversation"]] == ["human", "agent", "human"]
    assert all("outcome" not in t and "event_id" not in t for t in context["recent_conversation"])


def test_the_prompt_explains_the_notes():
    from aca.cognition.snapshot import Snapshot
    from aca.workers.llm.prompt import build_prompt

    system, _ = build_prompt(Snapshot("c", "w", 1, "t", {}))
    for word in ("human_local_time", "chose_silence", "failed", "superseded"):
        assert word in system
