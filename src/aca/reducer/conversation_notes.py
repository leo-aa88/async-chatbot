"""The conversation as the model sees it: what became of each human message that got no reply.

Without this the agent saw an unanswered message and couldn't tell whether it had chosen silence, a
reply had failed, or a newer message had overtaken it. So it apologised for deliberate silences,
answered stale messages, or claimed it had chosen a silence that was really a failure (DESIGN 15:
silence is a choice, a failure is a failure). Each note comes from the message's cycle trace.
"""

from __future__ import annotations

from datetime import datetime

from .context import ReducerContext

CHOSE_SILENCE = "chose_silence"  # seen, and left unanswered: a choice, and final
FAILED = "failed"                # a reply was attempted and failed: never a choice
SUPERSEDED = "superseded"        # a newer message arrived before the reply

_FAILURE_NOTES = frozenset({"worker_failure", "parse_failure"})


def outcome_note(action: str | None, notes: str | None, invalidated: bool) -> str | None:
    """A human message's outcome from its cycle trace, or ``None`` (answered, or still in flight)."""
    if action == "failed":
        return FAILED
    if action != "silence":
        return None
    if invalidated:
        return SUPERSEDED
    # A reply-optional cycle records a failed or unparseable model call as silence (DESIGN 22.3); to
    # the agent it is still a failure, not something it chose.
    if notes in _FAILURE_NOTES:
        return FAILED
    return CHOSE_SILENCE


def recent_conversation(ctx: ReducerContext) -> list[dict]:
    """The recent turns, chronological, with an ``outcome`` on each human message that got no reply."""
    turns = ctx.stores.outbox.recent_turns(with_event_ids=True)
    outcomes = ctx.stores.work.trace_outcomes([t["event_id"] for t in turns if t.get("event_id")])
    for turn in turns:
        trace = outcomes.get(turn.pop("event_id", None) or "")
        note = outcome_note(*trace) if trace else None
        if note:
            turn["outcome"] = note
    return turns


def human_local_time(now_local: datetime) -> str:
    """"Thursday 1 October 2026, 14:05 (America/Sao_Paulo)": the configured local time (invariant 10)."""
    zone = getattr(now_local.tzinfo, "key", None) or str(now_local.tzinfo)
    return f"{now_local:%A} {now_local.day} {now_local:%B %Y, %H:%M} ({zone})"
