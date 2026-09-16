"""Current-state revalidation of an in-flight cognition result (DESIGN 16.2, 22.2, 29.4 step 5).

A worker consumed an immutable snapshot at ``basis_revision``; by the time its result arrives,
committed state may have advanced. Before an unsolicited/optional result is allowed to speak, the
reducer must re-check whether the *specific* thing it was asked to say still makes sense — not
just the generic budget/mode/quiet gates. This catches the case DESIGN §16.2 exists to prevent:
resurfacing a thought the human already implicitly answered.

Two independent invalidation signals are checked:

1. **A newer human event arrived after the cycle started.** The conversation moved on, so an
   unsolicited item about an older thought is stale (DESIGN §22.2, §13.6).
2. **The candidate itself is no longer valid.** A deferred intent was resolved or expired, or the
   topic/memory it referenced no longer exists.

Mandatory obligations are deliberately *not* subject to this drop path — a task is always
answered (invariant 25, DESIGN §31.10); only proactive/optional-reactive results may be dropped.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..domain.enums import CandidateKind
from ..domain.runtime import WorkItem
from .context import ReducerContext


def _candidate(work: WorkItem) -> dict[str, Any] | None:
    source = work.snapshot.get("context", {}).get("source", {}) or {}
    candidate = source.get("candidate")
    return candidate if isinstance(candidate, dict) else None


def _candidate_invalidated(ctx: ReducerContext, candidate: dict[str, Any], now: datetime) -> bool:
    kind = candidate.get("kind")
    cid = candidate.get("id")
    if not cid:
        return False
    if kind == CandidateKind.DEFERRED_INTENT.value:
        intent = ctx.stores.memory.get_intent(cid)
        return intent is None or intent.status != "pending" or intent.expires_at <= now
    if kind == CandidateKind.TOPIC.value:
        return ctx.stores.memory.get_topic(cid) is None
    if kind == CandidateKind.PROVISIONAL_MEMORY.value:
        return ctx.stores.memory.get_memory(cid) is None
    return False


def is_superseded(ctx: ReducerContext, work: WorkItem, now: datetime) -> tuple[bool, str]:
    """Return ``(superseded, reason)`` for a proactive/optional result against current state."""
    conversation = ctx.stores.state.load_conversation()
    last_human = conversation.last_human_message_at
    if last_human is not None and last_human > work.created_at:
        # A human turn landed after this cycle began; the unsolicited thought is now stale.
        return True, "newer_human_event"

    candidate = _candidate(work)
    if candidate is not None and _candidate_invalidated(ctx, candidate, now):
        return True, "candidate_resolved_or_gone"

    return False, ""
