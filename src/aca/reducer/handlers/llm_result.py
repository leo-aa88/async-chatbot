"""LLMResult handler — the validation pipeline (DESIGN 22.1, 29.4).

    Language-model output is data, never authority.

Pipeline: identity/duplicate check -> worker-failure check -> schema parse + clamp ->
branch by cycle type -> **current-state revalidation** -> bounded proposal application ->
optional durable outbound intent.

Revalidation is scoped correctly by cycle type (DESIGN §29.4 step 5, §31.10):

* **proactive / reactive-optional** results are re-checked against *current* committed state and
  dropped if superseded — a newer human event arrived, or the specific candidate was resolved /
  expired / removed (``revalidation.is_superseded``, DESIGN §16.2, §22.2, invariant 8/9). They
  are never regenerated inline.
* **mandatory** obligations are always satisfied or surfaced as a visible failure — never dropped
  as a personality choice and never disguised as silence (invariant 25). Per §31.10 v0 does not
  add head-of-line blocking, so a mandatory result is not superseded by later turns.
"""

from __future__ import annotations

from dataclasses import replace

from ...domain.enums import (
    ActionKind,
    EnrichmentStatus,
    ObligationStatus,
    OutboundKind,
    WorkStatus,
)
from ...domain.events import LLMResult
from ...domain.proposals import LLMDecision, parse_decision
from ...domain.runtime import WorkItem
from ...errors import ValidationError
from ..apply_proposals import apply_proposals
from ..context import ReducerContext
from ..gates import evaluate_proactive
from ..outbound import create_outbound
from ..revalidation import is_superseded
from ..support import charge_proactive_message
from .base import (
    CYCLE_MANDATORY,
    CYCLE_REACTIVE_OPTIONAL,
    HandlerOutcome,
)

# Sentinel a dispatcher writes into an LLMResult when the worker itself failed terminally
# (provider error / timeout) after exhausting retries, so a mandatory obligation is surfaced as
# a failure rather than left pending forever (invariant 25).
WORKER_ERROR_KEY = "__worker_error__"


def _finalize(ctx, cycle_id, action, *, useful=False, invalidated=False, note=None) -> None:
    """Record the outcome on the cycle's trace so `aca logs` answers "did it speak?" (DESIGN 26)."""
    ctx.stores.work.finalize_trace(
        cycle_id, action=action, useful_enrichment=useful,
        pre_outbox_invalidated=invalidated, note=note,
    )


def _cycle_type(work: WorkItem) -> str:
    return str(work.snapshot.get("context", {}).get("source", {}).get("cycle_type", CYCLE_REACTIVE_OPTIONAL))


def _channel(work: WorkItem) -> str:
    return str(work.snapshot.get("context", {}).get("source", {}).get("channel", "cli"))


def handle_llm_result(ctx: ReducerContext, event: LLMResult) -> HandlerOutcome:
    work = ctx.stores.work.get_work(event.work_id)
    if work is None or work.cycle_id != event.cycle_id:
        return HandlerOutcome(note="unknown_work", committed=False)
    if work.status is WorkStatus.COMPLETED:
        return HandlerOutcome(note="duplicate_result", committed=False)  # idempotent (invariant 24)

    cycle_type = _cycle_type(work)
    now = event.timestamp

    if isinstance(event.result, dict) and WORKER_ERROR_KEY in event.result:
        return _on_worker_failure(ctx, event, work, cycle_type, str(event.result[WORKER_ERROR_KEY]), now)

    try:
        decision = parse_decision(event.result)
    except ValidationError as exc:
        return _on_parse_failure(ctx, event, cycle_type, str(exc), now)

    if cycle_type == CYCLE_MANDATORY:
        outcome = _finish_mandatory(ctx, event, work, decision, now)
    elif cycle_type == CYCLE_REACTIVE_OPTIONAL:
        outcome = _finish_reactive(ctx, event, work, decision, now)
    else:
        outcome = _finish_proactive(ctx, event, work, decision, now)

    ctx.stores.work.complete(event.work_id, event.event_id, now)
    return outcome


def _on_worker_failure(ctx, event, work: WorkItem, cycle_type, error, now) -> HandlerOutcome:
    # The work item was already marked FAILED_TERMINAL by the dispatcher; do not re-complete it.
    if cycle_type == CYCLE_MANDATORY:
        obligation = ctx.stores.work.obligation_by_work(event.work_id)
        if obligation is not None:
            ctx.stores.work.update_obligation(
                replace(obligation, status=ObligationStatus.FAILED, last_error=f"worker_error:{error}")
            )
        _finalize(ctx, event.cycle_id, "failed", note="worker_failure")
        return HandlerOutcome(note=f"mandatory_worker_failure:{error}")
    _finalize(ctx, event.cycle_id, "silence", note="worker_failure")
    return HandlerOutcome(note=f"worker_failure:{error}")


def _on_parse_failure(ctx, event, cycle_type, error, now) -> HandlerOutcome:
    ctx.stores.work.complete(event.work_id, event.event_id, now)
    if cycle_type == CYCLE_MANDATORY:
        obligation = ctx.stores.work.obligation_by_work(event.work_id)
        if obligation is not None:
            ctx.stores.work.update_obligation(
                replace(obligation, status=ObligationStatus.FAILED, last_error=error)
            )
        _finalize(ctx, event.cycle_id, "failed", note="parse_failure")
        return HandlerOutcome(note=f"mandatory_parse_failure:{error}")
    # Proactive/reactive parse failure: no output, no regeneration (DESIGN 22.3).
    _finalize(ctx, event.cycle_id, "silence", note="parse_failure")
    return HandlerOutcome(note=f"parse_failure:{error}")


def _finish_mandatory(ctx, event, work: WorkItem, decision: LLMDecision, now) -> HandlerOutcome:
    obligation = ctx.stores.work.obligation_by_work(event.work_id)
    useful = apply_proposals(ctx, decision.proposals, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        # Intentional silence is not a valid terminal action for a task (DESIGN 15, invariant 25).
        if obligation is not None:
            ctx.stores.work.update_obligation(
                replace(obligation, status=ObligationStatus.FAILED, last_error="non_speak_for_mandatory")
            )
        _finalize(ctx, event.cycle_id, "failed", useful=useful, note="non_speak_for_mandatory")
        return HandlerOutcome(note="mandatory_non_speak_failure")

    message_id = create_outbound(
        ctx, kind=OutboundKind.MANDATORY, channel=_channel(work), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=obligation.source_event_id if obligation else None,
        now=now,
    )
    if obligation is not None:
        ctx.stores.work.update_obligation(
            replace(obligation, status=ObligationStatus.SATISFIED, satisfied_by_message_id=message_id)
        )
    _finalize(ctx, event.cycle_id, "speak", useful=useful)
    return HandlerOutcome(deliver=True, note="mandatory_satisfied")


def _finish_reactive(ctx, event, work: WorkItem, decision: LLMDecision, now) -> HandlerOutcome:
    useful = apply_proposals(ctx, decision.proposals, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        _finalize(ctx, event.cycle_id, "silence", useful=useful)
        return HandlerOutcome(note="reactive_silence")

    # An optional reply is droppable: if the conversation moved on or the candidate is gone, the
    # reply is stale (DESIGN §22.2). Optional silence is a valid terminal action here (DESIGN 15).
    superseded, reason = is_superseded(ctx, work, now)
    if superseded:
        _finalize(ctx, event.cycle_id, "silence", useful=useful, invalidated=True, note=reason)
        return HandlerOutcome(note=f"reactive_superseded:{reason}")

    # A chosen optional reply is reactive: delivered like a mandatory item, not proactive-gated.
    create_outbound(
        ctx, kind=OutboundKind.MANDATORY, channel=_channel(work), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=None, now=now,
    )
    _finalize(ctx, event.cycle_id, "speak", useful=useful)
    return HandlerOutcome(deliver=True, note="reactive_reply")


def _release_enrichment_claim(ctx, work: WorkItem, now) -> None:
    """Release a PENDING_ENRICHMENT claim the wake made if this result didn't enrich the memory.

    Resets to RAW so it can be retried later, or to DO_NOT_ENRICH after the configured attempt cap
    so a salient memory can't become a recurring token leak (DESIGN 12.8, invariant 23).
    """
    candidate = work.snapshot.get("context", {}).get("source", {}).get("candidate") or {}
    memory_id = candidate.get("provisional_memory_id") or (
        candidate.get("id") if candidate.get("kind") == "PROVISIONAL_MEMORY" else None
    )
    if not memory_id:
        return
    memory = ctx.stores.memory.get_memory(memory_id)
    if memory is None or memory.enrichment_status is not EnrichmentStatus.PENDING_ENRICHMENT:
        return  # already ENRICHED by this cycle, or never claimed — nothing to release
    attempts = memory.enrichment_attempts + 1
    status = (
        EnrichmentStatus.DO_NOT_ENRICH
        if attempts >= ctx.config.budgets.max_enrichment_attempts
        else EnrichmentStatus.RAW
    )
    ctx.stores.memory.update_memory(
        replace(memory, enrichment_status=status, enrichment_attempts=attempts)
    )


def _finish_proactive(ctx, event, work: WorkItem, decision: LLMDecision, now) -> HandlerOutcome:
    useful = apply_proposals(ctx, decision.proposals, now)
    _release_enrichment_claim(ctx, work, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        ctx.stores.work.finalize_trace(
            event.cycle_id, action="silence", useful_enrichment=useful,
            pre_outbox_invalidated=False, note=None,
        )
        return HandlerOutcome(reschedule=True, note="proactive_silence_enrich" if useful else "proactive_silence")

    # Pre-outbox revalidation against CURRENT state: generic hard gates AND candidate-specific
    # supersession (DESIGN 16.2, 22.2, invariant 9). A superseded result is dropped, not regenerated.
    superseded, reason = is_superseded(ctx, work, now)
    gate = evaluate_proactive(ctx, now, superseded=superseded)
    if not gate.allowed:
        detail = reason if superseded else gate.reason
        ctx.stores.work.finalize_trace(
            event.cycle_id, action="silence", useful_enrichment=useful,
            pre_outbox_invalidated=True, note=f"pre_outbox:{detail}",
        )
        return HandlerOutcome(reschedule=True, note=f"pre_outbox_invalidated:{detail}")

    charge_proactive_message(ctx, now)
    candidate = work.snapshot.get("context", {}).get("source", {}).get("candidate") or {}
    create_outbound(
        ctx, kind=OutboundKind.PROACTIVE, channel=_channel(work), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=None, now=now,
        # Carry the candidate so repeat-suppression is recorded on DELIVERY, not here at decision
        # time (DESIGN 6.1: anchor to delivered, not decided). While this item is in-flight the
        # candidate is excluded from re-selection; that dedup releases if it expires/fails.
        candidate_kind=candidate.get("kind"), candidate_id=candidate.get("id"),
    )
    ctx.stores.work.finalize_trace(
        event.cycle_id, action="speak", useful_enrichment=useful,
        pre_outbox_invalidated=False, note=None,
    )
    return HandlerOutcome(deliver=True, reschedule=True, note="proactive_speak")
