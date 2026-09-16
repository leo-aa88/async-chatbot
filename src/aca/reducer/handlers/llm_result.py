"""LLMResult handler — the validation pipeline (DESIGN 22.1, 29.4).

    Language-model output is data, never authority.

Pipeline: identity/duplicate check -> schema parse + clamp -> branch by cycle type ->
current-state revalidation -> bounded proposal application -> optional durable outbound intent.
A stale/superseded proactive result is dropped or deferred, never regenerated inline
(invariant 8). A mandatory obligation is satisfied, superseded, or surfaced as a visible
failure — never disguised as intentional silence (DESIGN 22.3, invariant 25).
"""

from __future__ import annotations

from dataclasses import replace

from ...domain.enums import (
    ActionKind,
    ObligationStatus,
    OutboundKind,
    WorkStatus,
)
from ...domain.events import LLMResult
from ...domain.proposals import LLMDecision, parse_decision
from ...errors import ValidationError
from ..apply_proposals import apply_proposals
from ..context import ReducerContext
from ..gates import evaluate_proactive
from ..outbound import create_outbound
from ..support import charge_proactive_message
from ..workitems import create_llm_work  # noqa: F401  (kept for symmetry/testing imports)
from .base import (
    CYCLE_MANDATORY,
    CYCLE_PROACTIVE,
    CYCLE_REACTIVE_OPTIONAL,
    HandlerOutcome,
)


def handle_llm_result(ctx: ReducerContext, event: LLMResult) -> HandlerOutcome:
    work = ctx.stores.work.get_work(event.work_id)
    if work is None or work.cycle_id != event.cycle_id:
        return HandlerOutcome(note="unknown_work", committed=False)
    if work.status is WorkStatus.COMPLETED:
        return HandlerOutcome(note="duplicate_result", committed=False)  # idempotent (invariant 24)

    cycle_type = str(work.snapshot.get("context", {}).get("source", {}).get("cycle_type", CYCLE_PROACTIVE))
    now = event.timestamp

    try:
        decision = parse_decision(event.result)
    except ValidationError as exc:
        return _on_parse_failure(ctx, event, cycle_type, str(exc), now)

    if cycle_type == CYCLE_MANDATORY:
        outcome = _finish_mandatory(ctx, event, decision, now)
    elif cycle_type == CYCLE_REACTIVE_OPTIONAL:
        outcome = _finish_reactive(ctx, event, decision, now)
    else:
        outcome = _finish_proactive(ctx, event, decision, now)

    ctx.stores.work.complete(event.work_id, event.event_id, now)
    return outcome


def _channel(ctx: ReducerContext, event: LLMResult) -> str:
    work = ctx.stores.work.get_work(event.work_id)
    return str(work.snapshot.get("context", {}).get("source", {}).get("channel", "cli")) if work else "cli"


def _on_parse_failure(ctx, event, cycle_type, error, now) -> HandlerOutcome:
    ctx.stores.work.complete(event.work_id, event.event_id, now)
    if cycle_type == CYCLE_MANDATORY:
        obligation = ctx.stores.work.obligation_by_work(event.work_id)
        if obligation is not None:
            ctx.stores.work.update_obligation(
                replace(obligation, status=ObligationStatus.FAILED, last_error=error)
            )
        return HandlerOutcome(note=f"mandatory_parse_failure:{error}")
    # Proactive/reactive parse failure: no output, no regeneration (DESIGN 22.3).
    return HandlerOutcome(note=f"parse_failure:{error}")


def _finish_mandatory(ctx, event, decision: LLMDecision, now) -> HandlerOutcome:
    obligation = ctx.stores.work.obligation_by_work(event.work_id)
    apply_proposals(ctx, decision.proposals, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        # Intentional silence is not a valid terminal action for a task (DESIGN 15, invariant 25).
        if obligation is not None:
            ctx.stores.work.update_obligation(
                replace(obligation, status=ObligationStatus.FAILED, last_error="non_speak_for_mandatory")
            )
        return HandlerOutcome(note="mandatory_non_speak_failure")

    message_id = create_outbound(
        ctx, kind=OutboundKind.MANDATORY, channel=_channel(ctx, event), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=obligation.source_event_id if obligation else None,
        now=now,
    )
    if obligation is not None:
        ctx.stores.work.update_obligation(
            replace(obligation, status=ObligationStatus.SATISFIED, satisfied_by_message_id=message_id)
        )
    return HandlerOutcome(deliver=True, note="mandatory_satisfied")


def _finish_reactive(ctx, event, decision: LLMDecision, now) -> HandlerOutcome:
    apply_proposals(ctx, decision.proposals, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        return HandlerOutcome(note="reactive_silence")
    # A chosen optional reply is reactive: delivered like a mandatory item, not proactive-gated.
    create_outbound(
        ctx, kind=OutboundKind.MANDATORY, channel=_channel(ctx, event), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=None, now=now,
    )
    return HandlerOutcome(deliver=True, note="reactive_reply")


def _finish_proactive(ctx, event, decision: LLMDecision, now) -> HandlerOutcome:
    useful = apply_proposals(ctx, decision.proposals, now)
    if decision.action is not ActionKind.SPEAK or not decision.message:
        ctx.stores.work.finalize_trace(
            event.cycle_id, action="silence", useful_enrichment=useful,
            pre_outbox_invalidated=False, note=None,
        )
        return HandlerOutcome(reschedule=True, note="proactive_silence_enrich" if useful else "proactive_silence")

    # Pre-outbox revalidation against CURRENT state (DESIGN 22.2, invariant 9).
    gate = evaluate_proactive(ctx, now)
    if not gate.allowed:
        # Superseded/blocked: drop; never regenerate inline (invariant 8).
        ctx.stores.work.finalize_trace(
            event.cycle_id, action="silence", useful_enrichment=useful,
            pre_outbox_invalidated=True, note=f"pre_outbox:{gate.reason}",
        )
        return HandlerOutcome(reschedule=True, note=f"pre_outbox_invalidated:{gate.reason}")

    charge_proactive_message(ctx, now)
    create_outbound(
        ctx, kind=OutboundKind.PROACTIVE, channel=_channel(ctx, event), text=decision.message,
        cycle_id=event.cycle_id, source_event_id=None, now=now,
    )
    ctx.stores.work.finalize_trace(
        event.cycle_id, action="speak", useful_enrichment=useful,
        pre_outbox_invalidated=False, note=None,
    )
    return HandlerOutcome(deliver=True, reschedule=True, note="proactive_speak")
