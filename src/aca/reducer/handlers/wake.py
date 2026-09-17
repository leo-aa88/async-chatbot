"""StochasticWake handler (DESIGN 10.6, 19, 29, invariant 34).

A wake is validated before any cognition: lifecycle must be RUNNING and the wake's
``runtime_session_id`` and ``scheduler_generation`` must be current. A stale wake (an overdue
timer that fired after resume) is harmlessly dropped. A valid wake samples one candidate
against NOTHING, applies the cheap worthiness gate and hard proactive gates, and — only if
justified — dispatches at most one proactive generative call (invariant 3).
"""

from __future__ import annotations

from dataclasses import replace

from ... import ids
from ...cognition import budgets as budget
from ...domain.enums import CandidateKind, EnrichmentStatus, LifecycleState, WorkKind
from ...domain.events import StochasticWake
from ...domain.runtime import CognitionTrace
from ..context import ReducerContext
from ..gates import evaluate_proactive
from ..support import build_candidates, read_proactive_usage
from ..workitems import create_llm_work
from .base import CYCLE_PROACTIVE, HandlerOutcome


def _is_stale(ctx: ReducerContext, event: StochasticWake) -> bool:
    identity = ctx.stores.identity.load_identity()
    if identity is None or identity.lifecycle_state is not LifecycleState.RUNNING:
        return True
    if event.runtime_session_id != ctx.runtime_session_id:
        return True
    return event.scheduler_generation != ctx.scheduler_generation


def handle_stochastic_wake(ctx: ReducerContext, event: StochasticWake) -> HandlerOutcome:
    if _is_stale(ctx, event):
        # Harmless stale event: no cognition, no reschedule (the current generation owns timing).
        return HandlerOutcome(note="stale_wake", committed=False)

    now = event.timestamp
    cycle_id = ids.new_id(ids.CYCLE)
    candidates = build_candidates(ctx, now)
    from ...cognition.selection import select

    selection = select(
        candidates,
        ctx.config.cognition.null_candidate_score,
        ctx.config.cognition.selection_temperature,
        ctx.rng,
    )
    if selection.is_nothing:
        return HandlerOutcome(
            reschedule=True,
            trace=_trace(cycle_id, now, candidate_null=True, action="silence", note="nothing"),
        )

    candidate = selection.candidate
    if candidate.score < ctx.config.cognition.semantic_worthiness_floor:
        return HandlerOutcome(
            reschedule=True,
            trace=_trace(cycle_id, now, candidate=candidate, action="silence", note="low_worth"),
        )

    usage = read_proactive_usage(ctx, now)
    if not budget.proactive_llm_call_allowed(usage, ctx.config.budgets).allowed:
        return HandlerOutcome(
            reschedule=True,
            trace=_trace(cycle_id, now, candidate=candidate, action="silence", note="llm_budget"),
        )

    gate = evaluate_proactive(ctx, now)
    needs_enrichment = _needs_enrichment(ctx, candidate)
    if not gate.allowed and not needs_enrichment:
        note = _silence_note(ctx, candidate, gate)
        return HandlerOutcome(
            reschedule=True,
            trace=_trace(cycle_id, now, candidate=candidate, action="silence", note=note),
        )

    _materialize_selected(ctx, candidate, now)
    _claim_enrichment(ctx, candidate, now)  # RAW -> PENDING so concurrent wakes don't re-enrich
    from ..support import charge_proactive_llm

    charge_proactive_llm(ctx, now)  # a proactive generative call is being dispatched
    work_id = create_llm_work(
        ctx,
        cycle_id=cycle_id,
        source_event_id=event.event_id,
        source_context={
            "kind": "stochastic_wake",
            "response_required": False,
            "cycle_type": CYCLE_PROACTIVE,
            "channel": "cli",
            "output_eligible": gate.allowed,
            "candidate": _candidate_context(ctx, candidate),
        },
        kind=WorkKind.LLM_COGNITION if gate.allowed else WorkKind.LLM_ENRICHMENT,
        now=now,
    )
    return HandlerOutcome(
        dispatch_work_ids=[work_id],
        reschedule=True,
        trace=_trace(cycle_id, now, candidate=candidate, llm_called=True, note="proactive_dispatch"),
    )


def _silence_note(ctx: ReducerContext, candidate, gate) -> str:
    """Observability: distinguish a candidate held back by the enrichment quality gate.

    A RAW provisional memory that lost enrichment eligibility purely because it fell below the
    salience floor is recorded as ``enrichment_gated:low_salience`` (not a generic ``blocked``),
    so ``aca logs`` shows which junk memories were kept out of the topic set (DESIGN 12.3).
    """
    if candidate.kind is CandidateKind.PROVISIONAL_MEMORY:
        memory = ctx.stores.memory.get_memory(candidate.id)
        if memory is not None and memory.enrichment_status is EnrichmentStatus.RAW:
            return "enrichment_gated:low_salience"
    return f"blocked:{gate.reason}"


def _needs_enrichment(ctx: ReducerContext, candidate) -> bool:
    if candidate.kind is not CandidateKind.PROVISIONAL_MEMORY:
        return False
    memory = ctx.stores.memory.get_memory(candidate.id)
    return memory is not None and _enrichment_eligible(ctx, memory)


def _enrichment_eligible(ctx: ReducerContext, memory) -> bool:
    """RAW and salient enough to be worth promoting to a topic (quality gate, DESIGN 12.3/12.8).

    Low-salience memories (short acknowledgements, filler) stay RAW and retrievable but are never
    enriched into topics — this keeps junk like "ok I get" out of the topic set.
    """
    return (
        memory.enrichment_status is EnrichmentStatus.RAW
        and memory.salience >= ctx.config.memory.enrichment_salience_floor
    )


def _claim_enrichment(ctx: ReducerContext, candidate, now) -> None:
    """Claim an enrichment-eligible provisional memory (DESIGN 12.8, invariant 23).

    Marking it PENDING_ENRICHMENT means ``_needs_enrichment`` returns False for any concurrent
    wake, so the same memory can't have two enrichment jobs in flight. Only salient-enough
    memories are claimed; the claim is released back to RAW (or DO_NOT_ENRICH after the attempt
    cap) when the result is processed if it wasn't actually enriched.
    """
    if candidate.kind is not CandidateKind.PROVISIONAL_MEMORY:
        return
    memory = ctx.stores.memory.get_memory(candidate.id)
    if memory is not None and _enrichment_eligible(ctx, memory):
        ctx.stores.memory.update_memory(
            replace(memory, enrichment_status=EnrichmentStatus.PENDING_ENRICHMENT)
        )


def _materialize_selected(ctx: ReducerContext, candidate, now) -> None:
    """Reset the selected item's decay baseline to now (reinforcement, DESIGN 12.6)."""
    if candidate.kind is CandidateKind.PROVISIONAL_MEMORY:
        memory = ctx.stores.memory.get_memory(candidate.id)
        if memory:
            ctx.stores.memory.update_memory(replace(memory, last_activated_at=now))
    elif candidate.kind is CandidateKind.TOPIC:
        topic = ctx.stores.memory.get_topic(candidate.id)
        if topic:
            ctx.stores.memory.update_topic(replace(topic, last_activated_at=now))


def _candidate_context(ctx: ReducerContext, candidate) -> dict:
    data = {"kind": candidate.kind.value, "id": candidate.id, "salience": candidate.score}
    if candidate.kind is CandidateKind.PROVISIONAL_MEMORY:
        memory = ctx.stores.memory.get_memory(candidate.id)
        if memory:
            data.update({"text": memory.text, "provisional_memory_id": memory.id})
    elif candidate.kind is CandidateKind.TOPIC:
        topic = ctx.stores.memory.get_topic(candidate.id)
        if topic:
            data["summary"] = topic.summary
    elif candidate.kind is CandidateKind.DEFERRED_INTENT:
        intent = ctx.stores.memory.get_intent(candidate.id)
        if intent:
            data["intent"] = intent.intent
    return data


def _trace(
    cycle_id, now, *, candidate=None, candidate_null=False, llm_called=False, action=None, note="",
) -> CognitionTrace:
    # A wake that terminates without dispatching an LLM records action="silence" now; a dispatched
    # proactive cycle leaves action=None until the LLMResult finalizes it (speak/silence).
    return CognitionTrace(
        cycle_id=cycle_id,
        created_at=now,
        trigger="StochasticWake",
        candidate_kind=candidate.kind.value if candidate else (CandidateKind.NOTHING.value if candidate_null else None),
        candidate_id=candidate.id if candidate else None,
        candidate_was_null=candidate_null,
        llm_called=llm_called,
        action=action,
        notes=note,
    )
