"""HumanMessage handler (DESIGN 13, 14, 15, 29.2).

A human message is a *stimulus*, not an RPC call. It always updates perceived state and is
always persisted (invariant 21), but only the response-required path guarantees outward
language. Task classification is conservative (invariant 12); re-prompts force the mandatory
path (invariant 13). Optional turns compete through candidate selection + a stochastic response
gate (DESIGN 14.2) and may result in intentional, non-broken silence (DESIGN 15).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from ... import ids
from ...cognition.classifier import ClassificationContext, classify
from ...cognition.gating import OptionalResponseFactors, optional_response_probability
from ...cognition.selection import select
from ...domain.enums import (
    CandidateKind,
    ConversationMode,
    EnrichmentStatus,
    MessageClass,
    ObligationStatus,
    WorkKind,
)
from ...domain.events import HumanMessage
from ...domain.runtime import CognitionTrace, ResponseObligation
from ...domain.state import ProvisionalMemory
from ...cognition.activation import half_life_to_rate_per_hour
from ..context import ReducerContext
from ..support import build_candidates, infer_mode
from ..workitems import create_embedding_work, create_llm_work
from .base import CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL, HandlerOutcome

_MAX_RECENT_TURNS = 10

_SALIENCE_BY_CLASS = {
    MessageClass.DIRECT_TASK: 0.7,
    MessageClass.TASK_QUESTION: 0.7,
    MessageClass.REPROMPT: 0.75,
    MessageClass.HIGH_INFORMATION: 0.8,
    MessageClass.STATEMENT: 0.5,
    MessageClass.SOCIAL_QUESTION: 0.45,
    MessageClass.ACKNOWLEDGEMENT: 0.15,
    MessageClass.CONVERSATION_CLOSER: 0.15,
    MessageClass.LOW_INFORMATION: 0.2,
}


def _previous_context(ctx: ReducerContext) -> ClassificationContext:
    conversation = ctx.stores.state.load_conversation()
    last_human = conversation.last_human_message_at
    last_agent = conversation.last_agent_delivered_at
    previous_silent = last_human is not None and (last_agent is None or last_agent < last_human)
    previous_text = None
    for turn in reversed(ctx.stores.outbox.recent_turns(limit=6)):
        if turn["role"] == "human":
            previous_text = turn["text"]
            break
    seconds = None
    if last_human is not None:
        seconds = (ctx.clock.now_utc() - last_human).total_seconds()
    return ClassificationContext(
        previous_human_text=previous_text,
        previous_turn_was_silent=previous_silent,
        seconds_since_previous_human=seconds,
    )


def _update_conversation(ctx: ReducerContext, now: datetime) -> None:
    conversation = ctx.stores.state.load_conversation()
    recent = (*conversation.recent_human_turn_timestamps, now)[-_MAX_RECENT_TURNS:]
    updated = replace(
        conversation,
        last_human_message_at=now,
        active_observed_silence_seconds=0.0,  # a present human resets observed silence
        recent_human_turn_timestamps=recent,
    )
    updated = replace(updated, mode=infer_mode(updated, now))
    ctx.stores.state.save_conversation(updated)


def _create_memory(ctx: ReducerContext, event: HumanMessage, salience: float, now: datetime) -> str:
    rate = half_life_to_rate_per_hour(ctx.config.memory.default_decay_half_life_hours)
    memory_id = ids.new_id(ids.PROVISIONAL_MEMORY)
    ctx.stores.memory.insert_memory(
        ProvisionalMemory(
            id=memory_id,
            event_id=event.event_id,
            text=event.text,
            activation=salience,
            salience=salience,
            decay_rate_per_hour=rate,
            created_at=now,
            last_activated_at=now,
            enrichment_status=EnrichmentStatus.RAW,
        )
    )
    return memory_id


def handle_human_message(ctx: ReducerContext, event: HumanMessage) -> HandlerOutcome:
    now = event.timestamp
    cycle_id = ids.new_id(ids.CYCLE)
    classification = classify(event.text, _previous_context(ctx))

    ctx.stores.outbox.add_human_turn(event.text, event.channel, event.event_id, now)
    _update_conversation(ctx, now)

    dispatch: list[str] = []
    memory_id: str | None = None
    if classification.embedding_eligible:
        salience = _SALIENCE_BY_CLASS.get(classification.message_class, 0.5)
        memory_id = _create_memory(ctx, event, salience, now)
        dispatch.append(
            create_embedding_work(
                ctx, memory_id=memory_id, source_event_id=event.event_id,
                cycle_id=cycle_id, now=now,
            )
        )

    if classification.possible_prior_miss:
        # Re-prompt is weak classification-quality evidence, zero adaptation weight (DESIGN 17.1).
        ctx.stores.state.add_feedback(None, "reprompt_possible_miss", 0.0, now)

    if classification.response_required:
        outcome = _mandatory_path(ctx, event, cycle_id, classification.message_class, now)
    else:
        outcome = _optional_path(ctx, event, cycle_id, memory_id, now)

    outcome.dispatch_work_ids = dispatch + outcome.dispatch_work_ids
    outcome.reschedule = True
    return outcome


def _mandatory_path(ctx, event, cycle_id, message_class, now) -> HandlerOutcome:
    obligation = ResponseObligation(
        id=ids.new_id(ids.OBLIGATION),
        source_event_id=event.event_id,
        created_at=now,
        status=ObligationStatus.PENDING,
    )
    work_id = create_llm_work(
        ctx,
        cycle_id=cycle_id,
        source_event_id=event.event_id,
        source_context={
            "kind": "human_message",
            "text": event.text,
            "message_class": message_class.value,
            "response_required": True,
            "cycle_type": CYCLE_MANDATORY,
            "channel": event.channel,
        },
        kind=WorkKind.LLM_COGNITION,
        now=now,
    )
    obligation = replace(obligation, work_id=work_id)
    ctx.stores.work.insert_obligation(obligation)
    trace = _trace(cycle_id, event, now, llm_called=True, note="mandatory_response")
    return HandlerOutcome(dispatch_work_ids=[work_id], trace=trace)


def _optional_path(ctx, event, cycle_id, memory_id, now) -> HandlerOutcome:
    candidates = build_candidates(ctx, now)
    selection = select(
        candidates,
        ctx.config.cognition.null_candidate_score,
        ctx.config.cognition.selection_temperature,
        ctx.rng,
    )
    if selection.is_nothing:
        return HandlerOutcome(trace=_trace(cycle_id, event, now, candidate_null=True, note="silence_nothing"))

    if selection.candidate.score < ctx.config.cognition.semantic_worthiness_floor:
        return HandlerOutcome(
            trace=_trace(cycle_id, event, now, candidate=selection.candidate, note="silence_low_worth")
        )

    model = ctx.stores.state.load_self_model()
    conversation = ctx.stores.state.load_conversation()
    factors = OptionalResponseFactors(
        desire=min(1.0, selection.candidate.score),
        relevance=1.0 if selection.candidate.id == memory_id else 0.5,
        initiative=model.initiative,
        inhibition=model.inhibition,
        active_mode=conversation.mode is ConversationMode.ACTIVE,
    )
    if ctx.rng.uniform() >= optional_response_probability(factors):
        return HandlerOutcome(
            trace=_trace(cycle_id, event, now, candidate=selection.candidate, note="silence_stochastic")
        )

    work_id = create_llm_work(
        ctx,
        cycle_id=cycle_id,
        source_event_id=event.event_id,
        source_context={
            "kind": "human_message",
            "text": event.text,
            "response_required": False,
            "cycle_type": CYCLE_REACTIVE_OPTIONAL,
            "channel": event.channel,
            "candidate": _candidate_context(ctx, selection.candidate),
        },
        now=now,
    )
    trace = _trace(cycle_id, event, now, candidate=selection.candidate, llm_called=True, note="reactive_optional")
    return HandlerOutcome(dispatch_work_ids=[work_id], trace=trace)


def _candidate_context(ctx: ReducerContext, candidate) -> dict:
    data = {"kind": candidate.kind.value, "id": candidate.id, "salience": candidate.score}
    if candidate.kind is CandidateKind.PROVISIONAL_MEMORY:
        memory = ctx.stores.memory.get_memory(candidate.id)
        if memory:
            data.update({"text": memory.text, "provisional_memory_id": memory.id})
    elif candidate.kind is CandidateKind.TOPIC:
        topic = ctx.stores.memory.get_topic(candidate.id)
        if topic:
            data.update({"summary": topic.summary})
    elif candidate.kind is CandidateKind.DEFERRED_INTENT:
        intent = ctx.stores.memory.get_intent(candidate.id)
        if intent:
            data.update({"intent": intent.intent})
    return data


def _trace(cycle_id, event, now, *, candidate=None, candidate_null=False, llm_called=False, note="") -> CognitionTrace:
    conversation_mode = None
    return CognitionTrace(
        cycle_id=cycle_id,
        created_at=now,
        source_event_id=event.event_id,
        trigger="HumanMessage",
        candidate_kind=candidate.kind.value if candidate else (CandidateKind.NOTHING.value if candidate_null else None),
        candidate_id=candidate.id if candidate else None,
        candidate_was_null=candidate_null,
        llm_called=llm_called,
        conversation_mode=conversation_mode,
        notes=note,
    )
