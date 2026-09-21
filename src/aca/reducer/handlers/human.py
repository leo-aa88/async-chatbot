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
from ...cognition.activation import half_life_to_rate_per_hour
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
from ..context import ReducerContext
from ..discourse import is_focus_setting
from ..support import build_candidates, infer_mode_for
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

# Low-obligation, low-substance optional turns: persisted, but never earn a reactive reply
# (DESIGN 13.6 "acknowledgement/closer -> silence allowed", 15 silence-as-action).
_LOW_SUBSTANCE_CLASSES = frozenset(
    {
        MessageClass.ACKNOWLEDGEMENT,
        MessageClass.CONVERSATION_CLOSER,
        MessageClass.LOW_INFORMATION,
    }
)


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


def _update_conversation(ctx: ReducerContext, now: datetime, *, focus_memory_id: str | None) -> None:
    conversation = ctx.stores.state.load_conversation()
    recent = (*conversation.recent_human_turn_timestamps, now)[-_MAX_RECENT_TURNS:]
    updated = replace(
        conversation,
        last_human_message_at=now,
        active_observed_silence_seconds=0.0,  # a present human resets observed silence
        recent_human_turn_timestamps=recent,
        focus_memory_id=focus_memory_id,
    )
    updated = replace(updated, mode=infer_mode_for(ctx, updated, now))
    ctx.stores.state.save_conversation(updated)


def _next_focus(
    current: str | None, *, focus_setting: bool, memory_id: str | None, pre_mode: ConversationMode
) -> str | None:
    """The discourse focus after this turn (DESIGN §34.4).

    A subject survives a lapse into `DORMANT` only if a new substantive memory re-establishes one:
    a focus-setting turn that produced a memory sets it; any other turn defers to the pre-turn mode
    — keep the known-good subject while the conversation is live, clear it after dormancy so a bare
    acknowledgement can't revive a stale subject.
    """
    if focus_setting and memory_id is not None:
        return memory_id
    if pre_mode is ConversationMode.DORMANT:
        return None
    return current


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

    # The focus decision needs the PRE-turn mode (before cadence is refreshed) and this turn's new
    # memory id, so compute both before updating conversation state (DESIGN §34.4).
    pre_conversation = ctx.stores.state.load_conversation()
    pre_mode = infer_mode_for(ctx, pre_conversation, now)

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

    focus = _next_focus(
        pre_conversation.focus_memory_id,
        focus_setting=is_focus_setting(event.text, classification.message_class),
        memory_id=memory_id,
        pre_mode=pre_mode,
    )
    _update_conversation(ctx, now, focus_memory_id=focus)

    if classification.possible_prior_miss:
        # Re-prompt is weak classification-quality evidence, zero adaptation weight (DESIGN 17.1).
        ctx.stores.state.add_feedback(None, "reprompt_possible_miss", 0.0, now)

    if classification.response_required:
        outcome = _mandatory_path(ctx, event, cycle_id, classification.message_class, now)
    elif classification.message_class in _LOW_SUBSTANCE_CLASSES:
        # Acknowledgements, closers, and low-information turns are persisted but earn no reactive
        # reply — silence is a valid, first-class action here (DESIGN 13.6, 15). This also stops a
        # stray token from triggering a reply merely because some older memory is salient.
        outcome = HandlerOutcome(
            trace=_trace(cycle_id, event, now, action="silence", note="silence_low_substance")
        )
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
    trace = _trace(cycle_id, event, now, llm_called=True, note="mandatory_response",
                   cycle_type=CYCLE_MANDATORY)
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
        return HandlerOutcome(
            trace=_trace(cycle_id, event, now, candidate_null=True, action="silence", note="silence_nothing")
        )

    if selection.candidate.score < ctx.config.cognition.semantic_worthiness_floor:
        return HandlerOutcome(
            trace=_trace(cycle_id, event, now, candidate=selection.candidate, action="silence",
                         note="silence_low_worth")
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
            trace=_trace(cycle_id, event, now, candidate=selection.candidate, action="silence",
                         note="silence_stochastic")
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


def _trace(
    cycle_id, event, now, *, candidate=None, candidate_null=False, llm_called=False,
    action=None, note="", cycle_type=CYCLE_REACTIVE_OPTIONAL,
) -> CognitionTrace:
    # A branch that terminates without dispatching an LLM records its outcome immediately
    # (action="silence"); a dispatched branch leaves action=None until the LLMResult finalizes it.
    # cycle_type is fixed at dispatch (mandatory vs reactive) and never overwritten by finalize.
    return CognitionTrace(
        cycle_id=cycle_id,
        created_at=now,
        source_event_id=event.event_id,
        trigger="HumanMessage",
        cycle_type=cycle_type,
        candidate_kind=candidate.kind.value if candidate else (CandidateKind.NOTHING.value if candidate_null else None),
        candidate_id=candidate.id if candidate else None,
        candidate_was_null=candidate_null,
        llm_called=llm_called,
        action=action,
        notes=note,
    )
