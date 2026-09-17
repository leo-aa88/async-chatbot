"""Shared reducer helpers: mode inference, decay, candidate pool, wake signals, budgets.

Pure-ish helpers used by multiple handlers (DRY). They read through the stores and the injected
clock but never advance ``state_revision`` — that stays in ``reducer.py``.
"""

from __future__ import annotations

from datetime import datetime

from ..cognition import activation as act
from ..cognition import budgets as budget
from ..cognition.scheduler import WakeSignals
from ..cognition.selection import Candidate
from ..domain.enums import CandidateKind, ConversationMode
from ..domain.state import ConversationState, DeferredIntent, ProvisionalMemory, Topic
from .context import ReducerContext


def infer_mode(
    conversation: ConversationState,
    now: datetime,
    *,
    active_within_seconds: float = 180.0,
    idle_within_seconds: float = 1800.0,
) -> ConversationMode:
    """Infer conversation liveness from recent human cadence (DESIGN 7.4).

    Thresholds are injected from ``config.conversation`` (defaults match the production values)
    so proactive behavior can be exercised with short windows during testing (DESIGN 31.1).
    """
    last = conversation.last_human_message_at
    if last is None:
        return ConversationMode.DORMANT
    gap = (now - last).total_seconds()
    if gap <= active_within_seconds:
        return ConversationMode.ACTIVE
    if gap <= idle_within_seconds:
        return ConversationMode.IDLE
    return ConversationMode.DORMANT


def infer_mode_for(ctx: ReducerContext, conversation: ConversationState, now: datetime) -> ConversationMode:
    """Infer conversation mode using this agent's configured cadence thresholds."""
    return infer_mode(
        conversation, now,
        active_within_seconds=ctx.config.conversation.active_within_seconds,
        idle_within_seconds=ctx.config.conversation.idle_within_seconds,
    )


def effective_memory_activation(ctx: ReducerContext, m: ProvisionalMemory, now: datetime) -> float:
    persistence = ctx.stores.state.load_self_model().persistence
    return act.decayed_activation(
        m.activation, m.last_activated_at, now, m.decay_rate_per_hour,
        persistence=persistence, alpha=ctx.config.memory.persistence_decay_coefficient,
    )


def effective_topic_activation(ctx: ReducerContext, t: Topic, now: datetime) -> float:
    persistence = ctx.stores.state.load_self_model().persistence
    return act.decayed_activation(
        t.activation, t.last_activated_at, now, t.decay_rate_per_hour,
        persistence=persistence, alpha=ctx.config.memory.persistence_decay_coefficient,
    )


def effective_intent_activation(ctx: ReducerContext, i: DeferredIntent, now: datetime) -> float:
    persistence = ctx.stores.state.load_self_model().persistence
    return act.decayed_activation(
        i.activation, i.last_activated_at, now, i.decay_rate_per_hour,
        persistence=persistence, alpha=ctx.config.memory.persistence_decay_coefficient,
    )


def build_candidates(ctx: ReducerContext, now: datetime) -> list[Candidate]:
    """Assemble the scored candidate pool (topics + intents + memories) above the floor.

    ``score`` is the pre-temperature logit combining effective activation, importance/salience,
    and unfinished status (DESIGN 12.7, 19). NOTHING is added later by ``selection.select``.
    """
    floor = ctx.config.memory.candidate_activation_floor
    candidates: list[Candidate] = []

    for topic in ctx.stores.memory.all_topics():
        a = effective_topic_activation(ctx, topic, now)
        if a < floor:
            continue
        score = a + 0.5 * topic.importance + (0.3 if topic.unfinished else 0.0)
        candidates.append(Candidate(CandidateKind.TOPIC, topic.id, score))

    for intent in ctx.stores.memory.pending_intents():
        if intent.expires_at <= now:
            continue
        a = effective_intent_activation(ctx, intent, now)
        if a < floor:
            continue
        candidates.append(Candidate(CandidateKind.DEFERRED_INTENT, intent.id, a + 0.2))

    for memory in ctx.stores.memory.recent_memories():
        a = effective_memory_activation(ctx, memory, now)
        if a < floor:
            continue
        score = a + 0.5 * memory.salience
        candidates.append(Candidate(CandidateKind.PROVISIONAL_MEMORY, memory.id, score))

    return candidates


def wake_signals(ctx: ReducerContext, candidates: list[Candidate], now: datetime) -> WakeSignals:
    """Derive the hazard signals. Silence uses *active-observed* time only (DESIGN 10, inv 31)."""
    conversation = ctx.stores.state.load_conversation()
    silence_hours = conversation.active_observed_silence_seconds / 3600.0
    max_salience = max((c.score for c in candidates), default=0.0)
    max_salience = max(0.0, min(1.0, max_salience))
    unfinished = ctx.stores.memory.count_unfinished_topics()
    return WakeSignals(
        active_observed_silence_hours=silence_hours,
        max_candidate_salience=max_salience,
        unfinished_count=unfinished,
    )


# --- budget helpers (window-keyed; never bank across downtime, DESIGN 25) -----------------
def read_proactive_usage(ctx: ReducerContext, now: datetime) -> budget.ProactiveUsage:
    day = budget.day_window_id(now)
    hour = budget.hour_window_id(now)
    return budget.ProactiveUsage(
        llm_calls_today=ctx.stores.state.budget_count("llm_day", day),
        llm_calls_this_hour=ctx.stores.state.budget_count("llm_hour", hour),
        messages_this_hour=ctx.stores.state.budget_count("msg_hour", hour),
    )


def charge_proactive_llm(ctx: ReducerContext, now: datetime) -> None:
    ctx.stores.state.increment_budget("llm_day", budget.day_window_id(now))
    ctx.stores.state.increment_budget("llm_hour", budget.hour_window_id(now))


def charge_proactive_message(ctx: ReducerContext, now: datetime) -> None:
    ctx.stores.state.increment_budget("msg_hour", budget.hour_window_id(now))
