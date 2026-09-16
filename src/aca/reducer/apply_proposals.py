"""Applying validated, clamped LLM proposals to durable state (DESIGN 22, 17.2).

    The model interprets; deterministic policy assigns authority and magnitude.

Proposals arriving here are already whitelisted and numerically clamped (``domain.proposals``).
This module maps each to a bounded, auditable state change. No proposal may create a recursive
cognition event (invariant 7): enrichment and activation edits are terminal state writes.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .. import ids
from ..cognition.activation import half_life_to_rate_per_hour, reinforced
from ..domain.enums import EnrichmentStatus
from ..domain.proposals import Proposal
from ..domain.state import DeferredIntent, Topic
from .context import ReducerContext

_DEFERRED_INTENT_TTL_HOURS = 48.0

# Deterministic policy: engagement classification -> bounded adaptation weight (DESIGN 17.2).
_ENGAGEMENT_WEIGHTS = {
    "explicit_positive": 0.10,
    "substantive": 0.05,
    "substantive_engagement": 0.05,
    "short": 0.0,
    "polite": 0.0,
    "no_response": 0.0,
    "explicit_negative": -0.10,
}


def apply_proposals(ctx: ReducerContext, proposals: tuple[Proposal, ...], now: datetime) -> bool:
    """Apply all proposals; return whether any produced a useful state change (DESIGN 19.1)."""
    useful = False
    for proposal in proposals:
        useful = _apply_one(ctx, proposal, now) or useful
    return useful


def _apply_one(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    handler = _DISPATCH.get(proposal.type)
    return handler(ctx, proposal, now) if handler else False


def _enrich_memory(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    memory = ctx.stores.memory.get_memory(proposal.fields["provisional_memory_id"])
    if memory is None:
        return False
    if memory.enrichment_status is EnrichmentStatus.ENRICHED:
        # Idempotent by memory identity: a second concurrent enrichment result must not create a
        # duplicate topic (DESIGN 12.8, invariant 23).
        return False
    ctx.stores.memory.update_memory(
        replace(memory, enrichment_status=EnrichmentStatus.ENRICHED, last_activated_at=now)
    )
    summary = proposal.fields.get("topic_summary") or memory.text[:200]
    rate = half_life_to_rate_per_hour(ctx.config.memory.default_decay_half_life_hours)
    ctx.stores.memory.insert_topic(
        Topic(
            id=ids.new_id(ids.TOPIC),
            summary=summary,
            tags=tuple(proposal.fields.get("tags", ())),
            activation=memory.activation,
            importance=memory.salience,
            decay_rate_per_hour=rate,
            created_at=now,
            last_activated_at=now,
            unfinished=True,
            source="provisional_memory",
        )
    )
    return True


def _adjust_topic(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    topic = ctx.stores.memory.get_topic(proposal.fields["topic_id"])
    if topic is None:
        return False
    new_activation = reinforced(topic.activation, proposal.fields["delta"])
    ctx.stores.memory.update_topic(
        replace(topic, activation=max(0.0, new_activation), last_activated_at=now)
    )
    return True


def _create_intent(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    rate = half_life_to_rate_per_hour(ctx.config.memory.default_decay_half_life_hours)
    ctx.stores.memory.insert_intent(
        DeferredIntent(
            id=ids.new_id(ids.DEFERRED_INTENT),
            intent=proposal.fields["intent"],
            activation=0.6,
            decay_rate_per_hour=rate,
            created_at=now,
            last_activated_at=now,
            expires_at=now + timedelta(hours=_DEFERRED_INTENT_TTL_HOURS),
            status="pending",
            topic_id=proposal.fields.get("topic_id"),
            provisional_memory_id=proposal.fields.get("provisional_memory_id"),
        )
    )
    return True


def _resolve_intent(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    intent = ctx.stores.memory.get_intent(proposal.fields["intent_id"])
    if intent is None:
        return False
    ctx.stores.memory.update_intent(replace(intent, status="resolved", last_activated_at=now))
    return True


def _engagement(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    classification = proposal.fields.get("classification", "")
    weight = _ENGAGEMENT_WEIGHTS.get(classification, 0.0)
    ctx.stores.state.add_feedback(proposal.fields["target_action_id"], classification, weight, now)
    return False  # observation recorded, but not a "useful enrichment" for the silent-rate metric


_DISPATCH = {
    "ENRICH_PROVISIONAL_MEMORY": _enrich_memory,
    "ADJUST_TOPIC_ACTIVATION": _adjust_topic,
    "CREATE_DEFERRED_INTENT": _create_intent,
    "RESOLVE_DEFERRED_INTENT": _resolve_intent,
    "USER_ENGAGEMENT_OBSERVATION": _engagement,
}
