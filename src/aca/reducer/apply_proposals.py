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
from ..cognition import textsim, vectors
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
    if memory.salience < ctx.config.memory.enrichment_salience_floor:
        # Quality gate: too low-value to promote to a topic. Mark it so it isn't retried, but keep
        # it retrievable as a RAW-tier memory (DESIGN 12.3, 12.8).
        ctx.stores.memory.update_memory(
            replace(memory, enrichment_status=EnrichmentStatus.DO_NOT_ENRICH)
        )
        return False
    ctx.stores.memory.update_memory(
        replace(memory, enrichment_status=EnrichmentStatus.ENRICHED, last_activated_at=now)
    )
    summary = proposal.fields.get("topic_summary") or memory.text[:200]
    existing = _find_similar_topic(ctx, summary, memory)
    if existing is not None:
        # Near-duplicate of an existing topic: reinforce it instead of spawning a parallel topic
        # (DESIGN 12.3). Keep the stronger activation/importance and count the added evidence.
        ctx.stores.memory.update_topic(
            replace(
                existing,
                activation=max(existing.activation, memory.activation),
                importance=max(existing.importance, memory.salience),
                evidence_count=existing.evidence_count + 1,
                last_activated_at=now,
            )
        )
        return True
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
            # A freshly enriched topic is a fact/thread with no *open* loop yet. It becomes
            # "unfinished" only when a deferred intent is attached to it (an actual unresolved
            # thread), and finished again when that intent resolves (DESIGN 12.7, 16). This stops
            # every topic from being permanently unfinished and thus permanently hot.
            unfinished=False,
            source="provisional_memory",
            source_memory_id=memory.id,
        )
    )
    return True


def _memory_embedding(ctx: ReducerContext, memory_id: str | None):
    """A memory's ``(model_version, vector)`` or None (topics inherit their source memory's)."""
    if not memory_id:
        return None
    pairs = ctx.stores.memory.embeddings_for_memories([memory_id])
    return pairs[0] if pairs else None


def _find_similar_topic(ctx: ReducerContext, summary: str, memory):
    """The best existing topic to merge this enrichment into, or None (DESIGN 12.3).

    Two independent, deterministic signals, each subject to a polarity veto so a reversal never
    merges into the thing it reverses:
    - lexical: summary text overlap >= ``topic_merge_similarity`` (catches near-identical wording).
    - semantic: cosine between this memory's embedding and the topic's (inherited from its source
      memory) >= ``topic_dedup_cosine`` (catches paraphrases). The cosine threshold is deliberately
      very high so related-but-distinct topics don't merge, and only same-model vectors compare.
    A threshold of 0 disables that signal; both 0 => no merging.

    Limitation: the polarity veto is computed on the *summaries*, while the semantic signal is on
    the *memory* embeddings. A reversal expressed only in memory text but not surfaced in either
    summary could therefore slip through the semantic path; embedding-level polarity would be needed
    to close that fully. The very high cosine threshold keeps the blast radius small in practice.
    """
    lex_threshold = ctx.config.memory.topic_merge_similarity
    dedup_cos = ctx.config.memory.topic_dedup_cosine
    if lex_threshold <= 0.0 and dedup_cos <= 0.0:
        return None
    new_emb = _memory_embedding(ctx, memory.id) if dedup_cos > 0.0 else None

    best, best_score = None, 0.0
    for topic in ctx.stores.memory.all_topics(limit=50):
        if textsim.polarity_conflict(summary, topic.summary):
            continue  # a polarity flip is never a duplicate, by either signal
        lexical = textsim.similarity(summary, topic.summary) if lex_threshold > 0.0 else 0.0
        semantic = 0.0
        if new_emb is not None and topic.source_memory_id:
            topic_emb = _memory_embedding(ctx, topic.source_memory_id)
            if topic_emb is not None and topic_emb[0] == new_emb[0]:  # same model (drift-safe)
                semantic = vectors.cosine(new_emb[1], topic_emb[1])
        qualifies = (lex_threshold > 0.0 and lexical >= lex_threshold) or (
            dedup_cos > 0.0 and semantic >= dedup_cos
        )
        score = max(lexical, semantic)
        if qualifies and score > best_score:
            best, best_score = topic, score
    return best


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
    topic_id = proposal.fields.get("topic_id")
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
            topic_id=topic_id,
            provisional_memory_id=proposal.fields.get("provisional_memory_id"),
        )
    )
    _mark_topic_unfinished(ctx, topic_id, True, now)  # a deferred intent is an open thread
    return True


def _resolve_intent(ctx: ReducerContext, proposal: Proposal, now: datetime) -> bool:
    intent = ctx.stores.memory.get_intent(proposal.fields["intent_id"])
    if intent is None:
        return False
    ctx.stores.memory.update_intent(replace(intent, status="resolved", last_activated_at=now))
    # The topic is finished again once no other pending intent still references it (DESIGN 12.7).
    if intent.topic_id and not ctx.stores.memory.has_pending_intent_for_topic(
        intent.topic_id, exclude_intent_id=intent.id
    ):
        _mark_topic_unfinished(ctx, intent.topic_id, False, now)
    return True


def _mark_topic_unfinished(ctx: ReducerContext, topic_id: str | None, unfinished: bool, now: datetime) -> None:
    """Flip a topic's open-thread flag, if the topic exists (no-op for a memory-only intent)."""
    if not topic_id:
        return
    topic = ctx.stores.memory.get_topic(topic_id)
    if topic is not None and topic.unfinished != unfinished:
        ctx.stores.memory.update_topic(replace(topic, unfinished=unfinished, last_activated_at=now))


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
