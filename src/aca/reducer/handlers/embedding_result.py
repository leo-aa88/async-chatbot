"""EmbeddingResult handler (DESIGN 29.3).

Attaches a local embedding to its provisional memory for later non-lexical retrieval. Embedding
completion must **not** recursively launch generative cognition (invariant 7). Idempotent by
``work_id``: a duplicate result after commit is dropped.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from ...cognition import textsim, vectors
from ...domain.enums import WorkStatus
from ...domain.events import EmbeddingResult
from ...domain.state import Topic
from ..context import ReducerContext
from .base import HandlerOutcome


def handle_embedding_result(ctx: ReducerContext, event: EmbeddingResult) -> HandlerOutcome:
    work = ctx.stores.work.get_work(event.work_id)
    if work is not None and work.status is WorkStatus.COMPLETED:
        return HandlerOutcome(note="duplicate_embedding", committed=False)

    if event.topic_id:
        # Topic-summary embedding (DESIGN 12.3): stored in its own table, keyed by topic id.
        topic = ctx.stores.memory.get_topic(event.topic_id)
        if topic is not None:
            ctx.stores.memory.insert_topic_embedding(
                event.topic_id, event.model_version, event.vector, event.timestamp
            )
            # Semantic dedup (stage 2), now that the SUMMARY vector exists: merge this topic into a
            # near-duplicate existing topic. This is the paraphrase-catching layer the enrich-time
            # lexical check can't do; it runs on summaries, not raw source-memory fragments.
            _merge_duplicate_topic(ctx, topic, event.model_version, event.vector, event.timestamp)
    else:
        memory = ctx.stores.memory.get_memory(event.provisional_memory_id)
        if memory is not None:
            ctx.stores.memory.insert_embedding(
                event.embedding_id, memory.id, event.model_version, event.vector, event.timestamp
            )
            ctx.stores.memory.update_memory(replace(memory, embedding_id=event.embedding_id))

    if work is not None:
        ctx.stores.work.complete(event.work_id, event.event_id, event.timestamp)
    # A topic merge here reduces the topic set but doesn't create a new wake opportunity (it only
    # collapses duplicates), so — like plain embedding attachment — no reschedule (DESIGN 29.3.7).
    return HandlerOutcome(note="embedding_attached")


def _merge_duplicate_topic(
    ctx: ReducerContext, topic: Topic, model_version: str, vector: list[float], now: datetime
) -> None:
    """Merge ``topic`` into an existing near-duplicate (by summary cosine), or do nothing.

    Compares only same-model summary vectors (drift-safe), vetoes polarity flips, and requires
    cosine >= ``topic_dedup_cosine`` (deliberately very high, so related-but-distinct topics stay
    separate). The surviving topic is the pre-existing one; it absorbs evidence/activation and the
    duplicate's deferred intents, then the duplicate is deleted (DESIGN 12.3). 0 disables.
    """
    dedup = ctx.config.memory.topic_dedup_cosine
    if dedup <= 0.0:
        return
    # Scans the 50 most-recently-active topics (same bound as the enrich-time lexical check); a
    # duplicate older than that is left to lexical/future comparison rather than an unbounded scan.
    others = [t for t in ctx.stores.memory.all_topics(limit=50) if t.id != topic.id]
    embeddings = ctx.stores.memory.topic_embeddings_by_id([t.id for t in others])
    best, best_cos = None, 0.0
    for other in others:
        emb = embeddings.get(other.id)
        if emb is None or emb[0] != model_version:  # missing / cross-model -> not comparable
            continue
        if textsim.polarity_conflict(topic.summary, other.summary):
            continue
        c = vectors.cosine(vector, emb[1])
        if c >= dedup and c > best_cos:
            best, best_cos = other, c
    if best is None:
        return
    ctx.stores.memory.reassign_topic_for_intents(topic.id, best.id)
    # The survivor is an open thread if either topic had a pending intent (DESIGN 12.7, 16).
    unfinished = best.unfinished or ctx.stores.memory.has_pending_intent_for_topic(best.id)
    ctx.stores.memory.update_topic(
        replace(
            best,
            activation=max(best.activation, topic.activation),
            importance=max(best.importance, topic.importance),
            evidence_count=best.evidence_count + topic.evidence_count,
            unfinished=unfinished,
            last_activated_at=now,
        )
    )
    ctx.stores.memory.delete_topic(topic.id)
