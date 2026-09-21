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

    merged = False
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
            merged = _merge_duplicate_topic(
                ctx, topic, event.model_version, event.vector, event.timestamp
            )
    else:
        memory = ctx.stores.memory.get_memory(event.provisional_memory_id)
        if memory is not None:
            ctx.stores.memory.insert_embedding(
                event.embedding_id, memory.id, event.model_version, event.vector, event.timestamp
            )
            ctx.stores.memory.update_memory(replace(memory, embedding_id=event.embedding_id))

    if work is not None:
        ctx.stores.work.complete(event.work_id, event.event_id, event.timestamp)
    # A plain embedding attachment changes no wake-relevant signal, so no reschedule. A merge does:
    # it resets the survivor's decay baseline (last_activated_at=now) and can raise its effective
    # activation above the pre-merge max, and it changes count_unfinished_topics — both feed
    # wake_signals, so the current timer is stale and we reschedule (DESIGN 29.3.7).
    if merged:
        return HandlerOutcome(reschedule=True, note="embedding_attached_merged")
    return HandlerOutcome(note="embedding_attached")


def _merge_duplicate_topic(
    ctx: ReducerContext, topic: Topic, model_version: str, vector: list[float], now: datetime
) -> bool:
    """Merge ``topic`` into an existing near-duplicate (by summary cosine). Return whether it merged.

    Compares only same-model summary vectors (drift-safe), vetoes polarity flips, and requires
    cosine >= ``topic_dedup_cosine`` (deliberately very high, so related-but-distinct topics stay
    separate). The surviving topic is the pre-existing one; it absorbs evidence/activation, the
    duplicate's deferred intents, its expression/in-flight suppression, and (if it had none) its
    source-memory provenance, then the duplicate is deleted (DESIGN 12.3). 0 disables.
    """
    dedup = ctx.config.memory.topic_dedup_cosine
    if dedup <= 0.0:
        return False
    # Scan every topic embedded under this model, not just the hottest 50: dedup is identity, so a
    # paraphrase of a cold topic must still merge (get_topic is loaded only for a near-duplicate).
    best, best_cos = None, 0.0
    for other_id, other_vec in ctx.stores.memory.all_topic_embeddings(
        model_version, exclude_topic_id=topic.id
    ):
        c = vectors.cosine(vector, other_vec)
        if c < dedup or c <= best_cos:
            continue
        other = ctx.stores.memory.get_topic(other_id)
        if other is None or textsim.polarity_conflict(topic.summary, other.summary):
            continue
        best, best_cos = other, c
    if best is None:
        return False
    ctx.stores.memory.reassign_topic_for_intents(topic.id, best.id)
    # Suppression must follow the merge, or the *consolidated* identity becomes the one that nags:
    # move the duplicate's expression record and any in-flight proactive item onto the survivor
    # (DESIGN 6, 11.3, 12.7).
    ctx.stores.memory.merge_expression("TOPIC", topic.id, "TOPIC", best.id)
    ctx.stores.outbox.reassign_proactive_candidate("TOPIC", topic.id, "TOPIC", best.id)
    # The survivor is an open thread if either topic had a pending intent (DESIGN 12.7, 16).
    unfinished = best.unfinished or ctx.stores.memory.has_pending_intent_for_topic(best.id)
    ctx.stores.memory.update_topic(
        replace(
            best,
            activation=max(best.activation, topic.activation),
            importance=max(best.importance, topic.importance),
            evidence_count=best.evidence_count + topic.evidence_count,
            unfinished=unfinished,
            source_memory_id=best.source_memory_id or topic.source_memory_id,
            last_activated_at=now,
        )
    )
    ctx.stores.memory.delete_topic(topic.id)
    return True
