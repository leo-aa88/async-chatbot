"""ReconcileEmbeddings handler: backfill missing topic-summary embeddings (DESIGN 12.3).

Enqueues one embedding job per topic that has no summary vector yet — topics created before the
embedding pipeline existed, or before a daemon carrying it ran. Without this, semantic dedup,
dominance, advance-rate, and the continuity gate stay blind to all such history.

Idempotent: topics already embedded (the store query) or with an embedding job already in flight
(the pending-work guard) are skipped, so firing this at every startup and on demand never piles up
duplicate work. Jobs run through the normal embedding path, so each landing also fires the
post-embedding semantic dedup (``embedding_result``) — merging any paraphrase duplicates that
predate the pipeline.
"""

from __future__ import annotations

from ...domain.events import ReconcileEmbeddings
from ..context import ReducerContext
from ..workitems import create_topic_embedding_work
from .base import HandlerOutcome

# Bound one reconcile so a large or first-time backlog is drained over several passes rather than
# dispatched as one burst; the next reconcile picks up the remainder.
_MAX_PER_RECONCILE = 200


def handle_reconcile_embeddings(ctx: ReducerContext, event: ReconcileEmbeddings) -> HandlerOutcome:
    # In-flight = PENDING *or* RUNNING (leased): a job dispatched by an earlier pass or recovered at
    # startup is already RUNNING, so pending()-only would let a second job be enqueued per topic.
    in_flight = ctx.stores.work.active_embedding_topic_ids()
    enqueued = 0
    for topic_id, summary in ctx.stores.memory.topics_without_embedding(limit=_MAX_PER_RECONCILE):
        if topic_id in in_flight:
            continue  # an embedding job for this topic is already queued
        ctx.deferred_work_ids.append(
            create_topic_embedding_work(
                ctx, topic_id=topic_id, summary=summary,
                source_event_id=event.event_id, now=event.timestamp,
            )
        )
        enqueued += 1
    return HandlerOutcome(committed=enqueued > 0, note=f"reconcile_embeddings:{enqueued}")
