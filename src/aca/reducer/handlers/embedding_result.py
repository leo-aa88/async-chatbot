"""EmbeddingResult handler (DESIGN 29.3).

Attaches a local embedding to its provisional memory for later non-lexical retrieval. Embedding
completion must **not** recursively launch generative cognition (invariant 7). Idempotent by
``work_id``: a duplicate result after commit is dropped.
"""

from __future__ import annotations

from dataclasses import replace

from ...domain.enums import WorkStatus
from ...domain.events import EmbeddingResult
from ..context import ReducerContext
from .base import HandlerOutcome


def handle_embedding_result(ctx: ReducerContext, event: EmbeddingResult) -> HandlerOutcome:
    work = ctx.stores.work.get_work(event.work_id)
    if work is not None and work.status is WorkStatus.COMPLETED:
        return HandlerOutcome(note="duplicate_embedding", committed=False)

    memory = ctx.stores.memory.get_memory(event.provisional_memory_id)
    if memory is not None:
        ctx.stores.memory.insert_embedding(
            event.embedding_id, memory.id, event.model_version, event.vector, event.timestamp
        )
        ctx.stores.memory.update_memory(replace(memory, embedding_id=event.embedding_id))

    if work is not None:
        ctx.stores.work.complete(event.work_id, event.event_id, event.timestamp)
    # Embedding does not change wake-relevant state; no reschedule (DESIGN 29.3 step 7).
    return HandlerOutcome(note="embedding_attached")
