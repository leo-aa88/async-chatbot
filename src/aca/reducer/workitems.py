"""Durable semantic-work creation (DESIGN 8.2, 23.5).

Work is persisted PENDING with an immutable snapshot *before* dispatch, so ownership survives a
crash and can be reclaimed by lease (invariant 24). Handlers call these helpers; the service
later dispatches the returned work id to a worker. At most one generative work item per cycle
(invariant 3) is the caller's responsibility — these helpers create exactly one each.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .. import ids
from ..cognition.snapshot import build_snapshot
from ..domain.enums import WorkKind, WorkStatus
from ..domain.runtime import WorkItem
from .context import ReducerContext


def _agent_state_summary(ctx: ReducerContext) -> dict[str, Any]:
    model = ctx.stores.state.load_self_model()
    conversation = ctx.stores.state.load_conversation()
    return {
        "initiative": model.initiative,
        "inhibition": model.inhibition,
        "persistence": model.persistence,
        "mode": conversation.mode.value,
        "dominant_topic": model.dominant_topic,
    }


def create_llm_work(
    ctx: ReducerContext,
    *,
    cycle_id: str,
    source_event_id: str,
    source_context: dict[str, Any],
    kind: WorkKind = WorkKind.LLM_COGNITION,
    now: datetime,
) -> str:
    """Persist one generative work item with a canonical snapshot; return its ``work_id``."""
    work_id = ids.new_id(ids.WORK_ITEM)
    basis_revision = ctx.stores.state.state_revision()
    snapshot = build_snapshot(
        cycle_id=cycle_id,
        work_id=work_id,
        basis_revision=basis_revision,
        agent_state=_agent_state_summary(ctx),
        source=source_context,
        recent_conversation=ctx.stores.outbox.recent_turns(),
        retrieved_topics=[
            {"id": t.id, "summary": t.summary, "activation": t.activation}
            for t in ctx.stores.memory.all_topics(limit=5)
        ],
        retrieved_memories=[
            {"id": m.id, "text": m.text, "activation": m.activation}
            for m in ctx.stores.memory.recent_memories(limit=5)
        ],
    )
    ctx.stores.work.insert_work(
        WorkItem(
            work_id=work_id,
            kind=kind,
            cycle_id=cycle_id,
            basis_revision=basis_revision,
            source_event_id=source_event_id,
            status=WorkStatus.PENDING,
            created_at=now,
            snapshot=snapshot.as_dict(),
        )
    )
    return work_id


def create_embedding_work(
    ctx: ReducerContext, *, memory_id: str, source_event_id: str, cycle_id: str, now: datetime
) -> str:
    """Persist one embedding work item for a provisional memory; return its ``work_id``."""
    work_id = ids.new_id(ids.WORK_ITEM)
    memory = ctx.stores.memory.get_memory(memory_id)
    ctx.stores.work.insert_work(
        WorkItem(
            work_id=work_id,
            kind=WorkKind.EMBEDDING,
            cycle_id=cycle_id,
            basis_revision=ctx.stores.state.state_revision(),
            source_event_id=source_event_id,
            status=WorkStatus.PENDING,
            created_at=now,
            snapshot={"provisional_memory_id": memory_id, "text": memory.text if memory else ""},
        )
    )
    return work_id
