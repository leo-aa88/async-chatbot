"""Durable semantic-work creation (DESIGN 8.2, 23.5).

Work is persisted PENDING with an immutable snapshot *before* dispatch, so ownership survives a
crash and can be reclaimed by lease (invariant 24). ``create_llm_work`` enforces at most one
generative work item per cycle (invariant 3) with a runtime guard, so the "one generative call
per cycle" rule is checked, not merely commented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .. import ids
from ..cognition.snapshot import build_snapshot
from ..domain.enums import WorkKind, WorkStatus
from ..domain.runtime import WorkItem
from ..errors import AcaError
from .context import ReducerContext


def _agent_state_summary(ctx: ReducerContext) -> dict[str, Any]:
    model = ctx.stores.state.load_self_model()
    conversation = ctx.stores.state.load_conversation()
    summary = {
        "initiative": model.initiative,
        "inhibition": model.inhibition,
        "persistence": model.persistence,
        "mode": conversation.mode.value,
        "dominant_topic": model.dominant_topic,
    }
    if ctx.config.identity.name:
        summary["name"] = ctx.config.identity.name  # durable self-name -> prompt (survives resets)
    return summary


def create_llm_work(
    ctx: ReducerContext,
    *,
    cycle_id: str,
    source_event_id: str,
    source_context: dict[str, Any],
    kind: WorkKind = WorkKind.LLM_COGNITION,
    now: datetime,
) -> str:
    """Persist one generative work item with a canonical snapshot; return its ``work_id``.

    Guards invariant 3: a cognition cycle may request at most one generative LLM call. A second
    generative work item for the same ``cycle_id`` is a programming error and is rejected rather
    than silently doubling LLM spend.
    """
    if ctx.stores.work.generative_exists_for_cycle(cycle_id):
        raise AcaError(f"cycle {cycle_id} already has a generative work item (invariant 3)")
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
