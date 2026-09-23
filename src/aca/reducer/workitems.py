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
from ..cognition.relevance import Retrievable, select_context
from ..cognition.snapshot import build_snapshot
from ..domain.cycles import CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL
from ..domain.enums import WorkKind, WorkStatus
from ..domain.runtime import WorkItem
from ..errors import AcaError
from ..persona import DEFAULT_PERSONA
from .context import ReducerContext

_CONTEXT_BUDGET = 5      # topics and memories each, per snapshot (unchanged)
_RELEVANT_SLOTS = 2      # of that budget, at most this many go to query-relevant older items
_RETRIEVAL_POOL = 200    # bounded, recency-ordered scan the relevance ranker searches
_REPLY_CYCLES = (CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL)


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
    if ctx.config.identity.persona != DEFAULT_PERSONA:
        # Voice selection -> prompt (wording only; no gate reads it). Omitted for the default persona
        # so default snapshots, and their prompt hashes, are unchanged.
        summary["persona"] = ctx.config.identity.persona
    return summary


def _retrieved_context(ctx: ReducerContext, source: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The bundle's retrieved topics and memories (DESIGN 21, 37).

    Recency fills the fixed budget as before; on a reply cycle up to ``_RELEVANT_SLOTS`` of it go to
    older items that share distinctive words with the human's turn, so a durable fact they ask about
    is visible even when newer, unrelated items would crowd it out. Proactive cycles have no query and
    keep the pure-recency bundle. Relevance-selected entries are marked ``"retrieved_for": "query"``.
    """
    query = str(source.get("text") or "") if source.get("cycle_type") in _REPLY_CYCLES else ""
    exclude = frozenset(filter(None, [source.get("turn_memory_id")]))
    memory = ctx.stores.memory
    topics = memory.all_topics(limit=_RETRIEVAL_POOL)
    memories = memory.recent_memories(limit=_RETRIEVAL_POOL)
    topic_items = [Retrievable(t.id, t.summary, t.last_activated_at) for t in topics]
    memory_items = [Retrievable(m.id, m.text, m.last_activated_at) for m in memories]
    picked_topics = select_context(topic_items, topic_items, query, budget=_CONTEXT_BUDGET,
                                   relevant_slots=_RELEVANT_SLOTS, exclude=exclude)
    picked_memories = select_context(memory_items, memory_items, query, budget=_CONTEXT_BUDGET,
                                     relevant_slots=_RELEVANT_SLOTS, exclude=exclude)
    recent_topics = {t.id for t in topic_items[:_CONTEXT_BUDGET]}
    recent_memories = {m.id for m in memory_items[:_CONTEXT_BUDGET]}
    by_topic = {t.id: t for t in topics}
    by_memory = {m.id: m for m in memories}

    def _topic(item: Retrievable) -> dict[str, Any]:
        t = by_topic[item.id]
        entry = {"id": t.id, "summary": t.summary, "activation": t.activation}
        return entry if item.id in recent_topics else {**entry, "retrieved_for": "query"}

    def _memory(item: Retrievable) -> dict[str, Any]:
        m = by_memory[item.id]
        entry = {"id": m.id, "text": m.text, "activation": m.activation}
        return entry if item.id in recent_memories else {**entry, "retrieved_for": "query"}

    return {"retrieved_topics": [_topic(i) for i in picked_topics],
            "retrieved_memories": [_memory(i) for i in picked_memories]}


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
        **_retrieved_context(ctx, source_context),
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


def create_topic_embedding_work(
    ctx: ReducerContext, *, topic_id: str, summary: str, source_event_id: str, now: datetime
) -> str:
    """Persist one embedding work item for a topic *summary*; return its ``work_id``.

    The summary — not the topic's source memory's raw text — is the semantic unit for dedup and
    dominance (DESIGN 12.3). Runs through the same async embedding path as memory embeddings.
    """
    work_id = ids.new_id(ids.WORK_ITEM)
    ctx.stores.work.insert_work(
        WorkItem(
            work_id=work_id,
            kind=WorkKind.EMBEDDING,
            cycle_id=ids.new_id(ids.CYCLE),  # embeddings aren't subject to the one-generative guard
            basis_revision=ctx.stores.state.state_revision(),
            source_event_id=source_event_id,
            status=WorkStatus.PENDING,
            created_at=now,
            snapshot={"topic_id": topic_id, "text": summary},
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
