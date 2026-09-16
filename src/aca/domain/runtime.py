"""Runtime bookkeeping records: identity, sessions, work items, obligations, outbound, traces.

Pure data. The stores map these to/from SQLite rows and the reducer is the only writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import (
    ActionKind,
    ExitKind,
    LifecycleState,
    ObligationStatus,
    OutboundKind,
    OutboundStatus,
    WorkKind,
    WorkStatus,
)


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """Durable logical-agent identity (DESIGN 7.5, 23.6). Never re-minted on restart."""

    agent_id: str
    created_at: datetime
    lifecycle_state: LifecycleState
    current_runtime_session_id: str | None = None
    last_started_at: datetime | None = None
    last_active_at: datetime | None = None
    last_clean_suspend_at: datetime | None = None
    last_resume_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    total_active_seconds: int = 0
    last_runtime_exit_kind: ExitKind | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSession:
    """One contiguous execution of the agent service (DESIGN 4.11, 23.6)."""

    id: str
    agent_id: str
    started_at: datetime
    scheduler_generation: int
    closed_at: datetime | None = None
    exit_kind: ExitKind | None = None


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A durable, leased unit of semantic work (DESIGN 8.2, 23.5)."""

    work_id: str
    kind: WorkKind
    cycle_id: str
    basis_revision: int
    source_event_id: str
    status: WorkStatus
    created_at: datetime
    attempt_count: int = 0
    lease_until: datetime | None = None
    completed_at: datetime | None = None
    result_event_id: str | None = None
    last_error: str | None = None
    # Immutable snapshot the worker consumes; opaque to persistence.
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResponseObligation:
    """An auditable mandatory-response obligation (DESIGN 14.1, 23)."""

    id: str
    source_event_id: str
    created_at: datetime
    status: ObligationStatus = ObligationStatus.PENDING
    work_id: str | None = None
    satisfied_by_message_id: str | None = None
    superseded_by_id: str | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class Action:
    """A durable reducer decision to speak / defer / stay silent (DESIGN 23.7)."""

    id: str
    kind: ActionKind
    created_at: datetime
    cycle_id: str | None = None
    source_event_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    """A durable outbound item; delivered != decided (DESIGN 6.1, 23.7)."""

    message_id: str
    delivery_key: str
    action_id: str
    kind: OutboundKind
    channel: str
    payload: str
    status: OutboundStatus
    created_at: datetime
    expires_at: datetime | None = None
    delivered_at: datetime | None = None
    superseded_by_id: str | None = None
    last_delivery_error: str | None = None


@dataclass(frozen=True, slots=True)
class CognitionTrace:
    """An inspectable record of one cognition cycle (DESIGN 26)."""

    cycle_id: str
    created_at: datetime
    source_event_id: str | None = None
    basis_revision: int | None = None
    commit_revision: int | None = None
    trigger: str | None = None
    conversation_mode: str | None = None
    candidate_kind: str | None = None
    candidate_id: str | None = None
    candidate_was_null: bool = False
    llm_called: bool = False
    action: str | None = None
    useful_enrichment: bool = False
    pre_outbox_invalidated: bool = False
    notes: str | None = None
    rng_seed_fragment: str | None = None
    prompt_hash: str | None = None
