"""Startup recovery and reconciliation (DESIGN 29.1, 23.5, invariants 24, 25, 30-32).

Runs once before normal event processing. It classifies the previous shutdown, materializes
wall-time effects (proactive TTL expiry, reclaimed work leases), recovers durable work and
acknowledged-but-unreduced events, and reconciles mandatory obligations. Crucially it does
**not** count downtime as observed user silence and does **not** replay missed stochastic
wakes — the service samples exactly one fresh wake afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .. import ids
from ..clock import Clock
from ..domain.enums import ExitKind, LifecycleState, OutboundKind, OutboundStatus
from ..domain.events import Event
from ..domain.runtime import AgentIdentity, RuntimeSession
from ..domain.state import ConversationState, SelfModel
from ..config import Config
from ..persistence.stores import Stores


@dataclass(slots=True)
class RecoveryPlan:
    session_id: str
    scheduler_generation: int
    exit_kind: ExitKind
    downtime_seconds: float
    is_new_agent: bool
    dispatch_work_ids: list[str] = field(default_factory=list)
    replay_events: list[Event] = field(default_factory=list)


def bootstrap_identity(stores: Stores, clock: Clock, config: Config) -> tuple[AgentIdentity, bool]:
    """Load the durable identity or mint it exactly once (invariant 29)."""
    identity = stores.identity.load_identity()
    if identity is not None:
        return identity, False
    now = clock.now_utc()
    identity = AgentIdentity(
        agent_id=ids.new_id(ids.AGENT),
        created_at=now,
        lifecycle_state=LifecycleState.RECOVERING,
    )
    with stores.db.transaction():
        stores.identity.insert_identity(identity)
        stores.state.initialize(
            SelfModel(
                initiative=config.temperament.initiative,
                inhibition=config.temperament.inhibition,
                persistence=config.temperament.persistence,
            ),
            ConversationState(),
        )
    return identity, True


def _classify_downtime(
    identity: AgentIdentity, previous: RuntimeSession | None, now: datetime
) -> tuple[ExitKind, float]:
    """Classify the previous unavailable interval (DESIGN 29.1 step 5)."""
    if previous is None:
        return ExitKind.UNKNOWN, 0.0
    if previous.closed_at is not None and previous.exit_kind is ExitKind.CLEAN_SUSPEND:
        reference = identity.last_clean_suspend_at or previous.closed_at
        return ExitKind.CLEAN_SUSPEND, max(0.0, (now - reference).total_seconds())
    # No clean-close marker: unclean interruption; lower-bound downtime from last heartbeat.
    reference = identity.last_heartbeat_at or previous.started_at
    return ExitKind.UNCLEAN_INTERRUPTION, max(0.0, (now - reference).total_seconds())


def recover(stores: Stores, clock: Clock, config: Config) -> RecoveryPlan:
    """Perform reconciliation and return a plan for the service to execute."""
    now = clock.now_utc()
    identity, is_new = bootstrap_identity(stores, clock, config)
    previous_session = stores.identity.latest_session()
    exit_kind, downtime = (ExitKind.UNKNOWN, 0.0) if is_new else _classify_downtime(
        identity, previous_session, now
    )

    generation = stores.state.scheduler_generation() + 1
    session = RuntimeSession(
        id=ids.new_id(ids.RUNTIME_SESSION),
        agent_id=identity.agent_id,
        started_at=now,
        scheduler_generation=generation,
    )

    with stores.db.transaction():
        stores.identity.insert_session(session)
        stores.identity.update_identity(
            _entering_recovery(identity, session.id, now)
        )
        stores.state.set_scheduler_generation(generation)
        _reclaim_leases(stores, now)
        _expire_stale_proactive(stores, now)

    dispatch_ids = [w.work_id for w in stores.work.pending()]
    replay_events = stores.events.unreduced()

    return RecoveryPlan(
        session_id=session.id,
        scheduler_generation=generation,
        exit_kind=exit_kind,
        downtime_seconds=downtime,
        is_new_agent=is_new,
        dispatch_work_ids=dispatch_ids,
        replay_events=replay_events,
    )


def _entering_recovery(identity: AgentIdentity, session_id: str, now: datetime) -> AgentIdentity:
    from dataclasses import replace

    return replace(
        identity,
        lifecycle_state=LifecycleState.RECOVERING,
        current_runtime_session_id=session_id,
        last_started_at=now,
        last_heartbeat_at=now,
    )


def _reclaim_leases(stores: Stores, now: datetime) -> None:
    """Requeue work whose RUNNING lease expired during downtime (DESIGN 23.5)."""
    for work in stores.work.expired_running(now):
        stores.work.requeue(work.work_id, error="lease_reclaimed_on_recovery")


def _expire_stale_proactive(stores: Stores, now: datetime) -> None:
    """Expire proactive outbox items past TTL; never dump a stale backlog (DESIGN 23.7)."""
    for message in stores.outbox.pending_by_kind(OutboundKind.PROACTIVE):
        if message.expires_at is not None and message.expires_at <= now:
            stores.outbox.set_status(message.message_id, OutboundStatus.EXPIRED, error="expired:recovery")
