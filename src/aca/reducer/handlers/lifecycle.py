"""Lifecycle event handlers (DESIGN 8.4, 29).

Lifecycle events use the same reducer/event semantics as conversation, but they may not
synthesize missed cognition (DESIGN 8.4). In particular, ``AgentResumed`` / interruption
detection record temporal facts and downtime **without** adding downtime to agent-observed user
silence (invariant 31) and **without** replaying missed stochastic wakes (invariant 32).
"""

from __future__ import annotations

from dataclasses import replace

from ...domain.enums import ExitKind, LifecycleState
from ...domain.events import (
    AgentResumed,
    AgentStarted,
    AgentSuspending,
    RuntimeInterruptionDetected,
)
from ..context import ReducerContext
from .base import HandlerOutcome


def handle_agent_started(ctx: ReducerContext, event: AgentStarted) -> HandlerOutcome:
    identity = ctx.stores.identity.load_identity()
    if identity is not None:
        ctx.stores.identity.update_identity(
            replace(
                identity,
                last_started_at=event.timestamp,
                last_active_at=event.timestamp,
                last_heartbeat_at=event.timestamp,
            )
        )
    return HandlerOutcome(note="agent_started")


def handle_agent_suspending(ctx: ReducerContext, event: AgentSuspending) -> HandlerOutcome:
    identity = ctx.stores.identity.load_identity()
    if identity is not None:
        ctx.stores.identity.update_identity(
            replace(
                identity,
                lifecycle_state=LifecycleState.SUSPENDED,
                last_clean_suspend_at=event.timestamp,
                last_active_at=event.timestamp,
                last_runtime_exit_kind=ExitKind(event.exit_kind),
            )
        )
    return HandlerOutcome(note="agent_suspending")


def handle_agent_resumed(ctx: ReducerContext, event: AgentResumed) -> HandlerOutcome:
    identity = ctx.stores.identity.load_identity()
    if identity is not None:
        ctx.stores.identity.update_identity(
            replace(identity, last_resume_at=event.timestamp, last_active_at=event.timestamp)
        )
    # Downtime is NOT added to observed silence and missed wakes are NOT replayed.
    return HandlerOutcome(note=f"agent_resumed:downtime={event.downtime_seconds:.0f}s")


def handle_runtime_interruption(
    ctx: ReducerContext, event: RuntimeInterruptionDetected
) -> HandlerOutcome:
    identity = ctx.stores.identity.load_identity()
    if identity is not None:
        ctx.stores.identity.update_identity(
            replace(
                identity,
                last_resume_at=event.timestamp,
                last_active_at=event.timestamp,
                last_runtime_exit_kind=ExitKind(event.exit_kind),
            )
        )
    return HandlerOutcome(note=f"interruption:downtime={event.downtime_seconds:.0f}s")
