"""Event model (DESIGN 8).

All inputs, lifecycle transitions, and async work completions enter through one event stream
and are reduced by the single writer. Events are immutable. Each event carries a stable
``event_id`` (the ingress idempotency key for client-originated events) and a UTC timestamp.

Events serialize to/from a ``(type, payload)`` pair so they can live in the durable inbox and
travel over IPC. The registry keeps (de)serialization in one place (DRY).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from ..clock import from_rfc3339, to_rfc3339
from ..errors import ValidationError
from .enums import ActionKind, ExitKind


@dataclass(frozen=True, slots=True)
class Event:
    """Base event. ``TYPE`` is the persisted discriminator; subclasses set it."""

    TYPE: ClassVar[str] = "Event"

    event_id: str
    timestamp: datetime
    source: str = "system"

    def payload(self) -> dict[str, Any]:
        """Type-specific fields (everything except the common envelope)."""
        data = asdict(self)
        for envelope_key in ("event_id", "timestamp", "source"):
            data.pop(envelope_key, None)
        return data


# --- inbound / world events --------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HumanMessage(Event):
    TYPE: ClassVar[str] = "HumanMessage"
    text: str = ""
    channel: str = "cli"


@dataclass(frozen=True, slots=True)
class ExternalEvent(Event):
    TYPE: ClassVar[str] = "ExternalEvent"
    kind: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# --- stochastic / scheduling events ------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StochasticWake(Event):
    TYPE: ClassVar[str] = "StochasticWake"
    runtime_session_id: str = ""
    scheduler_generation: int = 0
    scheduled_at_utc: str = ""


# --- lifecycle events (DESIGN 8.4) -------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AgentStarted(Event):
    TYPE: ClassVar[str] = "AgentStarted"
    runtime_session_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentSuspending(Event):
    TYPE: ClassVar[str] = "AgentSuspending"
    runtime_session_id: str = ""
    exit_kind: str = ExitKind.CLEAN_SUSPEND.value


@dataclass(frozen=True, slots=True)
class AgentResumed(Event):
    TYPE: ClassVar[str] = "AgentResumed"
    runtime_session_id: str = ""
    downtime_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RuntimeInterruptionDetected(Event):
    TYPE: ClassVar[str] = "RuntimeInterruptionDetected"
    runtime_session_id: str = ""
    downtime_seconds: float = 0.0
    exit_kind: str = ExitKind.UNCLEAN_INTERRUPTION.value


# --- async worker result events ----------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EmbeddingResult(Event):
    TYPE: ClassVar[str] = "EmbeddingResult"
    work_id: str = ""
    provisional_memory_id: str = ""
    topic_id: str = ""  # set instead of provisional_memory_id when embedding a topic summary
    embedding_id: str = ""
    model_version: str = ""
    vector: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LLMResult(Event):
    TYPE: ClassVar[str] = "LLMResult"
    work_id: str = ""
    cycle_id: str = ""
    basis_revision: int = 0
    # Raw, still-untrusted result payload. Validated by domain.proposals before any commit.
    result: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True, slots=True)
class DeliveryResult(Event):
    TYPE: ClassVar[str] = "DeliveryResult"
    message_id: str = ""
    delivery_key: str = ""
    delivered: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentActionFeedback(Event):
    TYPE: ClassVar[str] = "AgentActionFeedback"
    target_action_id: str = ""
    classification: str = ""


_REGISTRY: dict[str, type[Event]] = {
    cls.TYPE: cls
    for cls in (
        HumanMessage,
        ExternalEvent,
        StochasticWake,
        AgentStarted,
        AgentSuspending,
        AgentResumed,
        RuntimeInterruptionDetected,
        EmbeddingResult,
        LLMResult,
        DeliveryResult,
        AgentActionFeedback,
    )
}


def event_type_name(event: Event) -> str:
    return type(event).TYPE


def serialize(event: Event) -> dict[str, Any]:
    """Serialize an event to a JSON-safe dict envelope."""
    return {
        "event_id": event.event_id,
        "type": event_type_name(event),
        "timestamp": to_rfc3339(event.timestamp),
        "source": event.source,
        "payload": event.payload(),
    }


def deserialize(envelope: dict[str, Any]) -> Event:
    """Reconstruct an event from a serialized envelope."""
    type_name = envelope.get("type")
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise ValidationError(f"unknown event type: {type_name!r}")
    payload = dict(envelope.get("payload", {}))
    known = ActionKind  # noqa: F841 (kept for clarity of intent; enums validated downstream)
    try:
        return cls(
            event_id=str(envelope["event_id"]),
            timestamp=from_rfc3339(str(envelope["timestamp"])),
            source=str(envelope.get("source", "system")),
            **payload,
        )
    except (KeyError, TypeError) as exc:
        raise ValidationError(f"malformed {type_name} event: {exc}") from exc
