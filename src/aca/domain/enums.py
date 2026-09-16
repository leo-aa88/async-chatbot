"""Enumerations for lifecycle, conversation, classification, and durable state machines.

Values are stored as strings in SQLite, so the ``.value`` is the persisted form. Never
renumber or rename an existing value without a migration.
"""

from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    """Logical lifecycle states (DESIGN 7.5)."""

    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    RECOVERING = "RECOVERING"


class ExitKind(str, Enum):
    """How a previous runtime session ended (DESIGN 7.5, 29.6)."""

    CLEAN_SUSPEND = "CLEAN_SUSPEND"
    HOST_RESTART = "HOST_RESTART"
    UNCLEAN_INTERRUPTION = "UNCLEAN_INTERRUPTION"
    UNKNOWN = "UNKNOWN"


class ConversationMode(str, Enum):
    """Liveness of the conversation, distinct from lifecycle (DESIGN 7.4)."""

    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    DORMANT = "DORMANT"


class MessageClass(str, Enum):
    """Cheap human-message classes (DESIGN 20)."""

    DIRECT_TASK = "DIRECT_TASK"
    TASK_QUESTION = "TASK_QUESTION"
    SOCIAL_QUESTION = "SOCIAL_QUESTION"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    CONVERSATION_CLOSER = "CONVERSATION_CLOSER"
    STATEMENT = "STATEMENT"
    HIGH_INFORMATION = "HIGH_INFORMATION"
    LOW_INFORMATION = "LOW_INFORMATION"
    REPROMPT = "REPROMPT"

    @property
    def response_required(self) -> bool:
        """Whether this class deterministically demands a response (DESIGN 13.1, 20)."""
        return self in _RESPONSE_REQUIRED_CLASSES


_RESPONSE_REQUIRED_CLASSES = frozenset(
    {MessageClass.DIRECT_TASK, MessageClass.TASK_QUESTION, MessageClass.REPROMPT}
)


class EnrichmentStatus(str, Enum):
    """Provisional-memory enrichment lifecycle (DESIGN 12.8)."""

    RAW = "RAW"
    PENDING_ENRICHMENT = "PENDING_ENRICHMENT"
    ENRICHED = "ENRICHED"
    FAILED = "FAILED"
    DO_NOT_ENRICH = "DO_NOT_ENRICH"


class WorkStatus(str, Enum):
    """Durable work-item state machine (DESIGN 23.5)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED_TERMINAL = "FAILED_TERMINAL"


class WorkKind(str, Enum):
    """Kinds of asynchronous semantic work."""

    LLM_COGNITION = "LLM_COGNITION"
    LLM_ENRICHMENT = "LLM_ENRICHMENT"
    EMBEDDING = "EMBEDDING"


class ObligationStatus(str, Enum):
    """Mandatory response-obligation lifecycle (DESIGN 14.1, 23)."""

    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


class OutboundKind(str, Enum):
    """Semantic class of an outbound item (DESIGN 6.1)."""

    MANDATORY = "MANDATORY"
    PROACTIVE = "PROACTIVE"


class OutboundStatus(str, Enum):
    """Outbound delivery lifecycle (DESIGN 23.7)."""

    PENDING_DELIVERY = "PENDING_DELIVERY"
    DELIVERING = "DELIVERING"
    DELIVERED = "DELIVERED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class ActionKind(str, Enum):
    """A reducer decision recorded durably (DESIGN 22)."""

    SPEAK = "SPEAK"
    DEFER = "DEFER"
    SILENCE = "SILENCE"
    ACKNOWLEDGE = "ACKNOWLEDGE"


class CandidateKind(str, Enum):
    """What kind of thing won stochastic selection (DESIGN 12.7)."""

    TOPIC = "TOPIC"
    DEFERRED_INTENT = "DEFERRED_INTENT"
    PROVISIONAL_MEMORY = "PROVISIONAL_MEMORY"
    NOTHING = "NOTHING"
