"""Cognitive state entities: provisional memories, topics, deferred intents, self model.

These are pure data records. Stored ``activation`` is the value *as of* ``last_activated_at``;
effective activation at a later time is computed lazily (DESIGN 12.4) by ``cognition.activation``
— never mutated by a background tick. Keeping the "materialized at" timestamp on the record is
what makes lazy decay correct across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .enums import ConversationMode, EnrichmentStatus


@dataclass(frozen=True, slots=True)
class ProvisionalMemory:
    """A cheap, locally-addressable representation of a nontrivial human message (DESIGN 12.2)."""

    id: str
    event_id: str
    text: str
    activation: float
    salience: float
    decay_rate_per_hour: float
    created_at: datetime
    last_activated_at: datetime
    keywords: tuple[str, ...] = ()
    embedding_id: str | None = None
    enrichment_status: EnrichmentStatus = EnrichmentStatus.RAW
    enrichment_attempts: int = 0
    next_enrichment_after: datetime | None = None
    last_enrichment_error: str | None = None


@dataclass(frozen=True, slots=True)
class Topic:
    """A semantically enriched topic (DESIGN 12.3)."""

    id: str
    summary: str
    activation: float
    importance: float
    decay_rate_per_hour: float
    created_at: datetime
    last_activated_at: datetime
    unfinished: bool = False
    tags: tuple[str, ...] = ()
    source: str = "human_message"
    # The provisional memory this topic was enriched from, if any. Lets repeat-suppression treat
    # a topic and its source memory as the same thought so content isn't spoken twice.
    source_memory_id: str | None = None
    # How many enrichments have landed on this topic. Starts at 1; incremented each time a new
    # provisional memory is merged into it instead of creating a near-duplicate topic (DESIGN 12.3).
    evidence_count: int = 1


@dataclass(frozen=True, slots=True)
class DeferredIntent:
    """A persisted latent intention to speak later (DESIGN 16)."""

    id: str
    intent: str
    activation: float
    decay_rate_per_hour: float
    created_at: datetime
    last_activated_at: datetime
    expires_at: datetime
    status: str = "pending"
    topic_id: str | None = None
    provisional_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class SelfModel:
    """The agent's model of its own recent behavior and tendencies (DESIGN 7.3)."""

    initiative: float
    inhibition: float
    persistence: float
    recent_proactive_messages: int = 0
    last_outward_action_at: datetime | None = None
    last_delivered_proactive_at: datetime | None = None
    dominant_topic: str | None = None

    def with_delivered_proactive(self, at: datetime) -> SelfModel:
        """Return a copy reflecting a freshly *delivered* proactive message (DESIGN 11.4)."""
        from dataclasses import replace

        return replace(
            self,
            recent_proactive_messages=self.recent_proactive_messages + 1,
            last_outward_action_at=at,
            last_delivered_proactive_at=at,
        )


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Tracked cadence used to infer conversation mode (DESIGN 7.4)."""

    mode: ConversationMode = ConversationMode.DORMANT
    last_human_message_at: datetime | None = None
    last_agent_delivered_at: datetime | None = None
    # Accumulated silence counted only while RUNNING (DESIGN 7.6). Stored in seconds.
    active_observed_silence_seconds: float = 0.0
    recent_human_turn_timestamps: tuple[datetime, ...] = field(default_factory=tuple)
    # Discourse focus (DESIGN §34, v0.7): id of the provisional memory of the current
    # focus-setting turn — the conversation's *subject*. The discourse gate scores a proactive
    # candidate against this while mode is IDLE. Set/held/cleared by the human-message handler.
    focus_memory_id: str | None = None
    # The subject a §35.3 `CLEAR` most recently *closed* — its provisional-memory id, kept distinct
    # from the current focus (DESIGN §34.11, v0.8). Disambiguates the three meanings of
    # `focus_memory_id is None` (never had one / closed on purpose / lapsed) *with identity*, so while
    # IDLE with no active focus the gate can **renag only the closed thread** — a candidate related to
    # it is muted (don't reopen a resolved subject) while a genuinely unrelated worthwhile thought
    # still speaks: the exact inversion of the open-focus rule. Cleared once a new subject is set or a
    # lull ends; `DORMANT` ignores it (resurfacing preserved). `None` = nothing currently closed.
    closed_focus_memory_id: str | None = None
