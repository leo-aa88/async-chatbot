"""Deterministic compact snapshot construction for the generative LLM boundary (DESIGN 21).

The worker consumes an *immutable* snapshot. For test/replay determinism the snapshot must be
canonical: stable field ordering, deterministic retrieval ordering, canonical serialization,
explicit timestamps from state (no implicit wall reads), and a stable prompt-template version.
A ``prompt_hash`` over the canonical form detects accidental prompt drift (DESIGN 21).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PROMPT_TEMPLATE_VERSION = "v0.6"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """An immutable, canonical context bundle plus its prompt hash."""

    cycle_id: str
    work_id: str
    basis_revision: int
    template_version: str
    context: dict[str, Any] = field(default_factory=dict)

    def canonical_json(self) -> str:
        """Deterministic serialization (sorted keys, no whitespace drift)."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "work_id": self.work_id,
            "basis_revision": self.basis_revision,
            "template_version": self.template_version,
            "context": self.context,
        }

    def prompt_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


def build_snapshot(
    *,
    cycle_id: str,
    work_id: str,
    basis_revision: int,
    agent_state: dict[str, Any],
    source: dict[str, Any] | None,
    recent_conversation: list[dict[str, Any]],
    retrieved_topics: list[dict[str, Any]],
    retrieved_memories: list[dict[str, Any]],
    deferred_intent: dict[str, Any] | None = None,
) -> Snapshot:
    """Assemble a canonical snapshot (DESIGN 21 suggested context bundle).

    Retrieval lists are sorted by a stable key so ordering is deterministic and independent of
    dict/query iteration order. Complete conversation history is deliberately excluded.
    """
    context = {
        "agent_state": agent_state,
        "source": source,
        "recent_conversation": recent_conversation,  # caller supplies in chronological order
        "retrieved_topics": sorted(retrieved_topics, key=_stable_key),
        "retrieved_memories": sorted(retrieved_memories, key=_stable_key),
        "deferred_intent": deferred_intent,
    }
    return Snapshot(
        cycle_id=cycle_id,
        work_id=work_id,
        basis_revision=basis_revision,
        template_version=PROMPT_TEMPLATE_VERSION,
        context=context,
    )


def _stable_key(item: dict[str, Any]) -> tuple[Any, ...]:
    # Deterministic tie-broken ordering: highest score first, then id ascending.
    return (-float(item.get("score", item.get("activation", 0.0))), str(item.get("id", "")))
