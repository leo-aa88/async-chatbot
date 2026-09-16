"""Worker protocols (DESIGN 21, 22, 28.4).

The generative and embedding boundaries are narrow, async, and return plain data. A real
Anthropic-backed LLM worker or a local embedding model slots in behind these protocols without
touching cognition semantics — the reducer only ever sees result *events*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..cognition.snapshot import Snapshot


@dataclass(frozen=True, slots=True)
class LLMOutput:
    """Raw, still-untrusted worker output plus token accounting."""

    result: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingOutput:
    """A local embedding vector and the model version that produced it."""

    vector: list[float]
    model_version: str


@runtime_checkable
class LLMWorker(Protocol):
    async def run(self, snapshot: Snapshot) -> LLMOutput:
        """Interpret the snapshot and return an untrusted typed-proposal result (DESIGN 22)."""


@runtime_checkable
class EmbeddingWorker(Protocol):
    async def embed(self, text: str) -> EmbeddingOutput:
        """Produce a local embedding for semantic addressability (DESIGN 12.2, 28.4)."""
