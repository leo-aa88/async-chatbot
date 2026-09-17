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
    """An embedding vector and the model version that produced it.

    Local by default (``FakeEmbeddingWorker``, DESIGN 28.4); a real provider may be opted into via
    ``config.embedding`` (``ProviderEmbeddingWorker``), which is a network call, not local.
    """

    vector: list[float]
    model_version: str


@runtime_checkable
class LLMWorker(Protocol):
    async def run(self, snapshot: Snapshot) -> LLMOutput:
        """Interpret the snapshot and return an untrusted typed-proposal result (DESIGN 22)."""


@runtime_checkable
class EmbeddingWorker(Protocol):
    async def embed(self, text: str) -> EmbeddingOutput:
        """Produce an embedding for semantic addressability (DESIGN 12.2, 28.4).

        Local and free by default (the fake worker). A configured real provider is an opted-in
        remote call: because embedding runs on nearly every substantive message (DESIGN 12.2), that
        choice sends most user text off-host at higher frequency than generative calls — see
        ``config.Embedding``.
        """
