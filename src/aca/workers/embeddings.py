"""Real embedding worker over an OpenAI-style ``/embeddings`` endpoint (DESIGN 12.2, 28.4).

The default ``FakeEmbeddingWorker`` hashes tokens into buckets — deterministic, but not semantic, so
it can't tell "consciousness" relates to "Goodhart". A real provider gives provisional memories
genuine semantic vectors, which is the foundation for semantic retrieval, topic-dominance metrics,
and semantic de-duplication. Provider-agnostic via an OpenAI-compatible endpoint; the vector the
provider returns is data the reducer stores, never authority. ``httpx`` is imported lazily so the
package still imports with only the fake worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import WorkerError
from .base import EmbeddingOutput


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingAdapter:
    """OpenAI-style ``POST {base_url}/embeddings`` (OpenAI, and compatible endpoints)."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float

    async def embed(self, text: str) -> EmbeddingOutput:
        import httpx

        payload = {"model": self.model, "input": text}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/embeddings", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                # Surface the provider's error body, not a bare status (same as the LLM adapters).
                raise WorkerError(f"embeddings {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
        items = data.get("data") or []
        if not items or "embedding" not in items[0]:
            raise WorkerError(f"embeddings response missing data: {str(data)[:200]}")
        vector = [float(x) for x in items[0]["embedding"]]
        model_version = str(data.get("model") or self.model)
        return EmbeddingOutput(vector=vector, model_version=model_version)


class ProviderEmbeddingWorker:
    """Adapts an embedding adapter to the ``EmbeddingWorker`` protocol."""

    def __init__(self, adapter: OpenAIEmbeddingAdapter) -> None:
        self._adapter = adapter
        self.model_version = adapter.model

    async def embed(self, text: str) -> EmbeddingOutput:
        return await self._adapter.embed(text)
