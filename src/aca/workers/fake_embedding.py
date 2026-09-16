"""Deterministic fake embedding worker.

Produces a small, L2-normalized vector by hashing token n-grams into fixed buckets. It is fully
deterministic (same text → same vector), needs no model download, and is stable/versioned so
indexes are reproducible (DESIGN 28.4). Good enough to exercise semantic-addressability code
paths in tests; a real local model replaces it behind ``EmbeddingWorker``.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import EmbeddingOutput

_DIM = 64
_MODEL_VERSION = "fake-hash-embedding-v1"
_WORD = re.compile(r"[a-z0-9']+")


def _bucket(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _DIM


def embed_text(text: str) -> list[float]:
    """Pure embedding function (exposed for tests and reuse)."""
    vector = [0.0] * _DIM
    tokens = _WORD.findall(text.lower())
    for token in tokens:
        vector[_bucket(token)] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class FakeEmbeddingWorker:
    """A zero-dependency deterministic embedding worker."""

    model_version = _MODEL_VERSION

    async def embed(self, text: str) -> EmbeddingOutput:
        return EmbeddingOutput(vector=embed_text(text), model_version=_MODEL_VERSION)
