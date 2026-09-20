"""Deterministic vector helpers for semantic metrics (DESIGN 12.2).

Pure functions over embedding vectors — cosine similarity and a cheap single-link clustering used to
measure how much of a set of utterances collapses into one semantic neighborhood. No model calls;
given the same vectors in the same order, the result is identical (replay-safe). Callers must only
pass vectors from the *same* embedding model — cosine across models is meaningless (drift).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1, 1]; 0 if either vector is zero or lengths differ."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dominant_cluster_fraction(vectors: list[Vector], threshold: float) -> tuple[int, int]:
    """Return ``(largest_cluster_size, total)`` under greedy single-link clustering.

    Each vector joins the first existing cluster whose representative (its first member) it is at
    least ``threshold`` similar to, else it starts a new cluster. Iteration order is the caller's,
    so pass a stable order (e.g. sorted by id) for determinism. ``total`` is ``len(vectors)``; the
    fraction ``largest / total`` is how concentrated the set is around one theme.
    """
    if not vectors:
        return (0, 0)
    reps: list[Vector] = []
    sizes: list[int] = []
    for vec in vectors:
        for i, rep in enumerate(reps):
            if cosine(vec, rep) >= threshold:
                sizes[i] += 1
                break
        else:
            reps.append(vec)
            sizes.append(1)
    return (max(sizes), len(vectors))
