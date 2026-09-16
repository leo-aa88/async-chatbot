"""Seedable random source for the stochastic core (DESIGN 10, 26.1).

A captured seed reproduces every local stochastic decision (wake sampling, candidate
selection). Cognition never touches the global ``random`` module directly; it receives an
``Rng`` so replay and tests are deterministic. Only the *local* core is reproducible — live
LLM/embedding results must additionally be recorded (invariant 28).
"""

from __future__ import annotations

import math
import random
from typing import Sequence


class Rng:
    """A thin, deterministic wrapper over ``random.Random`` with the primitives we need."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int | None:
        return self._seed

    def uniform(self) -> float:
        """Uniform draw in the half-open interval [0, 1)."""
        return self._random.random()

    def exponential(self, rate_per_hour: float) -> float:
        """Sample an exponential inter-arrival time in **hours** for the given hazard rate.

        Δt ~ Exp(λ). A non-positive rate means "effectively never" and returns infinity so the
        scheduler treats it as no scheduled wake rather than crashing.
        """
        if rate_per_hour <= 0.0:
            return math.inf
        # Inverse-CDF sampling keeps the draw a pure function of one uniform, which is easy to
        # reason about during replay.
        u = self.uniform()
        # Guard the log domain against u == 0.0.
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        return -math.log(1.0 - u) / rate_per_hour

    def choice_index(self, weights: Sequence[float]) -> int:
        """Return an index sampled proportionally to non-negative ``weights``.

        Used for softmax candidate selection. Assumes weights are already exponentiated and
        non-negative; a degenerate all-zero vector falls back to a uniform choice.
        """
        total = math.fsum(weights)
        if total <= 0.0:
            return self._random.randrange(len(weights))
        target = self.uniform() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                return index
        return len(weights) - 1  # floating-point safety net

    def seed_fragment(self) -> str:
        """A short, stable fragment of the seed for trace records."""
        return "none" if self._seed is None else f"{self._seed & 0xFFFF:04x}"
