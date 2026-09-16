"""Stochastic candidate selection with a NULL candidate (DESIGN 12.7).

The candidate pool is ``enriched topics + deferred intents + provisional memories + NOTHING``.
The agent must neither always pick the highest-activation item nor be forced to pick anything.
A softmax over scores ``z_i`` at temperature ``T`` includes an explicit NOTHING score:

    P(T_i) = e^{z_i/T} / (e^{z_∅/T} + Σ_j e^{z_j/T})

Selection is a pure function of the candidate scores and the injected ``Rng``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.enums import CandidateKind
from ..rng import Rng


@dataclass(frozen=True, slots=True)
class Candidate:
    """A scored selection candidate. ``score`` is the pre-temperature logit ``z_i``."""

    kind: CandidateKind
    id: str
    score: float


@dataclass(frozen=True, slots=True)
class Selection:
    """The selected candidate (or a NOTHING sentinel) plus the probability it was chosen with."""

    candidate: Candidate
    probability: float

    @property
    def is_nothing(self) -> bool:
        return self.candidate.kind is CandidateKind.NOTHING


NOTHING = Candidate(kind=CandidateKind.NOTHING, id="", score=0.0)


def _softmax_weights(scores: list[float], temperature: float) -> list[float]:
    # Subtract the max for numerical stability before exponentiating.
    hottest = max(scores)
    return [math.exp((s - hottest) / temperature) for s in scores]


def select(
    candidates: list[Candidate],
    null_score: float,
    temperature: float,
    rng: Rng,
) -> Selection:
    """Sample one candidate (or NOTHING) proportionally to the tempered softmax.

    ``temperature`` must be > 0. The NOTHING candidate always participates, so the agent can
    decline to select anything even when real candidates exist.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    pool = [*candidates, Candidate(kind=CandidateKind.NOTHING, id="", score=null_score)]
    scores = [c.score for c in pool]
    weights = _softmax_weights(scores, temperature)
    total = math.fsum(weights)
    index = rng.choice_index(weights)
    probability = weights[index] / total if total > 0 else 0.0
    return Selection(candidate=pool[index], probability=probability)
