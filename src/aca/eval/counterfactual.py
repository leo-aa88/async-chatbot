"""Counterfactual sensitivity: does a memory actually change a later choice?

The project's sharpest constraint is that continuity must have *causal consequences* — a memory
should alter what the agent notices, chooses, or avoids later, or it is theater, not continuity.
ACA is deterministic (seeded ``Rng`` + explicit ``Clock``), so replaying the *same* scenario with a
memory ablated isolates that memory's causal effect on the agent's proactive choices.

This module is pure: it extracts the proactive decision stream from cognition traces and diffs two
runs (baseline vs ablated). Running the scenarios (building a reducer, driving events, ablating a
memory) is the caller's job — the deterministic test ``Harness`` does exactly that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProactiveDecision:
    """What a self-initiated (StochasticWake) cycle chose: the candidate and the outcome."""

    cycle_id: str
    candidate_id: str | None
    action: str | None


@dataclass(frozen=True)
class CounterfactualResult:
    baseline: tuple[ProactiveDecision, ...]
    ablated: tuple[ProactiveDecision, ...]

    @staticmethod
    def _stream(ds: tuple[ProactiveDecision, ...]) -> list[tuple[str | None, str | None]]:
        return [(d.candidate_id, d.action) for d in ds]

    @property
    def changed(self) -> bool:
        """Whether ablating the memory changed the proactive choices at all."""
        return self._stream(self.baseline) != self._stream(self.ablated)

    @property
    def changed_cycles(self) -> int:
        """How many positional decisions differ between the two runs (length diff counts too)."""
        b, a = self._stream(self.baseline), self._stream(self.ablated)
        diff = sum(1 for x, y in zip(b, a, strict=False) if x != y)
        return diff + abs(len(b) - len(a))


def proactive_decisions(traces: Iterable) -> tuple[ProactiveDecision, ...]:
    """Extract the self-initiated decision stream from cognition traces (oldest first expected).

    ``traces`` are ``CognitionTrace``-like objects with ``trigger``/``cycle_id``/``candidate_id``/
    ``action``. Only ``StochasticWake`` cycles are proactive choices.
    """
    return tuple(
        ProactiveDecision(t.cycle_id, t.candidate_id, t.action)
        for t in traces
        if t.trigger == "StochasticWake"
    )


def compare(baseline_traces: Iterable, ablated_traces: Iterable) -> CounterfactualResult:
    """Diff two runs' proactive decision streams into a CounterfactualResult."""
    return CounterfactualResult(
        proactive_decisions(baseline_traces), proactive_decisions(ablated_traces)
    )
