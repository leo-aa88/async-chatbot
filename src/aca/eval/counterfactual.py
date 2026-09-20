"""Counterfactual sensitivity: does a memory change the agent's observable behavior?

Offline analysis only — nothing here is imported by the runtime, writes state, or is enforced by
the reducer. ACA is deterministic (seeded ``Rng`` + explicit ``Clock``), so replaying the *same*
scenario with a memory ablated isolates that memory's effect on what the agent actually does.

The instrument diffs the **observable proactive outcome** of two runs: for each ``StochasticWake``
cycle, the ``(action, effect)`` pair, where ``effect`` is the committed spoken payload (``None`` for
silence). It deliberately keys on the *effect*, not the internally selected candidate: an agent that
picks a different candidate but says the identical thing has changed nothing a person could observe,
so the diff must not report it as a change. ``candidate_id``/``cycle_id`` are retained on each record
for inspection but excluded from the comparison (``cycle_id`` is a fresh ``ids.new_id`` per run and
would otherwise make every cross-run diff spuriously "changed").

Running the scenarios (building a reducer, driving events, draining work, ablating a memory) is the
caller's job — the deterministic test ``Harness`` does exactly that, and resolves each cycle's
committed payload from the outbox to pass in as ``effects``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ProactiveDecision:
    """One self-initiated (StochasticWake) cycle: its observable outcome plus inspection fields.

    ``action``/``effect`` are the observable outcome (what a person experiences); ``candidate_id``
    and ``cycle_id`` are internal detail kept for debugging but excluded from equality/diffing.
    """

    cycle_id: str
    candidate_id: str | None
    action: str | None
    effect: str | None = None

    @property
    def observable(self) -> tuple[str | None, str | None]:
        return (self.action, self.effect)


@dataclass(frozen=True)
class CounterfactualResult:
    baseline: tuple[ProactiveDecision, ...]
    ablated: tuple[ProactiveDecision, ...]

    @staticmethod
    def _stream(ds: tuple[ProactiveDecision, ...]) -> list[tuple[str | None, str | None]]:
        return [d.observable for d in ds]

    @property
    def changed(self) -> bool:
        """Whether ablating the memory changed the observable proactive behavior at all."""
        return self._stream(self.baseline) != self._stream(self.ablated)

    @property
    def changed_cycles(self) -> int:
        """How many positional outcomes differ between the two runs (length diff counts too)."""
        b, a = self._stream(self.baseline), self._stream(self.ablated)
        diff = sum(1 for x, y in zip(b, a, strict=False) if x != y)
        return diff + abs(len(b) - len(a))


def proactive_decisions(
    traces: Iterable, *, effects: Mapping[str, str] | None = None
) -> tuple[ProactiveDecision, ...]:
    """Extract the self-initiated decision stream from cognition traces (oldest first expected).

    ``traces`` are ``CognitionTrace``-like objects with ``trigger``/``cycle_id``/``candidate_id``/
    ``action``. Only ``StochasticWake`` cycles are proactive choices. ``effects`` optionally maps a
    candidate id to the committed spoken payload for that candidate; when supplied, it fills each
    decision's observable ``effect`` (absent -> ``None``, i.e. silence or nothing delivered).
    """
    return tuple(
        ProactiveDecision(
            t.cycle_id,
            t.candidate_id,
            t.action,
            effect=(effects.get(t.candidate_id) if effects and t.candidate_id else None),
        )
        for t in traces
        if t.trigger == "StochasticWake"
    )


def compare(
    baseline_traces: Iterable,
    ablated_traces: Iterable,
    *,
    baseline_effects: Mapping[str, str] | None = None,
    ablated_effects: Mapping[str, str] | None = None,
) -> CounterfactualResult:
    """Diff two runs' observable proactive outcome streams into a CounterfactualResult."""
    return CounterfactualResult(
        proactive_decisions(baseline_traces, effects=baseline_effects),
        proactive_decisions(ablated_traces, effects=ablated_effects),
    )
