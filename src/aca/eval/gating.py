"""Counterfactual gating eval: is each proactive gate suppressing the *right* things? (DESIGN §27).

Offline analysis only — nothing here is imported by the runtime, writes state, or is enforced by the
reducer. It builds on the counterfactual instrument (`counterfactual.py`): ACA is deterministic
(seeded ``Rng`` + explicit ``Clock``), so replaying the *same* proactive scenario with one gate
**ablated** (disabled via config — e.g. ``memory.discourse_bridge_cosine <= 0`` turns the §34
discourse gate off) isolates what that gate, and only that gate, withheld.

The value question — *was a suppression right?* — cannot be answered by the gate itself (that is
circular) nor by the classifier's own labels (§27.2: "a classifier cannot detect its own false
negatives"). So it is supplied by an injected **oracle**: ``worth(candidate_id) -> bool`` ("is a
proactive utterance about this candidate worth saying?"). A deterministic test passes ground truth;
a human relabeling (§27.2) or a stronger-model judge supplies real labels later. Keeping the oracle
external is the whole point — the harness measures each gate's *impact* mechanically and leaves the
*worth* judgement to a source independent of the thing being judged.

Given a baseline stream and, per gate, the stream with that gate ablated, the eval reports:

* **suppression** — a wake silent in baseline but spoken once the gate was ablated: what the gate
  withheld. (The §34/§16.2 gates are post-selection *mutes*, so the candidate at that position is
  unchanged between the two runs — the ablated run speaks the *same* candidate the baseline muted.)
* **false negative** — a suppression the oracle deems worth saying: the gate silenced value.
* **suppression precision** — the fraction of a gate's suppressions that were right (withheld
  low-value speech). Low precision ⇒ the gate over-suppresses.
* **over-speech** (gate-agnostic) — an utterance spoken in the *baseline* that the oracle rejects:
  low-value speech no gate caught. The false-positive side of the whole gating system.

Position alignment: baseline and ablated must be the same deterministic wake sequence (same seed,
clock, and events, differing only in the ablated gate). After the first divergence, downstream
positions can drift — an extra early speak changes later cooldown/continuity/refractory state — so
the counts are a *first-order* attribution; callers isolate a gate cleanly with short scenarios
(often a single wake), exactly as the sibling counterfactual-sensitivity tests do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .counterfactual import ProactiveDecision

# "Is a proactive utterance about this candidate worth saying?" — the injected value oracle, keyed on
# the stable candidate id (never the fresh per-run cycle id). External by design (see module docs).
Worth = Callable[[str | None], bool]


def _spoke(d: ProactiveDecision) -> bool:
    return d.action == "speak"


@dataclass(frozen=True)
class GateSuppression:
    """A wake the gate changed: silent in baseline, spoken once the gate was ablated.

    ``would_say`` is the payload the ablated run committed — the utterance the gate withheld (``None``
    when the ablated run spoke but delivered no resolvable payload). ``candidate_id`` is the stable
    identity the oracle is keyed on; ``cycle_id`` is the ablated run's fresh id, for inspection only.
    """

    cycle_id: str
    candidate_id: str | None
    would_say: str | None


@dataclass(frozen=True)
class GateEval:
    """One gate's suppressions in a scenario, split by the oracle into right vs wrong."""

    gate: str
    suppressions: tuple[GateSuppression, ...]
    false_negatives: tuple[GateSuppression, ...]  # oracle: worth saying ⇒ the gate silenced value

    @property
    def suppressed(self) -> int:
        return len(self.suppressions)

    @property
    def false_negative_count(self) -> int:
        return len(self.false_negatives)

    @property
    def correct_suppressions(self) -> int:
        return self.suppressed - self.false_negative_count

    @property
    def suppression_precision(self) -> float | None:
        """Fraction of this gate's suppressions that were right (withheld low-value speech). ``None``
        when the gate suppressed nothing in the scenario — no evidence either way, not a perfect
        score, so callers must not read a bare ``0``/``1`` where there was no signal."""
        if self.suppressed == 0:
            return None
        return self.correct_suppressions / self.suppressed


@dataclass(frozen=True)
class GatingReport:
    """Per-gate evals plus the gate-agnostic over-speech for one scenario."""

    gates: tuple[GateEval, ...]
    over_speech: tuple[ProactiveDecision, ...]

    def summary(self) -> dict:
        """A compact, JSON-friendly rollup — the per-gate suppression/false-negative/precision plus
        the overall over-speech count."""
        return {
            "gates": {
                ge.gate: {
                    "suppressed": ge.suppressed,
                    "false_negatives": ge.false_negative_count,
                    "suppression_precision": ge.suppression_precision,
                }
                for ge in self.gates
            },
            "over_speech": len(self.over_speech),
        }


def gate_suppressions(
    baseline: Iterable[ProactiveDecision], ablated: Iterable[ProactiveDecision]
) -> tuple[GateSuppression, ...]:
    """Wakes the ablated gate suppressed: positions silent in baseline but spoken with the gate off.

    Position-aligned (see module docs on first-order attribution). A length difference between the
    streams is tolerated (``zip`` stops at the shorter) — callers keep the sequences aligned by
    driving the same wake events.
    """
    return tuple(
        GateSuppression(a.cycle_id, a.candidate_id, a.effect)
        for b, a in zip(baseline, ablated, strict=False)
        if not _spoke(b) and _spoke(a)
    )


def evaluate_gate(
    gate: str,
    baseline: Iterable[ProactiveDecision],
    ablated: Iterable[ProactiveDecision],
    *,
    worth: Worth,
) -> GateEval:
    """Classify one gate's suppressions with the injected oracle into correct vs false-negative."""
    suppressions = gate_suppressions(baseline, ablated)
    false_negatives = tuple(s for s in suppressions if worth(s.candidate_id))
    return GateEval(gate, suppressions, false_negatives)


def over_speech(
    baseline: Iterable[ProactiveDecision], *, worth: Worth
) -> tuple[ProactiveDecision, ...]:
    """Utterances spoken in the baseline that the oracle rejects — low-value speech no gate caught.

    The false-positive side of the gating system as a whole; gate-agnostic (it asks what got through
    *everything*, not which gate should have stopped it).
    """
    return tuple(d for d in baseline if _spoke(d) and not worth(d.candidate_id))


def evaluate(
    baseline: Iterable[ProactiveDecision],
    ablated_by_gate: Mapping[str, Iterable[ProactiveDecision]],
    *,
    worth: Worth,
) -> GatingReport:
    """Full gating report: each gate's suppressions classified, plus the overall over-speech.

    ``baseline`` is materialized once and reused across gates (streams may be one-shot iterables).
    """
    base = tuple(baseline)
    gates = tuple(
        evaluate_gate(gate, base, ablated, worth=worth)
        for gate, ablated in ablated_by_gate.items()
    )
    return GatingReport(gates, over_speech(base, worth=worth))
