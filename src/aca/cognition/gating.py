"""Outward-expression control: hard gates and soft expression factors (DESIGN 11, 14).

Two distinct categories (DESIGN 11):

* **Hard gates** are boolean and short-circuit unsolicited speech (capability, budget,
  cooldown, conversation mode, quiet hours, supersession). They never suppress an explicit
  user task (invariant 11).
* **Soft factors** are *combined* (not ordered) into a probability that a valid thought
  becomes language, via a logistic model (DESIGN 11.3, 14.2).

Refractory recovery affects expression only, never the wake hazard (DESIGN 11.4).
Everything here is pure; quiet-hours evaluation takes an explicit local ``datetime``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time

from ..config import QuietHours
from ..domain.enums import ConversationMode


def sigmoid(x: float) -> float:
    # Guard against overflow for large-magnitude inputs.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _parse_hhmm(value: str) -> time:
    hour, _, minute = value.partition(":")
    return time(int(hour), int(minute or 0))


def quiet_hours_active(now_local: datetime, quiet: QuietHours) -> bool:
    """Whether the configured local-time quiet interval is currently active (DESIGN 11.6).

    Handles intervals that wrap past midnight (e.g. ``01:00``-``08:00`` and ``22:00``-``06:00``).
    """
    if not quiet.enabled:
        return False
    start = _parse_hhmm(quiet.start_local)
    end = _parse_hhmm(quiet.end_local)
    current = now_local.time()
    if start == end:
        return False
    if start < end:
        return start <= current < end
    # Wrapping interval: active if after start OR before end.
    return current >= start or current < end


def refractory_factor(hours_since_delivered_proactive: float | None, tau_hours: float) -> float:
    """Suppression multiplier in [0, 1]; ``f = 1 - e^{-t/τ}`` (DESIGN 11.4).

    ``None`` (no proactive message ever delivered) means fully recovered → 1.0. A larger value
    means *more* recovered (less suppression). Callers subtract ``(1 - factor)`` as a penalty.
    """
    if hours_since_delivered_proactive is None:
        return 1.0
    if tau_hours <= 0.0:
        return 1.0
    t = max(0.0, hours_since_delivered_proactive)
    return 1.0 - math.exp(-t / tau_hours)


def proactive_allowed_in_mode(mode: ConversationMode) -> bool:
    """Whether unrelated proactive initiative is permitted in this mode (DESIGN 7.4, 13.6)."""
    # ACTIVE strongly suppresses unrelated initiative; IDLE allows limited; DORMANT allows.
    return mode is not ConversationMode.ACTIVE


@dataclass(frozen=True, slots=True)
class HardGateResult:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ProactiveHardGateInput:
    capability_allowed: bool
    budget_available: bool
    cooldown_clear: bool
    mode: ConversationMode
    quiet_active: bool
    superseded: bool


def evaluate_proactive_hard_gates(gate: ProactiveHardGateInput) -> HardGateResult:
    """Short-circuit evaluation of proactive hard gates (DESIGN 11.1)."""
    if not gate.capability_allowed:
        return HardGateResult(False, "capability_denied")
    if gate.superseded:
        return HardGateResult(False, "superseded")
    if not gate.budget_available:
        return HardGateResult(False, "budget_exhausted")
    if not gate.cooldown_clear:
        return HardGateResult(False, "cooldown_active")
    if not proactive_allowed_in_mode(gate.mode):
        return HardGateResult(False, "mode_suppresses_initiative")
    if gate.quiet_active:
        return HardGateResult(False, "quiet_hours")
    return HardGateResult(True, "ok")


@dataclass(frozen=True, slots=True)
class ProactiveExpressionFactors:
    """Soft factors for unsolicited speech (DESIGN 11.3)."""

    salience: float
    novelty: float
    initiative: float
    inhibition: float
    refractory_penalty: float  # (1 - refractory_factor); higher = more recently spoke

    # Weights (v0 defaults; tunable). Kept here so the model is self-documenting.
    w_salience: float = 1.4
    w_novelty: float = 1.0
    w_initiative: float = 1.2
    w_inhibition: float = 1.4
    w_refractory: float = 1.6
    bias: float = -1.0  # baseline reluctance: a thought is not a message


def proactive_speak_probability(f: ProactiveExpressionFactors) -> float:
    """P(speak | thought) for unsolicited speech (DESIGN 11.3)."""
    z = (
        f.bias
        + f.w_salience * f.salience
        + f.w_novelty * f.novelty
        + f.w_initiative * f.initiative
        - f.w_inhibition * f.inhibition
        - f.w_refractory * f.refractory_penalty
    )
    return sigmoid(z)


@dataclass(frozen=True, slots=True)
class OptionalResponseFactors:
    """Soft factors for optional *reactive* replies (DESIGN 14.2)."""

    desire: float
    relevance: float
    initiative: float
    inhibition: float
    active_mode: bool

    w_desire: float = 1.3
    w_relevance: float = 1.1
    w_initiative: float = 0.8
    w_inhibition: float = 1.2
    bias: float = -0.2


def optional_response_probability(f: OptionalResponseFactors) -> float:
    """P(respond) for an optional social/informational turn (DESIGN 14.2).

    ACTIVE mode raises response probability for substantive turns relative to IDLE (DESIGN 14.2).
    """
    active_boost = 0.9 if f.active_mode else 0.0
    z = (
        f.bias
        + active_boost
        + f.w_desire * f.desire
        + f.w_relevance * f.relevance
        + f.w_initiative * f.initiative
        - f.w_inhibition * f.inhibition
    )
    return sigmoid(z)
