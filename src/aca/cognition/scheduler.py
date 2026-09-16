"""Stochastic wake scheduling (DESIGN 10).

A piecewise-constant, state-dependent internal-activation hazard controls *opportunities* for
something to enter attention — not whether the agent speaks:

    λ_think = λ0 * f_observed_silence * f_salience * f_unfinished
    Δt ~ Exp(λ_think);  t_wake = t_n + Δt

Expression variables (initiative, inhibition, cooldown, refractory, quiet hours) are
**deliberately absent** here (DESIGN 10, invariant, 11.5): they regulate expression, not
thought opportunity. ``f_observed_silence`` uses agent-observed *active* time only — downtime
never masquerades as relational absence (DESIGN 10, invariant 31).

The multiplier shapes are bounded and monotonic; their exact form is an empirical v0 parameter
(DESIGN 31.14). They are kept simple and pure so behavior is easy to reason about and replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Cognition
from ..rng import Rng

# Tunable shape constants (DESIGN 31.14). Bounded so no single signal can dominate wildly.
_SILENCE_GAIN = 1.5
_SILENCE_TAU_HOURS = 6.0
_SALIENCE_GAIN = 1.0
_UNFINISHED_GAIN = 0.75
_UNFINISHED_CAP = 4


@dataclass(frozen=True, slots=True)
class WakeSignals:
    """State-derived inputs to the hazard. All time is *active-observed*, never raw wall time."""

    active_observed_silence_hours: float = 0.0
    max_candidate_salience: float = 0.0
    unfinished_count: int = 0


def f_observed_silence(hours: float) -> float:
    """Saturating increase in thought opportunity with active observed silence (DESIGN 10)."""
    hours = max(0.0, hours)
    return 1.0 + _SILENCE_GAIN * (1.0 - math.exp(-hours / _SILENCE_TAU_HOURS))


def f_salience(max_salience: float) -> float:
    """More salient available material raises thought opportunity."""
    return 1.0 + _SALIENCE_GAIN * max(0.0, min(1.0, max_salience))


def f_unfinished(count: int) -> float:
    """Unresolved threads raise thought opportunity, with diminishing returns."""
    capped = min(max(0, count), _UNFINISHED_CAP)
    return 1.0 + _UNFINISHED_GAIN * (capped / _UNFINISHED_CAP)


def compute_lambda(cognition: Cognition, signals: WakeSignals) -> float:
    """Compute the current per-hour internal-activation hazard ``λ_think``."""
    return (
        cognition.spontaneous_activation_rate_per_hour
        * f_observed_silence(signals.active_observed_silence_hours)
        * f_salience(signals.max_candidate_salience)
        * f_unfinished(signals.unfinished_count)
    )


def sample_delay_hours(cognition: Cognition, signals: WakeSignals, rng: Rng) -> float:
    """Sample the next wake delay Δt in hours. Returns ``inf`` if the hazard is zero."""
    lam = compute_lambda(cognition, signals)
    return rng.exponential(lam)
