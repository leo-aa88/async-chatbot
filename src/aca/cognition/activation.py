"""Lazy activation decay and reinforcement (DESIGN 12.4-12.6).

Activation decays as ``A(t) = A0 * e^{-k Δt}`` where ``k`` is per hour and ``Δt`` is elapsed
hours since the value was last materialized. Decay is computed on read — there is no periodic
decay tick (DESIGN 10.1). Persistence temperament slows effective decay (12.5):

    k_effective = k_base * (1 - alpha * P)

All functions are pure. ``Δt`` is derived from explicit timestamps, never a hidden wall read.
"""

from __future__ import annotations

import math
from datetime import datetime

_LN2 = math.log(2.0)


def half_life_to_rate_per_hour(half_life_hours: float) -> float:
    """Convert a decay half-life (hours) to a per-hour decay constant ``k``."""
    if half_life_hours <= 0.0:
        raise ValueError("half_life_hours must be > 0")
    return _LN2 / half_life_hours


def effective_decay_rate(base_rate_per_hour: float, persistence: float, alpha: float) -> float:
    """Apply persistence temperament to a base decay rate (DESIGN 12.5).

    ``persistence`` and ``alpha`` are both in [0, 1]; the result is clamped to be non-negative
    so a fully-persistent, high-alpha config cannot produce negative (growing) decay.
    """
    factor = 1.0 - alpha * persistence
    return max(0.0, base_rate_per_hour * factor)


def elapsed_hours(since: datetime, now: datetime) -> float:
    """Non-negative elapsed hours between two UTC datetimes.

    Clamped at zero so a slightly out-of-order timestamp can never *increase* activation via a
    negative exponent.
    """
    seconds = (now - since).total_seconds()
    return max(0.0, seconds) / 3600.0


def decayed_activation(
    stored_activation: float,
    stored_at: datetime,
    now: datetime,
    base_rate_per_hour: float,
    *,
    persistence: float = 0.0,
    alpha: float = 0.0,
) -> float:
    """Return the effective activation at ``now`` given a value materialized at ``stored_at``."""
    k = effective_decay_rate(base_rate_per_hour, persistence, alpha)
    dt = elapsed_hours(stored_at, now)
    return stored_activation * math.exp(-k * dt)


def reinforced(current_activation: float, boost: float, *, ceiling: float = 1.0) -> float:
    """Raise activation by ``boost`` (DESIGN 12.6), clamped to ``ceiling``."""
    return min(ceiling, max(0.0, current_activation + boost))
