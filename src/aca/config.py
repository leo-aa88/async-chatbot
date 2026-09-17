"""Configuration model (DESIGN 18, 25).

The v0 parameter space is deliberately small; each parameter has one primary mechanism
(DESIGN 18). Config is immutable once loaded. Duration-like fields accept human-readable
strings (``45m``) and are converted to explicit model units (hours/seconds) at load time so
no implicit unit conversion happens later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .durations import parse_hours, parse_seconds
from .errors import ConfigError


def _fraction(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number in [0, 1]") from exc
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {number}")
    return number


@dataclass(frozen=True, slots=True)
class Temperament:
    """Priors that shape *expression*, not thought opportunity (DESIGN 11, 17, 18)."""

    initiative: float = 0.50
    inhibition: float = 0.50
    persistence: float = 0.50

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Temperament:
        return Temperament(
            initiative=_fraction("temperament.initiative", data.get("initiative", 0.50)),
            inhibition=_fraction("temperament.inhibition", data.get("inhibition", 0.50)),
            persistence=_fraction("temperament.persistence", data.get("persistence", 0.50)),
        )


@dataclass(frozen=True, slots=True)
class QuietHours:
    """A hard gate on unsolicited outward speech during a local-time interval (DESIGN 11.6)."""

    enabled: bool = False
    start_local: str = "01:00"
    end_local: str = "08:00"

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> QuietHours:
        return QuietHours(
            enabled=bool(data.get("enabled", False)),
            start_local=str(data.get("start_local", "01:00")),
            end_local=str(data.get("end_local", "08:00")),
        )


@dataclass(frozen=True, slots=True)
class Timing:
    """Expression-timing constants. Values are stored in explicit units."""

    refractory_tau_hours: float = 2.0
    proactive_cooldown_seconds: float = parse_seconds("45m")
    proactive_burst_window_seconds: float = parse_seconds("5m")
    proactive_ttl_seconds: float = parse_seconds("6h")
    quiet_hours: QuietHours = field(default_factory=QuietHours)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Timing:
        return Timing(
            refractory_tau_hours=parse_hours(data.get("refractory_tau_hours", 2.0)),
            proactive_cooldown_seconds=parse_seconds(data.get("proactive_cooldown", "45m")),
            proactive_burst_window_seconds=parse_seconds(
                data.get("proactive_burst_window", "5m")
            ),
            proactive_ttl_seconds=parse_seconds(data.get("proactive_ttl", "6h")),
            quiet_hours=QuietHours.from_mapping(data.get("quiet_hours", {})),
        )


@dataclass(frozen=True, slots=True)
class Cognition:
    """Internal-activation (thought-opportunity) parameters (DESIGN 10, 12)."""

    spontaneous_activation_rate_per_hour: float = 0.25
    selection_temperature: float = 0.7
    null_candidate_score: float = 0.5
    semantic_worthiness_floor: float = 0.35

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Cognition:
        rate = float(data.get("spontaneous_activation_rate_per_hour", 0.25))
        if rate < 0.0:
            raise ConfigError("cognition.spontaneous_activation_rate_per_hour must be >= 0")
        temperature = float(data.get("selection_temperature", 0.7))
        if temperature <= 0.0:
            raise ConfigError("cognition.selection_temperature must be > 0")
        return Cognition(
            spontaneous_activation_rate_per_hour=rate,
            selection_temperature=temperature,
            null_candidate_score=float(data.get("null_candidate_score", 0.5)),
            semantic_worthiness_floor=_fraction(
                "cognition.semantic_worthiness_floor",
                data.get("semantic_worthiness_floor", 0.35),
            ),
        )


@dataclass(frozen=True, slots=True)
class Conversation:
    """Cadence thresholds for inferring conversation mode (DESIGN 7.4, open tuning 31.1).

    A human turn within ``active_within`` keeps the conversation ACTIVE (proactive initiative
    suppressed); within ``idle_within`` it is IDLE; beyond that, DORMANT. Configurable so
    proactive behavior can be exercised without waiting the production defaults.
    """

    active_within_seconds: float = parse_seconds("3m")
    idle_within_seconds: float = parse_seconds("30m")

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Conversation:
        active = parse_seconds(data.get("active_within", "3m"))
        idle = parse_seconds(data.get("idle_within", "30m"))
        if idle < active:
            raise ConfigError("conversation.idle_within must be >= active_within")
        return Conversation(active_within_seconds=active, idle_within_seconds=idle)


@dataclass(frozen=True, slots=True)
class Memory:
    """Memory-availability parameters (DESIGN 12)."""

    default_decay_half_life_hours: float = 24.0
    persistence_decay_coefficient: float = 0.6  # alpha in k_eff = k_base(1 - alpha*P)
    candidate_activation_floor: float = 0.05
    # A candidate the agent just spoke about proactively is excluded from re-selection for this
    # long, so it doesn't nag (DESIGN 6, 11.3, 12.7 repetition regulation). Human-scale by default.
    repeat_suppression_seconds: float = parse_seconds("6h")

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Memory:
        half_life = float(data.get("default_decay_half_life_hours", 24.0))
        if half_life <= 0.0:
            raise ConfigError("memory.default_decay_half_life_hours must be > 0")
        return Memory(
            default_decay_half_life_hours=half_life,
            persistence_decay_coefficient=_fraction(
                "memory.persistence_decay_coefficient",
                data.get("persistence_decay_coefficient", 0.6),
            ),
            candidate_activation_floor=_fraction(
                "memory.candidate_activation_floor",
                data.get("candidate_activation_floor", 0.05),
            ),
            repeat_suppression_seconds=parse_seconds(data.get("repeat_suppression", "6h")),
        )


@dataclass(frozen=True, slots=True)
class Budgets:
    """Hard proactive-cognition limits that never bank across downtime (DESIGN 25)."""

    proactive_llm_calls_per_day: int = 12
    proactive_llm_calls_per_hour: int = 2
    proactive_messages_per_hour: int = 2
    max_enrichment_attempts: int = 3

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Budgets:
        def positive_int(name: str, value: Any) -> int:
            number = int(value)
            if number < 0:
                raise ConfigError(f"{name} must be >= 0")
            return number

        return Budgets(
            proactive_llm_calls_per_day=positive_int(
                "budgets.proactive_llm_calls_per_day",
                data.get("proactive_llm_calls_per_day", 12),
            ),
            proactive_llm_calls_per_hour=positive_int(
                "budgets.proactive_llm_calls_per_hour",
                data.get("proactive_llm_calls_per_hour", 2),
            ),
            proactive_messages_per_hour=positive_int(
                "budgets.proactive_messages_per_hour",
                data.get("proactive_messages_per_hour", 2),
            ),
            max_enrichment_attempts=positive_int(
                "budgets.max_enrichment_attempts", data.get("max_enrichment_attempts", 3)
            ),
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level immutable configuration."""

    local_timezone: str = "UTC"
    rng_seed: int | None = None
    cognition: Cognition = field(default_factory=Cognition)
    temperament: Temperament = field(default_factory=Temperament)
    timing: Timing = field(default_factory=Timing)
    memory: Memory = field(default_factory=Memory)
    budgets: Budgets = field(default_factory=Budgets)
    conversation: Conversation = field(default_factory=Conversation)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Config:
        seed = data.get("rng_seed")
        return Config(
            local_timezone=str(data.get("local_timezone", "UTC")),
            rng_seed=None if seed is None else int(seed),
            cognition=Cognition.from_mapping(data.get("cognition", {})),
            temperament=Temperament.from_mapping(data.get("temperament", {})),
            timing=Timing.from_mapping(data.get("timing", {})),
            memory=Memory.from_mapping(data.get("memory", {})),
            budgets=Budgets.from_mapping(data.get("budgets", {})),
            conversation=Conversation.from_mapping(data.get("conversation", {})),
        )

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a copy with top-level fields replaced (useful for tests)."""
        return replace(self, **overrides)
