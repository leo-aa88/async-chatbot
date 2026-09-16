"""Clock abstraction with explicit semantics (DESIGN 10.5, invariant 33).

Three clock roles must never be conflated:

* **monotonic** — in-process timer durations and timeouts. Immune to wall-clock jumps.
* **UTC wall clock** — durable event timestamps, cross-restart elapsed time, budget windows.
* **configured local time** — quiet hours and user-facing time-of-day language only.

Cognition and the reducer receive a ``Clock`` by injection and must never read the wall clock
directly. ``ManualClock`` gives tests full control over both time axes independently, which is
what makes downtime/suspension and stochastic timing testable and replayable.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo


def to_rfc3339(moment: datetime) -> str:
    """Serialize a timezone-aware datetime as an RFC 3339 / UTC string."""
    if moment.tzinfo is None:
        raise ValueError("refusing to serialize a naive datetime")
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def from_rfc3339(text: str) -> datetime:
    """Parse an RFC 3339 timestamp into a timezone-aware UTC datetime."""
    normalized = text.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(UTC)


@runtime_checkable
class Clock(Protocol):
    """A source of time with separated monotonic and wall-clock axes."""

    def now_utc(self) -> datetime:
        """Current UTC wall-clock time (timezone-aware)."""

    def monotonic(self) -> float:
        """Monotonic seconds; only differences are meaningful."""

    def local_tz(self) -> tzinfo:
        """The configured local timezone for quiet hours / time-of-day."""

    def now_local(self) -> datetime:
        """Current time in the configured local timezone."""


class SystemClock:
    """Production clock backed by the operating system."""

    def __init__(self, local_timezone: str = "UTC") -> None:
        self._tz = ZoneInfo(local_timezone)

    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        import time

        return time.monotonic()

    def local_tz(self) -> tzinfo:
        return self._tz

    def now_local(self) -> datetime:
        return self.now_utc().astimezone(self._tz)


class ManualClock:
    """Deterministic clock for tests and replay.

    The wall and monotonic axes advance independently, so a test can simulate host downtime
    (wall advances, monotonic advances) or a suspended timer without coupling the two.
    """

    def __init__(
        self,
        start: datetime | None = None,
        local_timezone: str = "UTC",
        monotonic_start: float = 0.0,
    ) -> None:
        self._utc = (start or datetime(2026, 1, 1, tzinfo=UTC)).astimezone(UTC)
        self._mono = monotonic_start
        self._tz = ZoneInfo(local_timezone)

    def now_utc(self) -> datetime:
        return self._utc

    def monotonic(self) -> float:
        return self._mono

    def local_tz(self) -> tzinfo:
        return self._tz

    def now_local(self) -> datetime:
        return self._utc.astimezone(self._tz)

    # --- test controls -------------------------------------------------------------------
    def advance(self, seconds: float, *, monotonic: bool = True) -> None:
        """Advance the wall clock, and by default the monotonic clock in lockstep."""
        from datetime import timedelta

        self._utc = self._utc + timedelta(seconds=seconds)
        if monotonic:
            self._mono += seconds

    def advance_wall_only(self, seconds: float) -> None:
        """Advance wall time without advancing monotonic time (models host suspend/downtime)."""
        self.advance(seconds, monotonic=False)

    def set_local_timezone(self, name: str) -> None:
        self._tz = ZoneInfo(name)
