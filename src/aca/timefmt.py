"""Human-readable local timestamp formatting for the CLI (DESIGN 28.2).

Durable timestamps are stored tz-aware in UTC and travel the IPC wire as ISO-8601 strings
(machine-comparable). For *display* only, this renders them in the agent's configured local
timezone as ``YYYY-MM-DD HH:MM:SS TZ`` so a human reading ``aca logs``/``memories``/``topics`` or
the chat transcript sees a familiar wall-clock time, not a UTC ISO string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MISSING = "—"


def _zone(tz_name: str) -> ZoneInfo | type[UTC]:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def _coerce(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def human_time(value: object, tz_name: str = "UTC") -> str:
    """Render an ISO string or ``datetime`` as local ``YYYY-MM-DD HH:MM:SS TZ`` (or ``—``)."""
    dt = _coerce(value)
    if dt is None:
        return _MISSING
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_zone(tz_name)).strftime("%Y-%m-%d %H:%M:%S %Z")


def clock_time(value: object, tz_name: str = "UTC") -> str:
    """Compact local ``HH:MM:SS`` for per-message chat stamps (or ``—``)."""
    dt = _coerce(value)
    if dt is None:
        return _MISSING
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_zone(tz_name)).strftime("%H:%M:%S")
