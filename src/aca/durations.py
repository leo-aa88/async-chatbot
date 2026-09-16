"""Human-readable duration parsing.

Config may use strings such as ``45m`` or ``2h``; the model's canonical time unit is
**hours** (DESIGN 10.3). Conversion to model units is always explicit and happens here, never
implicitly scattered through the code.
"""

from __future__ import annotations

import re

from .errors import DurationParseError

_UNIT_SECONDS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])")


def parse_seconds(value: str | int | float) -> float:
    """Parse a duration into seconds.

    Accepts a bare number (interpreted as seconds), or a string of one or more
    ``<number><unit>`` tokens where unit is ``s``, ``m``, ``h``, or ``d`` (e.g. ``"1h30m"``).
    """
    if isinstance(value, (int, float)):
        if value < 0:
            raise DurationParseError(f"duration must be non-negative: {value!r}")
        return float(value)

    text = value.strip().lower()
    if not text:
        raise DurationParseError("empty duration string")

    # A bare numeric string means seconds.
    try:
        return parse_seconds(float(text))
    except ValueError:
        pass

    matches = list(_TOKEN.finditer(text))
    if not matches or "".join(m.group(0).replace(" ", "") for m in matches) != text.replace(
        " ", ""
    ):
        raise DurationParseError(f"cannot parse duration: {value!r}")

    total = 0.0
    for amount, unit in (m.groups() for m in matches):
        total += float(amount) * _UNIT_SECONDS[unit]
    return total


def parse_hours(value: str | int | float) -> float:
    """Parse a duration into hours (the canonical model unit for rates and decay)."""
    return parse_seconds(value) / 3600.0
