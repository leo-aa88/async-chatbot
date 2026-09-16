"""Shared row <-> value conversion helpers (DRY across stores)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..clock import from_rfc3339, to_rfc3339


def txt(moment: datetime | None) -> str | None:
    """Serialize an optional datetime to RFC 3339 text."""
    return None if moment is None else to_rfc3339(moment)


def dt(value: Any) -> datetime | None:
    """Parse optional RFC 3339 text into a datetime."""
    return None if value is None else from_rfc3339(str(value))


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


def as_bool(value: Any) -> bool:
    return bool(int(value)) if value is not None else False
