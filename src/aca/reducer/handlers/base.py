"""Handler outcome type and shared cycle-type constants."""

from __future__ import annotations

from dataclasses import dataclass, field

# Re-exported so handlers can keep importing them from here; defined in the neutral domain layer
# because they are part of the reducer<->worker snapshot contract.
from ...domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE, CYCLE_REACTIVE_OPTIONAL
from ...domain.runtime import CognitionTrace

__all__ = [
    "CYCLE_MANDATORY",
    "CYCLE_PROACTIVE",
    "CYCLE_REACTIVE_OPTIONAL",
    "HandlerOutcome",
]


@dataclass(slots=True)
class HandlerOutcome:
    """Side effects a handler asks the service to perform after the commit."""

    dispatch_work_ids: list[str] = field(default_factory=list)
    reschedule: bool = False
    deliver: bool = False
    trace: CognitionTrace | None = None
    note: str = ""
    # False for no-op events (stale wake, duplicate/unknown result): no revision advance.
    committed: bool = True
