"""Handler outcome type and shared cycle-type constants."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.runtime import CognitionTrace

# How a generative cycle should be finalized by the LLMResult handler.
CYCLE_MANDATORY = "mandatory"          # obligated reply (DESIGN 14.1)
CYCLE_REACTIVE_OPTIONAL = "reactive"   # chosen optional reply to a human turn (DESIGN 14.2)
CYCLE_PROACTIVE = "proactive"          # unsolicited initiative from a wake (DESIGN 14.3)


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
