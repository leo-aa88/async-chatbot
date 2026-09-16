"""Reducer execution context and result types.

``ReducerContext`` is the explicit dependency bundle handlers receive (SOLID: no hidden
globals). The service updates ``runtime_session_id``/``scheduler_generation`` on it across
resume/reschedule. ``ReduceResult`` reports side effects to the service; handlers never dispatch
work or touch timers themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..clock import Clock
from ..config import Config
from ..persistence.stores import Stores
from ..rng import Rng


@dataclass(slots=True)
class ReducerContext:
    stores: Stores
    clock: Clock
    rng: Rng
    config: Config
    agent_id: str
    runtime_session_id: str
    scheduler_generation: int


@dataclass(slots=True)
class ReduceResult:
    """What the reducer committed and what the service must do next."""

    revision: int
    dispatch_work_ids: list[str] = field(default_factory=list)
    reschedule: bool = False
    deliver: bool = False
    accepted: bool = True
    note: str = ""
