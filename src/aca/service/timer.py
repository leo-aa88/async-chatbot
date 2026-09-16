"""Cancellable in-process wake timer (DESIGN 10.1, 10.5).

Timer durations are driven by the event loop's monotonic clock, never a mutable wall clock, so
NTP corrections / DST / timezone changes cannot create phantom wakes (invariant 33). Any
committed state change that affects wake-relevant signals cancels and resamples the timer
(DESIGN 10.1). A fired timer only *enqueues* an event — it has no authority to cause cognition;
the reducer validates session/generation before acting (invariant 34).
"""

from __future__ import annotations

import asyncio
from typing import Callable


class CancellableTimer:
    """A single pending one-shot timer that can be rescheduled or cancelled."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    def reschedule(self, delay_seconds: float, callback: Callable[[], None]) -> None:
        """Cancel any pending fire and schedule ``callback`` after ``delay_seconds``.

        An infinite/NaN delay means "no wake scheduled" (hazard is zero) and simply cancels.
        """
        self.cancel()
        if not _is_finite_positive(delay_seconds):
            return
        self._task = asyncio.ensure_future(self._run(delay_seconds, callback))

    async def _run(self, delay_seconds: float, callback: Callable[[], None]) -> None:
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            return
        callback()

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None


def _is_finite_positive(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf")) and value > 0.0
