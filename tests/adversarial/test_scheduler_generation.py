"""Adversarial: every reschedule is a new scheduler generation (DESIGN 10.6, invariant 11).

The generation used to move only once per start (in recovery), so a wake already queued before a
reschedule still carried the current generation and fired alongside the new schedule's wake.
"""

from __future__ import annotations

from aca import ids
from aca.config import Config
from aca.domain.events import StochasticWake
from aca.service.service import AgentService


async def test_rescheduling_bumps_and_persists_the_generation(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    try:
        before = service.reducer.context.scheduler_generation
        service._schedule_wake()
        after = service.reducer.context.scheduler_generation
        assert after == before + 1
        assert service.stores.state.scheduler_generation() == after  # monotonic across restarts
    finally:
        await service.stop()


async def test_a_wake_queued_before_a_reschedule_is_dropped_as_stale(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    try:
        ctx = service.reducer.context
        queued = StochasticWake(
            event_id=ids.new_id(ids.EVENT), timestamp=service.clock.now_utc(), source="scheduler",
            runtime_session_id=ctx.runtime_session_id, scheduler_generation=ctx.scheduler_generation,
            scheduled_at_utc="",
        )
        service._schedule_wake()  # a committed change resamples the schedule

        result = service.reducer.reduce(queued)

        assert result.note == "stale_wake"
    finally:
        await service.stop()


async def test_the_timer_callback_carries_its_own_schedules_generation(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    try:
        generation = service.reducer.context.scheduler_generation
        service._on_wake_fired(generation)
        wake = service._queue.get_nowait()
        assert wake.scheduler_generation == generation
    finally:
        await service.stop()
