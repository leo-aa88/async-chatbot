"""Adversarial: a model result that finishes while the service stops is durably accepted.

stop() cancels the reducer loop, then drains in-flight workers. A result finished during the drain
used to land on a queue nobody read: never durably accepted, the work left RUNNING, and the model
call repeated after recovery reclaimed the lease.
"""

from __future__ import annotations

import asyncio

from aca import ids
from aca.config import Config
from aca.domain.enums import WorkKind, WorkStatus
from aca.domain.events import HumanMessage, StochasticWake
from aca.persistence.stores import Stores
from aca.service.service import AgentService
from aca.workers.fake_llm import FakeLLMWorker


class _SlowLLM:
    """Takes long enough that stop() is called while the call is in flight."""

    def __init__(self) -> None:
        self._inner = FakeLLMWorker()

    async def run(self, snapshot):
        await asyncio.sleep(0.3)
        return await self._inner.run(snapshot)


async def test_a_result_finished_during_stop_is_durably_accepted(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}), llm_worker=_SlowLLM())
    await service.start()
    await service.ingest_human_message(HumanMessage(
        event_id=ids.new_id(ids.EVENT), timestamp=service.clock.now_utc(), source="cli",
        text="Explain this stack trace."))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while loop.time() < deadline:  # wait until the LLM call is in flight
        rows = service.stores.db.query_all(
            "SELECT status FROM work_items WHERE kind=?", (WorkKind.LLM_COGNITION.value,))
        if rows and rows[0]["status"] == WorkStatus.RUNNING.value:
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("the LLM call never started")
    await service.stop()

    stores = Stores.open(tmp_path / "agent.db")
    results = stores.db.query_all("SELECT 1 FROM events WHERE type='LLMResult'")
    llm_work = [w for w in stores.work.pending() if w.kind is WorkKind.LLM_COGNITION]
    running = stores.db.query_all("SELECT 1 FROM work_items WHERE status=?", (WorkStatus.RUNNING.value,))
    stores.close()
    assert results, "the in-flight model result should have been durably accepted"
    assert llm_work == [] and running == [], "the model call must not be left to be repeated"


def _llm_work(tmp_path) -> list:
    stores = Stores.open(tmp_path / "agent.db")
    rows = stores.db.query_all("SELECT status FROM work_items WHERE kind=?", (WorkKind.LLM_COGNITION.value,))
    stores.close()
    return rows


async def test_a_wake_queued_at_stop_is_dropped_not_replayed(tmp_path):
    # A wake that fired just before stop() is still valid during shutdown (lifecycle is RUNNING until
    # AgentSuspending; session and generation are current). Reducing it would run a wake cycle whose
    # proactive work recovery dispatches after downtime, replaying a missed wake (invariant 9).
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    ctx = service.reducer.context
    service._enqueue(StochasticWake(
        event_id=ids.new_id(ids.EVENT), timestamp=service.clock.now_utc(), source="scheduler",
        runtime_session_id=ctx.runtime_session_id, scheduler_generation=ctx.scheduler_generation,
        scheduled_at_utc=""))
    await service.stop()

    stores = Stores.open(tmp_path / "agent.db")
    wake_cycles = stores.db.query_all("SELECT 1 FROM cognition_traces WHERE trigger='StochasticWake'")
    stores.close()
    assert wake_cycles == [], "a wake queued at stop must not run a cognition cycle"
    assert _llm_work(tmp_path) == [], "and must leave no proactive work for recovery to dispatch"


async def test_a_human_message_queued_at_stop_is_replayed_on_the_next_start(tmp_path):
    # A HumanMessage was durably accepted at ingress, so the next start replays it. Shutdown must not
    # start a cognition cycle for it, and must not lose it either.
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    event_id = ids.new_id(ids.EVENT)
    await service.ingest_human_message(HumanMessage(
        event_id=event_id, timestamp=service.clock.now_utc(), source="cli", text="Explain this stack trace."))
    await service.stop()  # the reducer loop never got to it

    stores = Stores.open(tmp_path / "agent.db")
    unreduced = [e.event_id for e in stores.events.unreduced()]
    stores.close()
    assert event_id in unreduced, "the message must survive shutdown unreduced"
    assert _llm_work(tmp_path) == [], "no cognition cycle is started for it during shutdown"

    restarted = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await restarted.start()  # replays accepted-but-unreduced events
    try:
        assert event_id not in [e.event_id for e in restarted.stores.events.unreduced()]
        assert _llm_work(tmp_path), "the replayed message gets its reply cycle after restart"
    finally:
        await restarted.stop()
