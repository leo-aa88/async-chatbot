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
from aca.domain.events import HumanMessage
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
