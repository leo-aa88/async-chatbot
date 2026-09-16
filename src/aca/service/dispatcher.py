"""Asynchronous worker dispatch (DESIGN 8.2, 28.1, invariant 39).

The reducer coroutine leases a durable work item, then hands it to a worker running as a
separate task. LLM work is network-bound async I/O; embedding work is CPU-bound and runs in a
process/thread executor so it never blocks the event loop (invariant 39). Workers only compute;
they return a *result event* to the queue and never touch the database (invariant 4). Leasing
and completion are bookkeeping performed on the reducer coroutine, keeping writes serialized.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Callable

from .. import ids
from ..clock import Clock
from ..cognition.snapshot import Snapshot
from ..domain.events import EmbeddingResult, Event, LLMResult
from ..domain.enums import WorkKind
from ..domain.runtime import WorkItem
from ..persistence.stores import Stores
from ..workers.base import EmbeddingWorker, LLMWorker

_LEASE_SECONDS = 120


class Dispatcher:
    def __init__(
        self,
        stores: Stores,
        clock: Clock,
        llm_worker: LLMWorker,
        embedding_worker: EmbeddingWorker,
        enqueue: Callable[[Event], None],
    ) -> None:
        self._stores = stores
        self._clock = clock
        self._llm = llm_worker
        self._embedding = embedding_worker
        self._enqueue = enqueue
        self._tasks: set[asyncio.Task[None]] = set()

    def dispatch(self, work_id: str) -> None:
        """Lease a work item (on the reducer coroutine) and spawn its worker task."""
        work = self._stores.work.get_work(work_id)
        if work is None:
            return
        lease_until = self._clock.now_utc() + timedelta(seconds=_LEASE_SECONDS)
        with self._stores.db.transaction():
            self._stores.work.lease(work_id, lease_until)
        work = self._stores.work.get_work(work_id)  # reload with incremented attempt
        task = asyncio.ensure_future(self._run(work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, work: WorkItem) -> None:
        try:
            event = await self._execute(work)
        except Exception as exc:  # worker failure: release lease for later recovery/retry
            with self._stores.db.transaction():
                self._stores.work.requeue(work.work_id, error=repr(exc))
            return
        self._enqueue(event)

    async def _execute(self, work: WorkItem) -> Event:
        if work.kind is WorkKind.EMBEDDING:
            return await self._run_embedding(work)
        return await self._run_llm(work)

    async def _run_embedding(self, work: WorkItem) -> Event:
        text = str(work.snapshot.get("text", ""))
        loop = asyncio.get_running_loop()
        # CPU-bound inference off the event loop (invariant 39).
        output = await loop.run_in_executor(None, _embed_sync, self._embedding, text)
        return EmbeddingResult(
            event_id=ids.new_id(ids.EVENT),
            timestamp=self._clock.now_utc(),
            source="embedding_worker",
            work_id=work.work_id,
            provisional_memory_id=str(work.snapshot.get("provisional_memory_id", "")),
            embedding_id=ids.new_id(ids.EMBEDDING),
            model_version=output.model_version,
            vector=output.vector,
        )

    async def _run_llm(self, work: WorkItem) -> Event:
        snapshot = Snapshot(
            cycle_id=work.snapshot.get("cycle_id", work.cycle_id),
            work_id=work.snapshot.get("work_id", work.work_id),
            basis_revision=work.snapshot.get("basis_revision", work.basis_revision),
            template_version=work.snapshot.get("template_version", "v0.6"),
            context=work.snapshot.get("context", {}),
        )
        output = await self._llm.run(snapshot)
        return LLMResult(
            event_id=ids.new_id(ids.EVENT),
            timestamp=self._clock.now_utc(),
            source="llm_worker",
            work_id=work.work_id,
            cycle_id=work.cycle_id,
            basis_revision=work.basis_revision,
            result=output.result,
            tokens_in=output.tokens_in,
            tokens_out=output.tokens_out,
        )

    async def drain(self) -> None:
        """Await all in-flight worker tasks (used on graceful shutdown)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


def _embed_sync(worker: EmbeddingWorker, text: str):
    """Run an async embedding worker to completion inside an executor thread."""
    return asyncio.run(worker.embed(text))
