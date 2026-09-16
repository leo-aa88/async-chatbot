"""AgentService: the resident daemon (DESIGN 28, 29).

One reducer coroutine consumes an ``asyncio.Queue`` (the single logical writer, DESIGN 8.1).
Ingress durably accepts a human event and ACKs *before* enqueuing it for reduction (invariant
14). A monotonic timer submits stochastic wakes; workers run as separate tasks and only return
result events. The service never mutates durable agent state except through the reducer.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from .. import ids
from ..clock import Clock, SystemClock
from ..cognition.scheduler import sample_delay_hours
from ..config import Config
from ..domain.enums import ExitKind, LifecycleState
from ..domain.events import (
    AgentResumed,
    AgentStarted,
    AgentSuspending,
    Event,
    HumanMessage,
    RuntimeInterruptionDetected,
    StochasticWake,
)
from ..persistence.stores import Stores
from ..reducer.context import ReducerContext
from ..reducer.reducer import Reducer
from ..reducer.support import build_candidates, wake_signals
from ..rng import Rng
from ..workers.base import EmbeddingWorker, LLMWorker
from ..workers.fake_embedding import FakeEmbeddingWorker
from ..workers.fake_llm import FakeLLMWorker
from .delivery import ClientSink, DeliveryPump
from .dispatcher import Dispatcher
from .lock import SingleInstanceLock
from .recovery import recover
from .timer import CancellableTimer

_HEARTBEAT_SECONDS = 30.0


async def _null_sink(channel: str, payload: str, delivery_key: str, message_id: str) -> bool:
    return False  # no client connected


class AgentService:
    def __init__(
        self,
        data_dir: str | Path,
        config: Config | None = None,
        *,
        clock: Clock | None = None,
        rng: Rng | None = None,
        llm_worker: LLMWorker | None = None,
        embedding_worker: EmbeddingWorker | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._config = config or Config()
        self._clock = clock or SystemClock(self._config.local_timezone)
        self._rng = rng or Rng(self._config.rng_seed)
        self._llm = llm_worker or FakeLLMWorker()
        self._embedding = embedding_worker or FakeEmbeddingWorker()
        self._lock = SingleInstanceLock(self._data_dir)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._timer = CancellableTimer()
        self._sink: ClientSink = _null_sink
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        # Wired in start().
        self._stores: Stores | None = None
        self._reducer: Reducer | None = None
        self._dispatcher: Dispatcher | None = None
        self._pump: DeliveryPump | None = None

    # --- lifecycle -----------------------------------------------------------------------
    async def start(self) -> None:
        self._lock.acquire()
        self._stores = Stores.open(self._data_dir / "agent.db")
        plan = recover(self._stores, self._clock, self._config)

        ctx = ReducerContext(
            stores=self._stores, clock=self._clock, rng=self._rng, config=self._config,
            agent_id=self._stores.identity.load_identity().agent_id,
            runtime_session_id=plan.session_id, scheduler_generation=plan.scheduler_generation,
        )
        self._reducer = Reducer(ctx)
        self._dispatcher = Dispatcher(self._stores, self._clock, self._llm, self._embedding, self._enqueue)
        self._pump = DeliveryPump(
            self._stores, self._clock, self._config, ctx, lambda *a, **k: self._sink(*a, **k),
            self._enqueue,
        )

        now = self._clock.now_utc()
        self._reducer.reduce(AgentStarted(ids.new_id(ids.EVENT), now, "service", plan.session_id))
        if plan.exit_kind is ExitKind.UNCLEAN_INTERRUPTION:
            self._reducer.reduce(
                RuntimeInterruptionDetected(
                    ids.new_id(ids.EVENT), now, "service", plan.session_id, plan.downtime_seconds
                )
            )
        elif not plan.is_new_agent:
            self._reducer.reduce(
                AgentResumed(ids.new_id(ids.EVENT), now, "service", plan.session_id, plan.downtime_seconds)
            )

        self._set_lifecycle(LifecycleState.RUNNING)
        self._running = True

        # Replay accepted-but-unreduced events, then redispatch recovered work.
        for event in plan.replay_events:
            await self._process(event)
        for work_id in plan.dispatch_work_ids:
            self._dispatcher.dispatch(work_id)

        self._schedule_wake()
        self._loop_task = asyncio.ensure_future(self._reducer_loop())
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    async def stop(self, exit_kind: ExitKind = ExitKind.CLEAN_SUSPEND) -> None:
        if not self._running:
            return
        self._running = False
        self._timer.cancel()
        for task in (self._loop_task, self._heartbeat_task):
            if task is not None:
                task.cancel()
        if self._dispatcher is not None:
            await self._dispatcher.drain()

        now = self._clock.now_utc()
        assert self._reducer is not None and self._stores is not None
        self._reducer.reduce(
            AgentSuspending(ids.new_id(ids.EVENT), now, "service", self._reducer.context.runtime_session_id,
                            exit_kind.value)
        )
        with self._stores.db.transaction():
            self._stores.identity.close_session(
                self._reducer.context.runtime_session_id, now, exit_kind
            )
        self._stores.close()
        self._lock.release()

    # --- ingress -------------------------------------------------------------------------
    async def ingest_human_message(self, event: HumanMessage) -> bool:
        """Durably accept + ACK before enqueuing for reduction (DESIGN 6.2, invariant 14)."""
        assert self._stores is not None
        with self._stores.db.transaction():
            newly = self._stores.events.accept(event, self._clock.now_utc())
        if newly:
            self._enqueue(event)
        return newly

    # --- delivery sink wiring ------------------------------------------------------------
    def set_sink(self, sink: ClientSink) -> None:
        self._sink = sink

    async def notify_client_connected(self) -> None:
        """On (re)connect, attempt delivery: mandatory first, one revalidated proactive."""
        if self._pump is not None:
            await self._pump.pump()

    # --- core loop -----------------------------------------------------------------------
    async def _reducer_loop(self) -> None:
        while self._running:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._process(event)

    async def _process(self, event: Event) -> None:
        assert self._reducer is not None and self._dispatcher is not None and self._pump is not None
        from ..domain.events import DeliveryResult

        result = self._reducer.reduce(event)
        if isinstance(event, DeliveryResult):
            self._pump.on_delivery_result(event.message_id)
        for work_id in result.dispatch_work_ids:
            self._dispatcher.dispatch(work_id)
        if result.reschedule:
            self._schedule_wake()
        if result.deliver:
            await self._pump.pump()

    def _enqueue(self, event: Event) -> None:
        self._queue.put_nowait(event)

    # --- wake scheduling -----------------------------------------------------------------
    def _schedule_wake(self) -> None:
        assert self._reducer is not None and self._stores is not None
        now = self._clock.now_utc()
        candidates = build_candidates(self._reducer.context, now)
        signals = wake_signals(self._reducer.context, candidates, now)
        delay_hours = sample_delay_hours(self._config.cognition, signals, self._rng)
        self._timer.reschedule(delay_hours * 3600.0, self._on_wake_fired)

    def _on_wake_fired(self) -> None:
        assert self._reducer is not None
        ctx = self._reducer.context
        self._enqueue(
            StochasticWake(
                event_id=ids.new_id(ids.EVENT),
                timestamp=self._clock.now_utc(),
                source="scheduler",
                runtime_session_id=ctx.runtime_session_id,
                scheduler_generation=ctx.scheduler_generation,
                scheduled_at_utc="",
            )
        )

    # --- heartbeat (operational only, invariant 35) --------------------------------------
    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(_HEARTBEAT_SECONDS)
            except asyncio.CancelledError:
                return
            self._heartbeat(_HEARTBEAT_SECONDS)

    def _heartbeat(self, elapsed_seconds: float) -> None:
        assert self._stores is not None
        now = self._clock.now_utc()
        identity = self._stores.identity.load_identity()
        conversation = self._stores.state.load_conversation()
        with self._stores.db.transaction():
            if identity is not None:
                self._stores.identity.update_identity(
                    replace(identity, last_heartbeat_at=now, last_active_at=now)
                )
            # Observed silence accrues only while RUNNING; downtime never contributes (inv 31).
            self._stores.state.save_conversation(
                replace(
                    conversation,
                    active_observed_silence_seconds=conversation.active_observed_silence_seconds
                    + elapsed_seconds,
                )
            )

    def _set_lifecycle(self, state: LifecycleState) -> None:
        assert self._stores is not None
        identity = self._stores.identity.load_identity()
        if identity is not None:
            with self._stores.db.transaction():
                self._stores.identity.update_identity(replace(identity, lifecycle_state=state))

    # --- read-only accessors for clients -------------------------------------------------
    @property
    def stores(self) -> Stores:
        assert self._stores is not None
        return self._stores

    @property
    def reducer(self) -> Reducer:
        assert self._reducer is not None
        return self._reducer

    @property
    def clock(self) -> Clock:
        return self._clock
