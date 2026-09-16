"""The serialized reducer: type dispatch, atomic commit, revision advance (DESIGN 8.1, 24).

    S_{n+1} = R(S_n, E_n, A_n)

One handler runs per event inside one transaction. A committed transition advances the
monotonic ``state_revision`` (only here, invariant 2); no-op events (stale wakes, duplicate
results) do not. Every event is durably logged and marked reduced in the same transaction as
its state change, so an acknowledged event cannot be lost or reduced twice (invariant 14).
Handlers request at most one generative call and never re-enter the reducer (invariant 3, 7).
"""

from __future__ import annotations

from dataclasses import replace

from ..domain.events import (
    AgentResumed,
    AgentStarted,
    AgentSuspending,
    DeliveryResult,
    EmbeddingResult,
    Event,
    HumanMessage,
    LLMResult,
    RuntimeInterruptionDetected,
    StochasticWake,
    event_type_name,
)
from .context import ReducerContext, ReduceResult
from .handlers.base import HandlerOutcome
from .handlers.delivery_result import handle_delivery_result
from .handlers.embedding_result import handle_embedding_result
from .handlers.human import handle_human_message
from .handlers.lifecycle import (
    handle_agent_resumed,
    handle_agent_started,
    handle_agent_suspending,
    handle_runtime_interruption,
)
from .handlers.llm_result import handle_llm_result
from .handlers.wake import handle_stochastic_wake


class Reducer:
    """Dispatches events to type handlers and enforces the single-writer commit discipline."""

    def __init__(self, ctx: ReducerContext) -> None:
        self._ctx = ctx
        self._handlers = {
            HumanMessage: handle_human_message,
            StochasticWake: handle_stochastic_wake,
            LLMResult: handle_llm_result,
            EmbeddingResult: handle_embedding_result,
            DeliveryResult: handle_delivery_result,
            AgentStarted: handle_agent_started,
            AgentSuspending: handle_agent_suspending,
            AgentResumed: handle_agent_resumed,
            RuntimeInterruptionDetected: handle_runtime_interruption,
        }

    @property
    def context(self) -> ReducerContext:
        return self._ctx

    def reduce(self, event: Event) -> ReduceResult:
        handler = self._handlers.get(type(event))
        if handler is None:
            return ReduceResult(revision=self._ctx.stores.state.state_revision(), accepted=False,
                                note=f"no_handler:{event_type_name(event)}")

        now = self._ctx.clock.now_utc()
        stores = self._ctx.stores
        with stores.db.transaction():
            stores.events.accept(event, now)  # idempotent; client events already accepted
            basis = stores.state.state_revision()
            outcome: HandlerOutcome = handler(self._ctx, event)
            revision = stores.state.advance_revision() if outcome.committed else basis
            self._persist_trace(outcome, basis, revision)
            stores.events.mark_reduced(event.event_id, now)

        return ReduceResult(
            revision=revision,
            dispatch_work_ids=list(outcome.dispatch_work_ids),
            reschedule=outcome.reschedule,
            deliver=outcome.deliver,
            note=outcome.note,
        )

    def _persist_trace(self, outcome: HandlerOutcome, basis: int, revision: int) -> None:
        if outcome.trace is None:
            return
        trace = replace(
            outcome.trace,
            basis_revision=basis,
            commit_revision=revision,
            rng_seed_fragment=self._ctx.rng.seed_fragment(),
        )
        self._ctx.stores.work.insert_trace(trace)
