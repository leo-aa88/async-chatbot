"""Shared test harness.

Drives the *synchronous* reducer directly with a ``ManualClock`` and a seeded ``Rng`` so every
test is deterministic and replayable (DESIGN 26.1). Fake workers run inline. The harness never
starts the async service — it exercises the cognition/reduction core, which is where the
invariants live.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from aca import ids
from aca.clock import ManualClock
from aca.config import Config
from aca.domain.enums import LifecycleState, WorkKind
from aca.domain.events import (
    DeliveryResult,
    EmbeddingResult,
    Event,
    HumanMessage,
    LLMResult,
    StochasticWake,
)
from aca.domain.runtime import AgentIdentity, RuntimeSession
from aca.domain.state import ConversationState, SelfModel
from aca.persistence.stores import Stores
from aca.reducer.context import ReducerContext
from aca.reducer.reducer import Reducer
from aca.rng import Rng
from aca.workers.fake_embedding import FakeEmbeddingWorker
from aca.workers.fake_llm import FakeLLMWorker


class Harness:
    def __init__(self, tmp_path, config: Config, clock: ManualClock, llm=None, embedding=None):
        self.stores = Stores.open(tmp_path / "agent.db")
        self.clock = clock
        self.config = config
        self.rng = Rng(config.rng_seed)
        self.llm = llm or FakeLLMWorker()
        self.embedding = embedding or FakeEmbeddingWorker()
        now = clock.now_utc()
        self.stores.identity.insert_identity(
            AgentIdentity(ids.new_id(ids.AGENT), now, LifecycleState.RUNNING, "sess_1")
        )
        self.stores.identity.insert_session(RuntimeSession("sess_1", "a", now, 1))
        self.stores.state.set_scheduler_generation(1)
        self.stores.state.initialize(
            SelfModel(config.temperament.initiative, config.temperament.inhibition,
                      config.temperament.persistence),
            ConversationState(),
        )
        self.ctx = ReducerContext(self.stores, clock, self.rng, config, "a", "sess_1", 1)
        self.reducer = Reducer(self.ctx)

    # --- driving -------------------------------------------------------------------------
    def send_human(self, text: str, *, event_id: str | None = None, channel: str = "cli"):
        event = HumanMessage(
            event_id=event_id or ids.new_id(ids.EVENT),
            timestamp=self.clock.now_utc(), source="cli", text=text, channel=channel,
        )
        with self.stores.db.transaction():
            self.stores.events.accept(event, self.clock.now_utc())
        return self.reducer.reduce(event)

    def wake(self, *, session_id: str = "sess_1", generation: int = 1):
        event = StochasticWake(
            event_id=ids.new_id(ids.EVENT), timestamp=self.clock.now_utc(), source="scheduler",
            runtime_session_id=session_id, scheduler_generation=generation,
        )
        return self.reducer.reduce(event)

    def reduce(self, event: Event):
        return self.reducer.reduce(event)

    def pending_work(self):
        return self.stores.work.pending()

    def run_work(self, work_id: str):
        """Run a single pending work item's worker and reduce the result."""
        work = self.stores.work.get_work(work_id)
        if work.kind is WorkKind.EMBEDDING:
            out = asyncio.run(self.embedding.embed(work.snapshot["text"]))
            event = EmbeddingResult(
                event_id=ids.new_id(ids.EVENT), timestamp=self.clock.now_utc(), source="w",
                work_id=work_id, provisional_memory_id=work.snapshot["provisional_memory_id"],
                embedding_id=ids.new_id(ids.EMBEDDING), model_version=out.model_version, vector=out.vector,
            )
        else:
            from aca.cognition.snapshot import Snapshot

            snap = Snapshot(
                cycle_id=work.cycle_id, work_id=work.work_id, basis_revision=work.basis_revision,
                template_version="v0.6", context=work.snapshot.get("context", {}),
            )
            out = asyncio.run(self.llm.run(snap))
            event = LLMResult(
                event_id=ids.new_id(ids.EVENT), timestamp=self.clock.now_utc(), source="w",
                work_id=work_id, cycle_id=work.cycle_id, basis_revision=work.basis_revision,
                result=out.result, tokens_in=out.tokens_in, tokens_out=out.tokens_out,
            )
        return self.reducer.reduce(event)

    def run_all_pending(self) -> int:
        """Run every currently-pending work item once; return how many ran."""
        pending = [w.work_id for w in self.stores.work.pending()]
        for work_id in pending:
            self.run_work(work_id)
        return len(pending)

    def deliver(self, message_id: str, *, delivered: bool = True, error: str | None = None):
        message = self.stores.outbox.get_message(message_id)
        event = DeliveryResult(
            event_id=ids.new_id(ids.EVENT), timestamp=self.clock.now_utc(), source="d",
            message_id=message_id, delivery_key=message.delivery_key, delivered=delivered, error=error,
        )
        return self.reducer.reduce(event)

    def close(self):
        self.stores.close()


@pytest.fixture
def clock():
    return ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))


@pytest.fixture
def config():
    return Config.from_mapping({"rng_seed": 1234})


@pytest.fixture
def harness(tmp_path, config, clock):
    h = Harness(tmp_path, config, clock)
    yield h
    h.close()


def make_harness(tmp_path, config, clock, **workers):
    return Harness(tmp_path, config, clock, **workers)
