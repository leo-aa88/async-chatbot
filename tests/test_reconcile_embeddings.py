"""Topic-embedding backfill (reconcile): enqueue summary embeddings for topics missing one.

Topics created before the summary-embedding pipeline existed have no vector, so semantic
dedup/dominance/continuity can't see them. ReconcileEmbeddings backfills them idempotently through
the normal embedding path, and each landing fires the post-embedding merge.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from conftest import Harness

from aca import ids
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import WorkKind, WorkStatus
from aca.domain.events import ReconcileEmbeddings
from aca.domain.runtime import WorkItem
from aca.domain.state import Topic
from aca.persistence.stores import Stores
from aca.service.service import AgentService
from aca.workers.base import EmbeddingOutput

_RATE = half_life_to_rate_per_hour(24.0)


class _ConstantEmbeddingWorker:
    """One fixed vector for any text, so distinct-wording paraphrases collide at cosine 1.0."""

    model_version = "const-v1"

    async def embed(self, text: str) -> EmbeddingOutput:
        return EmbeddingOutput(vector=[1.0, 0.0, 0.0], model_version="const-v1")


def _topic(h: Harness, tid: str, summary: str, *, embedded: bool = False) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id=tid, summary=summary, activation=0.8, importance=0.8, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, source_memory_id=None,
        ))
        if embedded:
            h.stores.memory.insert_topic_embedding(tid, "const-v1", [0.0, 1.0, 0.0], now)


def _reconcile(h: Harness) -> None:
    h.reduce(ReconcileEmbeddings(
        event_id=ids.new_id(ids.EVENT), timestamp=h.clock.now_utc(), source="test",
    ))


def _pending_topic_embed_ids(h: Harness) -> list[str]:
    return [
        w.snapshot["topic_id"]
        for w in h.pending_work()
        if w.kind is WorkKind.EMBEDDING and w.snapshot.get("topic_id")
    ]


def test_reconcile_enqueues_only_topics_missing_an_embedding(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t_has", "Already embedded topic", embedded=True)
    _topic(h, "t_missing", "Topic with no summary vector yet")
    _reconcile(h)
    assert _pending_topic_embed_ids(h) == ["t_missing"]  # embedded topic is skipped
    h.close()


def test_reconcile_skips_a_pending_backfill_job(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t_missing", "Topic with no summary vector yet")
    _reconcile(h)
    assert _pending_topic_embed_ids(h) == ["t_missing"]
    _reconcile(h)  # still PENDING, not yet landed
    assert _pending_topic_embed_ids(h) == ["t_missing"]  # not enqueued twice
    h.close()


def test_reconcile_skips_a_leased_running_backfill_job(tmp_path, clock):
    # The real idempotency case the pending()-only guard missed: the first pass's job is dispatched
    # (leased to RUNNING) before the next reconcile — a restart mid-backfill, or the on-demand CLI
    # path racing running jobs. It must not enqueue a duplicate.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    _topic(h, "t_missing", "Topic with no summary vector yet")
    _reconcile(h)
    (job,) = [w for w in h.pending_work() if w.kind is WorkKind.EMBEDDING]
    h.stores.work.lease(job.work_id, h.clock.now_utc() + timedelta(minutes=5))  # PENDING -> RUNNING
    assert _pending_topic_embed_ids(h) == []  # nothing pending; the job is in flight

    _reconcile(h)  # must skip the RUNNING job, not enqueue a second one
    assert _pending_topic_embed_ids(h) == []
    assert h.stores.work.active_embedding_topic_ids() == {"t_missing"}  # exactly one in flight
    h.close()


def test_reconcile_backfill_drives_a_paraphrase_merge(tmp_path, clock):
    # End-to-end: two pre-existing paraphrase topics with no vectors -> reconcile enqueues both ->
    # draining them (constant embedder) lands identical vectors -> the second merges into the first.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock,
                embedding=_ConstantEmbeddingWorker())
    _topic(h, "t1", "Plan to put the runtime on a robot")
    _topic(h, "t2", "Deploy the agent onto physical hardware")
    _reconcile(h)
    assert sorted(_pending_topic_embed_ids(h)) == ["t1", "t2"]

    h.run_all_pending()  # both embeddings land; the second triggers the post-embedding merge
    topics = h.stores.memory.all_topics()
    assert len(topics) == 1
    assert topics[0].evidence_count == 2
    h.close()


@pytest.mark.asyncio
async def test_service_startup_backfills_pre_existing_topics(tmp_path):
    # The production scenario: topics created before the pipeline sit with no vector. Starting the
    # service must backfill them without any user interaction.
    now = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    stores = Stores.open(tmp_path / "agent.db")
    with stores.db.transaction():
        for i in range(2):
            stores.memory.insert_topic(Topic(
                id=f"pre_{i}", summary=f"a pre-existing topic number {i}", activation=0.8,
                importance=0.8, decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
                source_memory_id=None,
            ))
    assert stores.memory.topics_without_embedding() != []  # start from an unembedded backlog
    stores.close()

    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))  # real FakeEmbeddingWorker
    await service.start()
    try:
        for _ in range(200):  # let the dispatched backfill jobs land
            if len(service.stores.memory.topics_without_embedding()) == 0:
                break
            await asyncio.sleep(0.02)
        assert service.stores.memory.topics_without_embedding() == []  # all backfilled at startup
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_startup_reconcile_skips_a_recovered_running_job(tmp_path):
    # Crash mid-backfill: a topic-embedding job was left RUNNING (leased, not expired). On restart,
    # the startup reconcile must not enqueue a second job for that topic.
    now = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    stores = Stores.open(tmp_path / "agent.db")
    with stores.db.transaction():
        stores.memory.insert_topic(Topic(
            id="t_running", summary="a topic whose embedding job is mid-flight", activation=0.8,
            importance=0.8, decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
            source_memory_id=None,
        ))
        stores.work.insert_work(WorkItem(
            work_id="w_running", kind=WorkKind.EMBEDDING, cycle_id="c", basis_revision=0,
            source_event_id="e", status=WorkStatus.RUNNING, created_at=now,
            lease_until=now + timedelta(hours=1),  # future lease -> not reclaimed as expired
            snapshot={"topic_id": "t_running", "text": "a topic whose embedding job is mid-flight"},
        ))
    stores.close()

    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    await service.start()
    try:
        await asyncio.sleep(0.05)  # let startup settle
        rows = service.stores.db.query_all(
            "SELECT COUNT(*) AS n FROM work_items WHERE kind='EMBEDDING' AND snapshot LIKE ?",
            ("%t_running%",),
        )
        assert rows[0]["n"] == 1  # the recovered job only; reconcile added no duplicate
    finally:
        await service.stop()
