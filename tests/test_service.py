"""Async integration test for the resident daemon over the real IPC socket."""

from __future__ import annotations

import asyncio

import pytest

from aca.config import Config
from aca.domain.enums import ObligationStatus
from aca.ipc.client import IpcClient
from aca.ipc.server import IpcServer
from aca.service.service import AgentService


class RaisingLLM:
    """A worker whose provider always errors (timeout/500/auth) — never returns a result."""

    async def run(self, snapshot):
        raise RuntimeError("provider is down")


@pytest.mark.asyncio
async def test_daemon_answers_task_and_dedupes_retry(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    server = IpcServer(service, tmp_path / "aca.sock")
    await service.start()
    await server.start()
    try:
        client = IpcClient(tmp_path / "aca.sock")
        received: list[str] = []
        got = asyncio.Event()

        async def on_msg(frame):
            received.append(frame["text"])
            got.set()

        subscription = asyncio.ensure_future(client.subscribe(on_msg))
        await asyncio.sleep(0.05)

        ack = await client.chat_send("Explain this stack trace.")
        assert ack["accepted"] is True

        retry = await client.chat_send("Explain this stack trace.", event_id=ack["event_id"])
        assert retry["accepted"] is False  # idempotent ingress

        await asyncio.wait_for(got.wait(), timeout=3)
        assert received and received[0].startswith("Here's what I can say")

        status = await client.status()
        assert status["pending_obligations"] == 0
        assert status["lifecycle_state"] == "RUNNING"

        subscription.cancel()
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_mandatory_obligation_surfaced_as_failed_when_worker_keeps_erroring(tmp_path):
    # Review finding #3: a live worker error must not requeue forever. After bounded retries the
    # obligation is surfaced as FAILED — never left silently pending (invariant 25).
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}), llm_worker=RaisingLLM())
    server = IpcServer(service, tmp_path / "aca.sock")
    await service.start()
    await server.start()
    try:
        client = IpcClient(tmp_path / "aca.sock")
        await client.chat_send("Explain this stack trace.")

        # Poll until the obligation resolves (bounded retries + backoff take a fraction of a second).
        async def obligations():
            return service.stores.db.query_all("SELECT status FROM response_obligations")

        deadline = asyncio.get_running_loop().time() + 5
        rows = []
        while asyncio.get_running_loop().time() < deadline:
            rows = await obligations()
            if rows and all(r["status"] != ObligationStatus.PENDING.value for r in rows):
                break
            await asyncio.sleep(0.1)

        assert rows, "a mandatory obligation should have been created"
        assert all(r["status"] == ObligationStatus.FAILED.value for r in rows)
        assert service.stores.work.pending_obligations() == []  # not silently pending
        agent_turns = [t for t in service.stores.outbox.recent_turns() if t["role"] == "agent"]
        assert agent_turns == []  # never faked a reply
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_proactive_message_reaches_connected_idle_client_without_reconnect(tmp_path):
    # The whole point of the project: a connected, idle client receives an unsolicited message,
    # no reconnect required. A short active_within lets the conversation leave ACTIVE quickly so
    # proactive initiative is permitted (DESIGN 7.4, 14.3).
    cfg = Config.from_mapping({
        "rng_seed": 5,
        "cognition": {"spontaneous_activation_rate_per_hour": 100000, "semantic_worthiness_floor": 0.2,
                      "selection_temperature": 0.3, "null_candidate_score": -5.0},
        "conversation": {"active_within": "1s", "idle_within": "12h"},
        "timing": {"proactive_cooldown": "0s", "proactive_ttl": "1h", "proactive_burst_window": "0s"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })
    service = AgentService(tmp_path, cfg)
    server = IpcServer(service, tmp_path / "aca.sock")
    await service.start()
    await server.start()
    try:
        client = IpcClient(tmp_path / "aca.sock")
        subscription = asyncio.ensure_future(client.subscribe(lambda f: asyncio.sleep(0)))
        await asyncio.sleep(0.05)
        # One substantive turn gives the agent something to resurface; then we stay idle+connected.
        await client.chat_send("I am fascinated by fluid mechanics and hope to return to it")

        async def wait_for_delivered_proactive():
            while True:
                row = service.stores.db.query_one(
                    "SELECT COUNT(*) AS n FROM outbound_messages "
                    "WHERE kind='PROACTIVE' AND status='DELIVERED'"
                )
                if row["n"] >= 1:
                    return
                await asyncio.sleep(0.05)

        # No reconnect anywhere in here — the same connection must receive the proactive item.
        await asyncio.wait_for(wait_for_delivered_proactive(), timeout=8)
        subscription.cancel()
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_closing_client_does_not_stop_agent(tmp_path):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}))
    server = IpcServer(service, tmp_path / "aca.sock")
    await service.start()
    await server.start()
    try:
        client = IpcClient(tmp_path / "aca.sock")
        await client.chat_send("hello there")  # one-shot connection, then closed
        # Agent remains running after the client connection closed (invariant 36).
        status = await client.status()
        assert status["lifecycle_state"] == "RUNNING"
    finally:
        await server.close()
        await service.stop()
