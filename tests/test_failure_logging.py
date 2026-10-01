"""A failed event never stops the reducer loop, and failures are said in the logs (invariant 25).

Before this, one exception in a reduction ended the loop task: the agent answered nothing until a
restart, and nothing anywhere said why. Failed model attempts and failed replies were recorded only in
the database.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from aca.config import Config
from aca.domain.enums import ObligationStatus
from aca.domain.events import ReconcileEmbeddings
from aca.ipc.client import IpcClient
from aca.ipc.server import IpcServer
from aca.service.service import AgentService


class RaisingLLM:
    async def run(self, snapshot):
        raise RuntimeError("provider is down")


async def _started(tmp_path, **kwargs):
    service = AgentService(tmp_path, Config.from_mapping({"rng_seed": 7}), **kwargs)
    server = IpcServer(service, tmp_path / "aca.sock")
    await service.start()
    await server.start()
    return service, server


@pytest.mark.asyncio
async def test_a_failed_reduction_never_stops_the_loop(tmp_path, caplog):
    # Bounded: before the fix the dead loop hung everything after it, shutdown included.
    await asyncio.wait_for(_failed_reduction_then_a_reply(tmp_path, caplog), timeout=15)


async def _failed_reduction_then_a_reply(tmp_path, caplog):
    service, server = await _started(tmp_path)
    reduce = service._reducer.reduce

    def failing_once(event):
        if isinstance(event, ReconcileEmbeddings):
            raise RuntimeError("boom")
        return reduce(event)

    service._reducer.reduce = failing_once
    try:
        await service.reconcile_embeddings()
        client = IpcClient(tmp_path / "aca.sock")
        received: list[str] = []
        got = asyncio.Event()

        async def on_msg(frame):
            received.append(frame["text"])
            got.set()

        subscription = asyncio.ensure_future(client.subscribe(on_msg))
        await asyncio.sleep(0.05)
        await client.chat_send("Explain this stack trace.")
        await asyncio.wait_for(got.wait(), timeout=3)  # the loop is still answering
        assert received
        assert "reducing ReconcileEmbeddings failed; the loop continues" in caplog.text
        subscription.cancel()
    finally:
        await server.close()
        await service.stop()


@pytest.mark.asyncio
async def test_failed_attempts_and_a_failed_reply_are_logged(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="aca")
    service, server = await _started(tmp_path, llm_worker=RaisingLLM())
    try:
        await IpcClient(tmp_path / "aca.sock").chat_send("Explain this stack trace.")
        deadline = asyncio.get_running_loop().time() + 5
        while asyncio.get_running_loop().time() < deadline:
            rows = service.stores.db.query_all("SELECT status FROM response_obligations")
            if rows and all(r["status"] == ObligationStatus.FAILED.value for r in rows):
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)
        text = caplog.text
        assert "attempt 1/3 failed" in text and "RuntimeError: provider is down" in text
        assert "failed for good" in text
        assert "reply failed" in text and "worker_error" in text
        assert "Explain this stack trace" not in text  # ids and codes only, never message text
    finally:
        await server.close()
        await service.stop()
