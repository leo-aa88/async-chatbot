"""IPC client used by the CLI (DESIGN 28.2).

Two access patterns: a one-shot request/response (status, chat send, logs...) and a long-lived
subscription that streams delivered agent messages. The client generates a stable ``event_id``
per human message so a retry after a lost ACK reuses it (invariant 14) and deduplicates pushed
messages by ``delivery_key`` (invariant 38).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .. import ids
from ..errors import IpcError
from . import protocol as p


class IpcClient:
    def __init__(self, socket_path: str | Path) -> None:
        self._socket_path = str(socket_path)

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        try:
            return await asyncio.open_unix_connection(self._socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise IpcError(f"agent service not reachable at {self._socket_path}") from exc

    async def request_once(self, op: str, **payload: Any) -> dict[str, Any]:
        reader, writer = await self._open()
        try:
            await p.write_message(writer, {"op": op, **payload})
            response = await p.read_message(reader)
            if response is None:
                raise IpcError("service closed connection without responding")
            return response
        finally:
            writer.close()

    async def chat_send(
        self, text: str, *, event_id: str | None = None, channel: str = "cli", input_mode: str = "text"
    ) -> dict:
        return await self.request_once(
            p.OP_CHAT, text=text, channel=channel, input_mode=input_mode,
            event_id=event_id or ids.new_id(ids.EVENT),
        )

    async def status(self) -> dict:
        return await self.request_once(p.OP_STATUS)

    async def memories(self) -> dict:
        return await self.request_once(p.OP_MEMORIES)

    async def topics(self) -> dict:
        return await self.request_once(p.OP_TOPICS)

    async def logs(self) -> dict:
        return await self.request_once(p.OP_LOGS)

    async def metrics(self) -> dict:
        return await self.request_once(p.OP_METRICS)

    async def shutdown(self) -> dict:
        return await self.request_once(p.OP_SHUTDOWN)

    async def reconcile_embeddings(self) -> dict:
        return await self.request_once(p.OP_RECONCILE)

    async def subscribe(self, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Open a subscription and stream delivered messages to ``handler`` until closed."""
        reader, writer = await self._open()
        seen_keys: set[str] = set()
        try:
            await p.write_message(writer, {"op": p.OP_SUBSCRIBE})
            while True:
                frame = await p.read_message(reader)
                if frame is None:
                    return
                if frame.get("kind") == p.PUSH_MESSAGE:
                    key = frame.get("delivery_key", "")
                    if key in seen_keys:
                        continue  # duplicate transport suppressed (invariant 38)
                    seen_keys.add(key)
                    await handler(frame)
        finally:
            writer.close()
