"""Newline-delimited JSON framing for IPC (DESIGN 28.2).

One JSON object per line. Requests carry an ``op``; responses carry ``ok`` plus data. Kept
deliberately tiny — the transport is an adapter, not cognition.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..errors import IpcError

# Client -> service operations.
OP_CHAT = "chat"            # {op, text, channel}
OP_STATUS = "status"
OP_MEMORIES = "memories"
OP_TOPICS = "topics"
OP_LOGS = "logs"
OP_METRICS = "metrics"
OP_SUBSCRIBE = "subscribe"  # register this connection to receive delivered agent messages
OP_SHUTDOWN = "shutdown"

# Service -> client push message kinds.
PUSH_MESSAGE = "message"    # {kind, text, channel, delivery_key, message_id}


def encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


async def read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one framed message, or ``None`` at clean EOF."""
    line = await reader.readline()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IpcError(f"malformed IPC frame: {exc}") from exc


async def write_message(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    writer.write(encode(obj))
    await writer.drain()


def ok(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, **(data or {})}


def err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}
