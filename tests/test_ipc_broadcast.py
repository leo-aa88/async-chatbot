"""Regression: a subscriber set that changes mid-broadcast must not abort the broadcast.

``_broadcast`` awaits each subscriber's write. While it is suspended, another connection can
subscribe, or ``_handle``'s ``finally`` can drop a closing one. Iterating the live set then raised
"Set changed size during iteration". The delivery pump reported that as a transport failure, even
though some subscribers had already received the message, so a mandatory message would be retried
and delivered to them twice.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aca.clock import ManualClock
from aca.ipc.server import IpcServer


class _Writer:
    """Minimal StreamWriter stand-in: records frames; drain yields and runs an optional hook."""

    def __init__(self, on_drain=None) -> None:
        self.frames: list[bytes] = []
        self._on_drain = on_drain

    def write(self, data: bytes) -> None:
        self.frames.append(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)
        if self._on_drain is not None:
            hook, self._on_drain = self._on_drain, None  # fire once
            hook()


class _FailingWriter(_Writer):
    """A subscriber whose socket fails on drain with the given error."""

    def __init__(self, error: OSError) -> None:
        super().__init__()
        self._error = error

    async def drain(self) -> None:
        await asyncio.sleep(0)
        raise self._error


async def test_subscriber_joining_mid_broadcast_does_not_abort_it(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    server = IpcServer(SimpleNamespace(clock=clock), tmp_path / "aca.sock")
    newcomer = _Writer()

    def a_client_subscribes() -> None:
        server._subscribers.add(newcomer)

    first = _Writer(on_drain=a_client_subscribes)
    second = _Writer(on_drain=a_client_subscribes)
    server._subscribers.update({first, second})

    delivered = await server._broadcast("cli", "hello", "dk_1", "msg_1")

    assert delivered is True
    assert len(first.frames) == 1 and len(second.frames) == 1
    # Snapshot semantics: a mid-broadcast joiner doesn't get the in-flight frame. Its own
    # subscribe-triggered pump handles delivery to it.
    assert newcomer.frames == []


async def test_subscriber_leaving_mid_broadcast_does_not_abort_it(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    server = IpcServer(SimpleNamespace(clock=clock), tmp_path / "aca.sock")
    writers: list[_Writer] = []

    def other_client_disconnects(me: int):
        return lambda: server._subscribers.discard(writers[1 - me])

    writers.append(_Writer(on_drain=other_client_disconnects(0)))
    writers.append(_Writer(on_drain=other_client_disconnects(1)))
    server._subscribers.update(writers)

    delivered = await server._broadcast("cli", "hello", "dk_1", "msg_1")

    assert delivered is True


@pytest.mark.parametrize("error", [ConnectionAbortedError(), ConnectionRefusedError(), OSError(113, "No route")])
async def test_one_subscriber_socket_error_does_not_fail_the_broadcast(tmp_path, error):
    # Any socket error from one subscriber used to escape unless it was a reset or broken pipe. The
    # pump then reported a transport failure although earlier subscribers had the frame, and the
    # mandatory retry delivered it to them twice.
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    server = IpcServer(SimpleNamespace(clock=clock), tmp_path / "aca.sock")
    healthy, broken = _Writer(), _FailingWriter(error)
    server._subscribers.update({healthy, broken})

    delivered = await server._broadcast("cli", "hello", "dk_1", "msg_1")

    assert delivered is True
    assert len(healthy.frames) == 1
    assert server._subscribers == {healthy}  # the dead writer is dropped


async def test_broadcast_reports_undelivered_when_every_subscriber_fails(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    server = IpcServer(SimpleNamespace(clock=clock), tmp_path / "aca.sock")
    server._subscribers.add(_FailingWriter(ConnectionAbortedError()))

    delivered = await server._broadcast("cli", "hello", "dk_1", "msg_1")

    assert delivered is False
    assert server._subscribers == set()
