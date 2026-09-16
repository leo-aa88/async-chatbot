"""Adversarial: reconnect never dogpiles proactive messages (invariant 38, DESIGN 23.7 rule 5).

Covers the review's finding #6: the ``proactive_burst_window_seconds`` knob is actually wired.
Two channels each holding a pending proactive item, pumped back-to-back before either
DeliveryResult is reduced (so the self-model cooldown hasn't updated yet), must yield at most one
delivery within the burst window.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import Harness

from aca import ids
from aca.domain.enums import OutboundKind, OutboundStatus
from aca.domain.runtime import OutboundMessage
from aca.service.delivery import DeliveryPump


def _pending_proactive(h: Harness, channel: str) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.outbox.insert_message(OutboundMessage(
            message_id=ids.new_id(ids.OUTBOUND), delivery_key=ids.new_delivery_key(),
            action_id="a", kind=OutboundKind.PROACTIVE, channel=channel, payload=f"thought {channel}",
            status=OutboundStatus.PENDING_DELIVERY, created_at=now,
            expires_at=now + timedelta(hours=1),
        ))


def _pump(h: Harness):
    delivered: list[str] = []
    events: list = []

    async def sink(channel, payload, delivery_key, message_id):
        delivered.append(message_id)
        return True

    pump = DeliveryPump(h.stores, h.clock, h.config, h.reducer.context, sink, events.append)
    return pump, delivered


@pytest.mark.asyncio
async def test_two_channels_do_not_dogpile_within_burst_window(harness: Harness):
    _pending_proactive(harness, "chan-a")
    _pending_proactive(harness, "chan-b")
    pump, delivered = _pump(harness)

    # Two reconnects back-to-back; DeliveryResults are NOT reduced in between (the racy case).
    await pump.pump()
    await pump.pump()  # monotonic clock has not advanced -> still inside the burst window

    assert len(delivered) == 1  # exactly one proactive delivered; the other stays pending


@pytest.mark.asyncio
async def test_delivery_resumes_after_burst_window_elapses(harness: Harness):
    _pending_proactive(harness, "chan-a")
    _pending_proactive(harness, "chan-b")
    pump, delivered = _pump(harness)

    await pump.pump()
    # Advance past the burst window (default 5m); the second item may now go.
    harness.clock.advance(harness.config.timing.proactive_burst_window_seconds + 1)
    await pump.pump()

    assert len(delivered) == 2
