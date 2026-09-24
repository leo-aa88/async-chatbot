"""Adversarial: two overlapping pump() passes never deliver the same item twice (invariant 38).

The reducer loop and ``notify_client_connected`` can both run ``DeliveryPump.pump()`` at the same
time. Each pass built its pending list once and then awaited every transport attempt, so while
pass 1 was suspended on message A, pass 2 could send message B, and when pass 1 resumed it sent B
again from its stale list. The same interleaving let two proactive items through one burst window,
because the window is only stamped after an attempt returns. Duplicate delivery is a correctness
failure in the frozen design, not something to leave to client-side dedup by ``delivery_key``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta

from conftest import Harness

from aca import ids
from aca.domain.enums import OutboundKind, OutboundStatus
from aca.domain.runtime import OutboundMessage
from aca.service.delivery import DeliveryPump


def _pending(h: Harness, kind: OutboundKind, channel: str) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.outbox.insert_message(OutboundMessage(
            message_id=ids.new_id(ids.OUTBOUND), delivery_key=ids.new_delivery_key(),
            action_id="a", kind=kind, channel=channel, payload=f"msg {channel}",
            status=OutboundStatus.PENDING_DELIVERY, created_at=now,
            expires_at=now + timedelta(hours=1) if kind is OutboundKind.PROACTIVE else None,
        ))


def _pump(h: Harness) -> tuple[DeliveryPump, list[str]]:
    sent: list[str] = []

    async def slow_sink(channel, payload, delivery_key, message_id):
        sent.append(message_id)
        await asyncio.sleep(0)  # transport yields, as a real socket write does
        return True

    return DeliveryPump(h.stores, h.clock, h.config, h.reducer.context, slow_sink, lambda _e: None), sent


async def test_overlapping_pumps_send_each_mandatory_message_once(harness: Harness):
    _pending(harness, OutboundKind.MANDATORY, "chan-a")
    _pending(harness, OutboundKind.MANDATORY, "chan-b")
    pump, sent = _pump(harness)

    await asyncio.gather(pump.pump(), pump.pump())

    assert sorted(Counter(sent).values()) == [1, 1]


async def test_overlapping_pumps_respect_the_proactive_burst_window(harness: Harness):
    _pending(harness, OutboundKind.PROACTIVE, "chan-a")
    _pending(harness, OutboundKind.PROACTIVE, "chan-b")
    pump, sent = _pump(harness)

    await asyncio.gather(pump.pump(), pump.pump())

    assert len(sent) == 1  # one proactive per burst window, however the passes interleave
