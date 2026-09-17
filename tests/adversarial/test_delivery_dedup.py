"""Adversarial: outbound delivery is idempotency-aware and deciding-to-speak != delivered.

Covers invariants 6, 37, 38: an undelivered item is not a visible turn; a duplicate
DeliveryResult is a no-op; a failed mandatory delivery stays pending (never silently lost).
"""

from __future__ import annotations

from datetime import timedelta

from conftest import Harness

from aca import ids
from aca.domain.enums import OutboundKind, OutboundStatus
from aca.domain.runtime import OutboundMessage


def _produce_mandatory(h: Harness):
    h.send_human("Explain this stack trace.")
    h.run_all_pending()
    return h.stores.outbox.deliverable()[0]


def test_decided_but_undelivered_is_not_a_visible_turn(harness: Harness):
    message = _produce_mandatory(harness)
    assert message.status is OutboundStatus.PENDING_DELIVERY
    agent_turns = [t for t in harness.stores.outbox.recent_turns() if t["role"] == "agent"]
    assert agent_turns == []  # not visible until delivered (invariant 37)


def test_duplicate_delivery_result_is_idempotent(harness: Harness):
    message = _produce_mandatory(harness)
    harness.deliver(message.message_id, delivered=True)
    harness.deliver(message.message_id, delivered=True)  # duplicate transport result
    agent_turns = [t for t in harness.stores.outbox.recent_turns() if t["role"] == "agent"]
    assert len(agent_turns) == 1  # exactly one user-visible turn (invariant 38)
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.DELIVERED


def test_failed_mandatory_delivery_stays_pending(harness: Harness):
    message = _produce_mandatory(harness)
    harness.deliver(message.message_id, delivered=False, error="no_client")
    # Mandatory items are never silently expired; they remain deliverable (invariant 25).
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.PENDING_DELIVERY
    assert any(m.message_id == message.message_id for m in harness.stores.outbox.deliverable())


def _pending_proactive(h: Harness) -> OutboundMessage:
    now = h.clock.now_utc()
    message = OutboundMessage(
        message_id=ids.new_id(ids.OUTBOUND), delivery_key=ids.new_delivery_key(), action_id="a",
        kind=OutboundKind.PROACTIVE, channel="cli", payload="an autonomous thought",
        status=OutboundStatus.PENDING_DELIVERY, created_at=now, expires_at=now + timedelta(hours=1),
    )
    with h.stores.db.transaction():
        h.stores.outbox.insert_message(message)
    return message


def test_proactive_with_no_client_is_held_then_delivered(harness: Harness):
    # Client absence is operational state, not failure (DESIGN 29.7). A proactive item generated
    # while no client is connected must be HELD (pending), not FAILED, so it delivers on reconnect.
    message = _pending_proactive(harness)
    harness.deliver(message.message_id, delivered=False, error="no_client")
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.PENDING_DELIVERY
    assert any(m.message_id == message.message_id for m in harness.stores.outbox.deliverable())

    # On reconnect the held item is delivered and becomes a visible agent turn.
    harness.deliver(message.message_id, delivered=True)
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.DELIVERED
    assert any(t["role"] == "agent" for t in harness.stores.outbox.recent_turns())


def test_proactive_transport_error_still_fails(harness: Harness):
    # A genuine transport error (not mere client absence) does not linger.
    message = _pending_proactive(harness)
    harness.deliver(message.message_id, delivered=False, error="transport:BrokenPipe")
    assert harness.stores.outbox.get_message(message.message_id).status is OutboundStatus.FAILED
