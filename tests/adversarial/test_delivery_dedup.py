"""Adversarial: outbound delivery is idempotency-aware and deciding-to-speak != delivered.

Covers invariants 6, 37, 38: an undelivered item is not a visible turn; a duplicate
DeliveryResult is a no-op; a failed mandatory delivery stays pending (never silently lost).
"""

from __future__ import annotations

from aca.domain.enums import OutboundStatus
from conftest import Harness


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
