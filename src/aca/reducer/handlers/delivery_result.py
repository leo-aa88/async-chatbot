"""DeliveryResult handler (DESIGN 23.7, 29.7).

Delivery is a separately authorized side effect; the adapter reports the outcome here and the
reducer records it. Only on success does the outbound item become a user-visible conversation
turn (invariant 37) and does ``delivered_at`` (the anchor for interruption cooldown/refractory,
invariant 6) get set. Idempotent by ``message_id``: a duplicate result after DELIVERED is a
no-op (invariant 38).
"""

from __future__ import annotations

from dataclasses import replace

from ...domain.enums import OutboundKind, OutboundStatus
from ...domain.events import DeliveryResult
from ..context import ReducerContext
from .base import HandlerOutcome


def handle_delivery_result(ctx: ReducerContext, event: DeliveryResult) -> HandlerOutcome:
    message = ctx.stores.outbox.get_message(event.message_id)
    if message is None:
        return HandlerOutcome(note="unknown_message", committed=False)
    if message.status is OutboundStatus.DELIVERED:
        return HandlerOutcome(note="duplicate_delivery", committed=False)  # idempotent

    now = event.timestamp
    ctx.stores.outbox.record_attempt(
        event.message_id, event.delivery_key, now, event.delivered, event.error
    )

    if not event.delivered:
        return _handle_undelivered(ctx, event, message)

    ctx.stores.outbox.set_status(event.message_id, OutboundStatus.DELIVERED, delivered_at=now)
    ctx.stores.outbox.add_agent_turn(message.payload, message.channel, message.message_id, now)

    conversation = ctx.stores.state.load_conversation()
    ctx.stores.state.save_conversation(replace(conversation, last_agent_delivered_at=now))

    if message.kind is OutboundKind.PROACTIVE:
        model = ctx.stores.state.load_self_model()
        ctx.stores.state.save_self_model(model.with_delivered_proactive(now))

    return HandlerOutcome(note="delivered")


# Delivery-adapter dispositions carried in DeliveryResult.error for a non-delivered proactive
# item: these are drops (never late delivery, DESIGN 23.7), distinct from a transport failure.
_PROACTIVE_DROP_PREFIXES = ("expired", "revalidation", "superseded", "coalesced")
# No client was connected to receive it. Client absence is operational state, not failure or
# conversational silence (DESIGN 29.7): the item stays pending and is delivered on reconnect
# (subject to TTL revalidation), rather than being lost.
_NO_CLIENT = "no_client"


def _handle_undelivered(ctx, event, message) -> HandlerOutcome:
    error = event.error or ""
    if message.kind is OutboundKind.PROACTIVE:
        if any(error.startswith(prefix) for prefix in _PROACTIVE_DROP_PREFIXES):
            status = (
                OutboundStatus.SUPERSEDED
                if error.startswith(("superseded", "coalesced"))
                else OutboundStatus.EXPIRED
            )
            ctx.stores.outbox.set_status(event.message_id, status, error=error)
            return HandlerOutcome(note=f"proactive_{status.value.lower()}")
        if error == _NO_CLIENT:
            # Held for reconnect, not failed — this is why an idle client eventually receives it.
            ctx.stores.outbox.set_status(event.message_id, OutboundStatus.PENDING_DELIVERY, error=error)
            return HandlerOutcome(note="proactive_held_no_client")
        # Genuine transport failure: do not linger or dogpile later.
        ctx.stores.outbox.set_status(event.message_id, OutboundStatus.FAILED, error=error)
        return HandlerOutcome(note="proactive_delivery_failed")
    # Mandatory items stay pending for retry; they are never silently expired (invariant 25).
    ctx.stores.outbox.set_status(event.message_id, OutboundStatus.PENDING_DELIVERY, error=error)
    return HandlerOutcome(note="mandatory_delivery_retry")
