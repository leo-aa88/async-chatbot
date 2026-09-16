"""Durable outbound-intent creation (DESIGN 6.1, 23.7).

Creating an outbound item persists an ``Action`` (the decision) and an ``OutboundMessage`` in
``PENDING_DELIVERY`` — a decision to speak, not yet a delivered turn (invariant 37). Proactive
items carry a bounded TTL and supersede any earlier still-pending proactive item on the same
channel so at most one remains eligible (DESIGN 6.1).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import ids
from ..domain.enums import ActionKind, OutboundKind, OutboundStatus
from ..domain.runtime import Action, OutboundMessage
from .context import ReducerContext


def _supersede_pending_proactive(ctx: ReducerContext, channel: str, new_message_id: str) -> None:
    for existing in ctx.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE):
        if existing.channel == channel and existing.message_id != new_message_id:
            ctx.stores.outbox.set_status(
                existing.message_id,
                OutboundStatus.SUPERSEDED,
                superseded_by_id=new_message_id,
            )


def create_outbound(
    ctx: ReducerContext,
    *,
    kind: OutboundKind,
    channel: str,
    text: str,
    cycle_id: str | None,
    source_event_id: str | None,
    now: datetime,
) -> str:
    """Persist a SPEAK action + outbound item; return the new ``message_id``."""
    action = Action(
        id=ids.new_id(ids.ACTION),
        kind=ActionKind.SPEAK,
        created_at=now,
        cycle_id=cycle_id,
        source_event_id=source_event_id,
        detail=kind.value,
    )
    ctx.stores.work.insert_action(action)

    message_id = ids.new_id(ids.OUTBOUND)
    expires_at: datetime | None = None
    if kind is OutboundKind.PROACTIVE:
        expires_at = now + timedelta(seconds=ctx.config.timing.proactive_ttl_seconds)

    message = OutboundMessage(
        message_id=message_id,
        delivery_key=ids.new_delivery_key(),
        action_id=action.id,
        kind=kind,
        channel=channel,
        payload=text,
        status=OutboundStatus.PENDING_DELIVERY,
        created_at=now,
        expires_at=expires_at,
    )
    ctx.stores.outbox.insert_message(message)

    if kind is OutboundKind.PROACTIVE:
        _supersede_pending_proactive(ctx, channel, message_id)

    return message_id
