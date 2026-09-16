"""Outbound delivery pump (DESIGN 6.1, 23.7, 29.7).

The pump is the delivery adapter's driver. It performs only authorized transport and reports
outcomes back as ``DeliveryResult`` events (invariant 4); it never writes outbound state
directly — the reducer owns every status transition. Rules enforced here (DESIGN 23.7):

* mandatory items are considered before proactive on (re)connect;
* a proactive item past its TTL, or failing delivery-time revalidation, is *dropped* (reported
  as ``expired``/``revalidation``), never delivered late;
* at most one proactive item is delivered per burst window — no dogpile on reconnect.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .. import ids
from ..clock import Clock
from ..config import Config
from ..domain.enums import OutboundKind
from ..domain.events import DeliveryResult
from ..persistence.stores import Stores
from ..reducer.context import ReducerContext
from ..reducer.gates import evaluate_proactive

# A sink attempts transport and returns True iff a client on the channel accepted it.
ClientSink = Callable[[str, str, str, str], Awaitable[bool]]


class DeliveryPump:
    def __init__(
        self,
        stores: Stores,
        clock: Clock,
        config: Config,
        reducer_ctx: ReducerContext,
        sink: ClientSink,
        enqueue: Callable[[DeliveryResult], None],
    ) -> None:
        self._stores = stores
        self._clock = clock
        self._config = config
        self._ctx = reducer_ctx
        self._sink = sink
        self._enqueue = enqueue
        self._inflight: set[str] = set()

    def on_delivery_result(self, message_id: str) -> None:
        self._inflight.discard(message_id)

    async def pump(self) -> None:
        """Attempt one delivery pass. Mandatory first, then at most one proactive item."""
        now = self._clock.now_utc()
        pending = [m for m in self._stores.outbox.deliverable() if m.message_id not in self._inflight]
        mandatory = [m for m in pending if m.kind is OutboundKind.MANDATORY]
        proactive = [m for m in pending if m.kind is OutboundKind.PROACTIVE]

        for message in mandatory:
            await self._attempt(message, now)

        await self._pump_one_proactive(proactive, now)

    async def _pump_one_proactive(self, proactive, now) -> None:
        delivered_one = False
        for message in proactive:
            if message.expires_at is not None and message.expires_at <= now:
                self._report(message, delivered=False, error="expired:ttl")
                continue
            gate = evaluate_proactive(self._ctx, now)
            if not gate.allowed:
                self._report(message, delivered=False, error=f"revalidation:{gate.reason}")
                continue
            if delivered_one:
                # Burst window: keep the rest pending rather than dogpiling (DESIGN 23.7).
                continue
            await self._attempt(message, now)
            delivered_one = True

    async def _attempt(self, message, now) -> None:
        self._inflight.add(message.message_id)
        try:
            ok = await self._sink(
                message.channel, message.payload, message.delivery_key, message.message_id
            )
        except Exception as exc:  # transport error -> reported failure, retryable if mandatory
            self._report(message, delivered=False, error=f"transport:{exc!r}")
            return
        self._report(message, delivered=ok, error=None if ok else "no_client")

    def _report(self, message, *, delivered: bool, error: str | None) -> None:
        # Mark in-flight so a pump pass before the DeliveryResult is reduced won't re-report it.
        self._inflight.add(message.message_id)
        self._enqueue(
            DeliveryResult(
                event_id=ids.new_id(ids.EVENT),
                timestamp=self._clock.now_utc(),
                source="delivery_adapter",
                message_id=message.message_id,
                delivery_key=message.delivery_key,
                delivered=delivered,
                error=error,
            )
        )
