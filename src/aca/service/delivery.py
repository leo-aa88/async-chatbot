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

import asyncio
from collections.abc import Awaitable, Callable

from .. import ids
from ..clock import Clock
from ..config import Config
from ..domain.enums import OutboundKind
from ..domain.events import DeliveryResult
from ..persistence.stores import Stores
from ..reducer.context import ReducerContext
from ..reducer.discourse import is_discourse_orphan
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
        # Monotonic timestamp of the last proactive delivery attempt; bounds the reconnect burst
        # window across pump() calls and channels so reconnect can't dogpile (DESIGN 23.7).
        self._last_proactive_mono: float | None = None
        # Serializes pump passes. The reducer loop and notify_client_connected can both call
        # pump(); each pass snapshots the pending list and then awaits transport, so overlapping
        # passes would resend items from a stale snapshot and let two proactive items through one
        # burst window (invariant 38). Nothing inside a pass re-enters pump(), so this cannot deadlock.
        self._lock = asyncio.Lock()

    def on_delivery_result(self, message_id: str) -> None:
        self._inflight.discard(message_id)

    async def pump(self) -> None:
        """Attempt one delivery pass. Mandatory first, then at most one proactive item."""
        async with self._lock:
            await self._pump_pass()

    async def _pump_pass(self) -> None:
        now = self._clock.now_utc()
        pending = [m for m in self._stores.outbox.deliverable() if m.message_id not in self._inflight]
        mandatory = [m for m in pending if m.kind is OutboundKind.MANDATORY]
        proactive = [m for m in pending if m.kind is OutboundKind.PROACTIVE]

        for message in mandatory:
            await self._attempt(message, now)

        await self._pump_one_proactive(proactive, now)

    async def _pump_one_proactive(self, proactive, now) -> None:
        # Enforce the reconnect burst window across pump() calls / channels: at most one proactive
        # delivery per window, so two channels reconnecting together cannot both slip through
        # before the self-model cooldown updates (DESIGN 23.7 rule 5, invariant 38).
        burst_elapsed = self._burst_window_elapsed()
        delivered_one = False
        for message in proactive:
            if message.expires_at is not None and message.expires_at <= now:
                self._report(message, delivered=False, error="expired:ttl")
                continue
            gate = evaluate_proactive(self._ctx, now)
            if not gate.allowed:
                self._report(message, delivered=False, error=f"revalidation:{gate.reason}")
                continue
            # Discourse re-check before transport (§34.6 checkpoint 3, invariant 9): a proactive
            # item on-topic at decision time may be delivered late, after the conversation moved on.
            # Uses the outbound row's persisted candidate (§23.7), not a live candidate object.
            if message.candidate_id is not None and is_discourse_orphan(
                self._ctx, message.candidate_kind, message.candidate_id, now
            ):
                self._report(message, delivered=False, error="revalidation:discourse_orphan")
                continue
            if delivered_one or not burst_elapsed:
                # Keep the rest pending rather than dogpiling; they retry on a later pump.
                continue
            await self._attempt(message, now)
            self._last_proactive_mono = self._clock.monotonic()
            delivered_one = True

    def _burst_window_elapsed(self) -> bool:
        if self._last_proactive_mono is None:
            return True
        window = self._config.timing.proactive_burst_window_seconds
        return (self._clock.monotonic() - self._last_proactive_mono) >= window

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
