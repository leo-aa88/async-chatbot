"""IPC server: bridges Unix-socket clients to the agent service (DESIGN 28.2).

The server is a thin adapter. It accepts client requests, forwards human messages through the
service's durable ingress (ACK only after commit), and serves read-only status/memory/topic/log
queries. Subscribed ``chat`` connections receive delivered agent messages pushed by the
service's delivery sink; each push carries a ``delivery_key`` so clients deduplicate (inv 38).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from ..domain.events import HumanMessage
from ..service.service import AgentService
from . import protocol as p


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 for the wire; the client renders it to local wall-clock time for display."""
    return value.isoformat() if value is not None else None


class IpcServer:
    def __init__(self, service: AgentService, socket_path: str | Path) -> None:
        self._service = service
        self._socket_path = Path(socket_path)
        self._subscribers: set[asyncio.StreamWriter] = set()
        self._server: asyncio.AbstractServer | None = None
        self._closed = asyncio.Event()

    async def start(self) -> None:
        if self._socket_path.exists():
            self._socket_path.unlink()
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._service.set_sink(self._broadcast)
        self._server = await asyncio.start_unix_server(self._handle, path=str(self._socket_path))

    async def serve_until_shutdown(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._socket_path.exists():
            self._socket_path.unlink()

    # --- connection handling -------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                message = await p.read_message(reader)
                if message is None:
                    break
                keep_open = await self._dispatch(message, writer)
                if not keep_open:
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._subscribers.discard(writer)
            writer.close()

    async def _dispatch(self, message: dict, writer: asyncio.StreamWriter) -> bool:
        op = message.get("op")
        if op == p.OP_CHAT:
            await self._on_chat(message, writer)
        elif op == p.OP_SUBSCRIBE:
            self._subscribers.add(writer)
            await p.write_message(writer, p.ok({"subscribed": True}))
            await self._service.notify_client_connected()
        elif op == p.OP_STATUS:
            await p.write_message(writer, p.ok(self._status()))
        elif op == p.OP_MEMORIES:
            await p.write_message(writer, p.ok({"memories": self._memories()}))
        elif op == p.OP_TOPICS:
            await p.write_message(writer, p.ok({"topics": self._topics()}))
        elif op == p.OP_LOGS:
            await p.write_message(writer, p.ok({"traces": self._logs()}))
        elif op == p.OP_METRICS:
            await p.write_message(writer, p.ok({"metrics": self._metrics()}))
        elif op == p.OP_SHUTDOWN:
            await p.write_message(writer, p.ok({"shutting_down": True}))
            self._closed.set()
            return False
        else:
            await p.write_message(writer, p.err(f"unknown op: {op!r}"))
        return True

    async def _on_chat(self, message: dict, writer: asyncio.StreamWriter) -> None:
        event = HumanMessage(
            event_id=str(message["event_id"]),
            timestamp=self._service.clock.now_utc(),  # server stamps acceptance time
            source="cli",
            text=str(message.get("text", "")),
            channel=str(message.get("channel", "cli")),
        )
        accepted = await self._service.ingest_human_message(event)
        await p.write_message(writer, p.ok({"accepted": accepted, "event_id": event.event_id}))

    async def _broadcast(self, channel: str, payload: str, delivery_key: str, message_id: str) -> bool:
        frame = {
            "kind": p.PUSH_MESSAGE, "channel": channel, "text": payload,
            "delivery_key": delivery_key, "message_id": message_id,
            "at": _iso(self._service.clock.now_utc()),
        }
        dead: list[asyncio.StreamWriter] = []
        delivered = False
        for sub in self._subscribers:
            try:
                await p.write_message(sub, frame)
                delivered = True
            except (ConnectionResetError, BrokenPipeError):
                dead.append(sub)
        for sub in dead:
            self._subscribers.discard(sub)
        return delivered

    # --- read models ---------------------------------------------------------------------
    def _status(self) -> dict:
        stores = self._service.stores
        identity = stores.identity.load_identity()
        conversation = stores.state.load_conversation()
        return {
            "agent_id": identity.agent_id if identity else None,
            "lifecycle_state": identity.lifecycle_state.value if identity else None,
            "runtime_session_id": self._service.reducer.context.runtime_session_id,
            "scheduler_generation": self._service.reducer.context.scheduler_generation,
            "state_revision": stores.state.state_revision(),
            "conversation_mode": conversation.mode.value,
            "pending_obligations": len(stores.work.pending_obligations()),
        }

    # The repeated-candidate window is proportional to the agent's own repeat-suppression cadence,
    # so "recently" means the same thing regardless of how the agent is tuned.
    _REPEATED_WINDOW_FACTOR = 24

    def _metrics(self) -> dict:
        from datetime import timedelta

        config = self._service.reducer.context.config
        window_s = config.memory.repeat_suppression_seconds * self._REPEATED_WINDOW_FACTOR
        since = self._service.clock.now_utc() - timedelta(seconds=window_s)
        metrics = self._service.stores.work.trace_metrics(repeated_since=since)
        metrics["repeated_window_seconds"] = int(window_s)
        largest, total = self._semantic_dominance(since, config.memory.semantic_neighbor_threshold)
        metrics["dominance_cluster"] = largest
        metrics["dominance_total"] = total
        return metrics

    def _semantic_dominance(self, since, threshold: float) -> tuple[int, int]:
        """Largest semantic cluster among recently self-voiced (embeddable) candidates.

        Observational only — never fed back into activation (that would be a self-reinforcing
        obsession loop). A spoken candidate is usually a TOPIC (enriched memories are dropped from
        candidacy), so a topic is resolved to its source memory's embedding; a raw memory candidate
        resolves directly; a deferred intent has no embedding and is skipped. Vectors are kept in
        candidate order (clustering is order-sensitive) and grouped by embedding model before
        comparing (cosine across models is meaningless).
        """
        from ..cognition.vectors import dominant_cluster_fraction

        mem = self._service.stores.memory
        candidates = self._service.stores.work.proactive_spoken_candidates(since=since)
        # Resolve each spoken candidate to the memory whose embedding represents it, in order.
        memory_ids: list[str] = []
        for kind, cid in candidates:
            if kind == "PROVISIONAL_MEMORY":
                memory_ids.append(cid)
            elif kind == "TOPIC":
                topic = mem.get_topic(cid)
                if topic is not None and topic.source_memory_id:
                    memory_ids.append(topic.source_memory_id)
            # DEFERRED_INTENT / NOTHING: no embedding to cluster
        embeddings = mem.embeddings_by_memory(memory_ids)
        ordered = [embeddings[mid] for mid in memory_ids if mid in embeddings]  # preserve order
        if len(ordered) < 2:
            return (0, len(ordered))
        # Cluster within each embedding model (cosine across models is meaningless), but count the
        # denominator as the whole embeddable set. The largest single-model cluster is the numerator.
        by_model: dict[str, list] = {}
        for model_version, vector in ordered:
            by_model.setdefault(model_version, []).append(vector)
        largest = max(dominant_cluster_fraction(g, threshold)[0] for g in by_model.values())
        return (largest, len(ordered))

    def _memories(self) -> list[dict]:
        return [
            {"id": m.id, "text": m.text, "activation": m.activation,
             "enrichment_status": m.enrichment_status.value,
             "created_at": _iso(m.created_at), "last_activated_at": _iso(m.last_activated_at)}
            for m in self._service.stores.memory.recent_memories(limit=20)
        ]

    def _topics(self) -> list[dict]:
        return [
            {"id": t.id, "summary": t.summary, "activation": t.activation,
             "unfinished": t.unfinished,
             "created_at": _iso(t.created_at), "last_activated_at": _iso(t.last_activated_at)}
            for t in self._service.stores.memory.all_topics(limit=20)
        ]

    def _logs(self) -> list[dict]:
        return [
            {"cycle_id": tr.cycle_id, "trigger": tr.trigger, "action": tr.action,
             "candidate_kind": tr.candidate_kind, "llm_called": tr.llm_called, "notes": tr.notes,
             "created_at": _iso(tr.created_at)}
            for tr in self._service.stores.work.recent_traces(limit=20)
        ]
