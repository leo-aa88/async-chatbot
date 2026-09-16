"""Identifier generation.

IDs are human-readable, prefixed, and sortable-ish (time-ordered random hex). Prefixes make
traces legible (``evt_``, ``pmem_``, ``topic_``...).

**Sanctioned clock/randomness exception (see AGENTS.md / CLAUDE.md).** The project rule is that
cognition and the reducer take time/randomness through an injected ``Clock``/``Rng``. ID minting
is the one deliberate exception: it reads ``time.time()`` and ``os.urandom`` directly. This is
safe for replay determinism because an ID is an *opaque durable key* — it is never an input to a
cognition decision (selection, gating, scheduling all read scores/state, never an ID). A row's
identity therefore differs between two otherwise-identical runs, but the decisions do not; tests
that assert on durable identity inject their own IDs rather than minting them here.
"""

from __future__ import annotations

import os
import time

# Canonical prefixes, one per durable entity kind. Centralized so they never drift.
EVENT = "evt"
PROVISIONAL_MEMORY = "pmem"
TOPIC = "topic"
DEFERRED_INTENT = "intent"
WORK_ITEM = "work"
CYCLE = "cog"
ACTION = "act"
OUTBOUND = "msg"
OBLIGATION = "oblig"
RUNTIME_SESSION = "sess"
AGENT = "agent"
TRACE = "trace"
DELIVERY = "dlv"
EMBEDDING = "emb"


def new_id(prefix: str) -> str:
    """Return a new prefixed, time-ordered identifier.

    Format: ``<prefix>_<millis_hex><random_hex>``. The millisecond timestamp keeps IDs
    roughly monotonic for readability; the random suffix guarantees uniqueness.
    """
    millis = int(time.time() * 1000)
    return f"{prefix}_{millis:011x}{os.urandom(5).hex()}"


def new_delivery_key() -> str:
    """Return an opaque delivery key used for at-least-once transport deduplication."""
    return new_id(DELIVERY)
