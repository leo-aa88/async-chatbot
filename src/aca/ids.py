"""Identifier generation.

IDs are human-readable, prefixed, and sortable-ish (time-ordered random hex). Prefixes make
traces legible (``evt_``, ``pmem_``, ``topic_``...). ID generation is the one place we allow
non-determinism from ``os.urandom``; tests that need deterministic IDs inject their own values
rather than reseeding this module.
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
