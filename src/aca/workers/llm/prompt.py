"""Deterministic, provider-neutral prompt construction for the generative boundary (DESIGN 21, 22).

Turns a canonical ``Snapshot`` into a ``(system, user)`` pair. The system prompt fixes the JSON
action contract the reducer expects; the user prompt is the compact, canonically-serialized
context. Determinism (sorted keys, explicit timestamps from state) keeps prompts stable for
replay and prompt-drift detection (DESIGN 21).
"""

from __future__ import annotations

import json

from ...cognition.snapshot import Snapshot
from ...domain.cycles import CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL

_SYSTEM = """\
You are the cognition core of an asynchronous, introspective conversational agent (ACA).
Human input is optional and your output is optional. You decide whether to speak.

Respond with a SINGLE JSON object and nothing else — no prose, no code fences:
  {"action": "speak" | "silence" | "defer" | "acknowledge",
   "message": "<text, required only when action is speak>",
   "proposals": [ ... optional typed proposals ... ]}

Rules by cycle type (given in context.source.cycle_type):
- "mandatory": the human asked something requiring a response — you MUST "speak" a helpful,
  direct answer. Silence is not acceptable.
- "reactive": an optional reply to what the human just said — "speak" briefly if it adds value,
  otherwise "silence".
- "proactive": an unsolicited thought surfaced on your own — "speak" only if it is genuinely
  worth interrupting for right now; otherwise "silence". Do not sound like you are resurfacing
  something the human said seconds ago.

Keep messages concise and natural. Only these proposal types are honored (others are ignored):
  {"type":"ENRICH_PROVISIONAL_MEMORY","provisional_memory_id":"...","topic_summary":"...","tags":[...]}
  {"type":"ADJUST_TOPIC_ACTIVATION","topic_id":"...","delta":<-0.5..0.5>}
  {"type":"CREATE_DEFERRED_INTENT","intent":"...","topic_id":"...","provisional_memory_id":"..."}
  {"type":"RESOLVE_DEFERRED_INTENT","intent_id":"..."}
"""


def build_prompt(snapshot: Snapshot) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the given snapshot."""
    context = json.dumps(snapshot.context, sort_keys=True, indent=2, default=str)
    user = (
        f"cycle_id: {snapshot.cycle_id}\n"
        f"basis_revision: {snapshot.basis_revision}\n"
        f"context:\n{context}\n\n"
        "Return your decision as the single JSON object described in the system prompt."
    )
    return _SYSTEM, user


def cycle_type(snapshot: Snapshot) -> str:
    """The cycle type carried in the snapshot (defaults to reactive-optional)."""
    source = snapshot.context.get("source") or {}
    return str(source.get("cycle_type", CYCLE_REACTIVE_OPTIONAL))


def wants_speech(snapshot: Snapshot) -> bool:
    """Whether this cycle should fall back to speaking if the model returns non-JSON prose."""
    source = snapshot.context.get("source") or {}
    return cycle_type(snapshot) in (CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL) or bool(
        source.get("response_required")
    )
