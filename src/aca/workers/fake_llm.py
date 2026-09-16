"""Deterministic fake generative worker.

Emulates the DESIGN 22 typed-proposal contract with zero network calls and fully deterministic
output, so the whole cognition pipeline (and its adversarial tests) runs offline and replayably
(DESIGN 26.1). A real Claude-backed worker slots in behind ``LLMWorker`` later.

Decision policy (interpretation only — the reducer decides what the output is *allowed* to do).
The worker chooses a voice appropriate to the cycle type (DESIGN 14) so a reply never sounds out
of time:

* **mandatory** reply requested  -> answer the task directly;
* **reactive** optional reply     -> respond to the message just sent (present tense);
* **proactive** initiative        -> a reflective opener ("I've been thinking...") that only
  makes sense when *resurfacing* an older thought, plus an enrichment proposal;
* nothing worth saying            -> SILENCE.
"""

from __future__ import annotations

from typing import Any

from ..cognition.snapshot import Snapshot
from ..domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE, CYCLE_REACTIVE_OPTIONAL
from .base import LLMOutput

_SALIENCE_SPEAK_THRESHOLD = 0.6


class FakeLLMWorker:
    """A deterministic stand-in for the generative model."""

    async def run(self, snapshot: Snapshot) -> LLMOutput:
        source: dict[str, Any] = snapshot.context.get("source") or {}
        cycle_type = source.get("cycle_type")
        if cycle_type == CYCLE_MANDATORY or source.get("response_required"):
            return self._mandatory_reply(source)
        if cycle_type == CYCLE_REACTIVE_OPTIONAL:
            return self._reactive_reply(source)
        if cycle_type == CYCLE_PROACTIVE:
            return self._proactive(source.get("candidate") or {})
        return LLMOutput(result={"action": "silence"}, tokens_in=64, tokens_out=1)

    @staticmethod
    def _mandatory_reply(source: dict[str, Any]) -> LLMOutput:
        text = str(source.get("text", "")).strip()
        message = (
            f"Here's what I can say about that: {text[:200]}"
            if text
            else "Could you clarify what you'd like me to do?"
        )
        return LLMOutput(result={"action": "speak", "message": message}, tokens_in=256, tokens_out=48)

    @staticmethod
    def _reactive_reply(source: dict[str, Any]) -> LLMOutput:
        # A reply to the message the user just sent — present tense, no "I've been thinking".
        text = str(source.get("text") or _subject(source.get("candidate") or {})).strip()
        message = f"Got it — noted: {text[:200]}" if text else "Got it."
        return LLMOutput(
            result={"action": "speak", "message": message}, tokens_in=200, tokens_out=24
        )

    @staticmethod
    def _proactive(candidate: dict[str, Any]) -> LLMOutput:
        salience = float(candidate.get("salience", 0.0))
        subject = _subject(candidate)
        proposals: list[dict[str, Any]] = []
        memory_id = candidate.get("provisional_memory_id") or (
            candidate.get("id") if candidate.get("kind") == "PROVISIONAL_MEMORY" else None
        )
        if memory_id:
            proposals.append(
                {
                    "type": "ENRICH_PROVISIONAL_MEMORY",
                    "provisional_memory_id": memory_id,
                    "topic_summary": subject[:200],
                }
            )
        if salience >= _SALIENCE_SPEAK_THRESHOLD and subject:
            message = f"I've been thinking about what you said: {subject[:180]}"
            return LLMOutput(
                result={"action": "speak", "message": message, "proposals": proposals},
                tokens_in=320,
                tokens_out=40,
            )
        return LLMOutput(
            result={"action": "silence", "proposals": proposals, "useful_enrichment": bool(proposals)},
            tokens_in=300,
            tokens_out=8,
        )


def _subject(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("summary") or candidate.get("intent") or candidate.get("text", "")
    ).strip()
