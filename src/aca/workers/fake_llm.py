"""Deterministic fake generative worker.

Emulates the DESIGN 22 typed-proposal contract with zero network calls and fully deterministic
output, so the whole cognition pipeline (and its adversarial tests) runs offline and replayably
(DESIGN 26.1). A real Claude-backed worker slots in behind ``LLMWorker`` later.

Decision policy (interpretation only — the reducer decides what the output is *allowed* to do):

* mandatory reply requested  -> SPEAK a helpful acknowledgement of the task;
* salient proactive candidate -> SPEAK a reflective opener + enrichment proposal;
* low-salience candidate      -> SILENCE + enrichment proposal (silent-but-useful, DESIGN 19.1);
* nothing worth saying        -> SILENCE.
"""

from __future__ import annotations

from typing import Any

from ..cognition.snapshot import Snapshot
from .base import LLMOutput

_SALIENCE_SPEAK_THRESHOLD = 0.6


class FakeLLMWorker:
    """A deterministic stand-in for the generative model."""

    async def run(self, snapshot: Snapshot) -> LLMOutput:
        source: dict[str, Any] = snapshot.context.get("source") or {}
        if source.get("response_required"):
            return self._mandatory_reply(source)
        candidate = source.get("candidate")
        if candidate:
            return self._proactive(candidate)
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
    def _proactive(candidate: dict[str, Any]) -> LLMOutput:
        salience = float(candidate.get("salience", 0.0))
        subject = str(
            candidate.get("summary") or candidate.get("intent") or candidate.get("text", "")
        ).strip()
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
