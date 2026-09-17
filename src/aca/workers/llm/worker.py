"""The provider-agnostic generative worker.

Builds the shared prompt, calls a ``ChatAdapter``, and coerces the reply into the raw
typed-proposal dict the reducer validates. If a provider returns prose instead of JSON, a
speech-oriented cycle (mandatory/reactive) falls back to treating that prose as the message so a
minor format slip still answers the user; a proactive cycle falls back to silence. All numeric
clamping / whitelisting still happens later in ``domain.proposals`` — this worker never mutates
state.
"""

from __future__ import annotations

import json
from typing import Any

from ...cognition.snapshot import Snapshot
from ..base import LLMOutput
from .adapters import ChatAdapter
from .prompt import build_prompt, wants_speech


class ProviderLLMWorker:
    """An ``LLMWorker`` backed by any ``ChatAdapter``."""

    def __init__(self, adapter: ChatAdapter) -> None:
        self._adapter = adapter

    async def run(self, snapshot: Snapshot) -> LLMOutput:
        system, user = build_prompt(snapshot)
        result = await self._adapter.complete(system, user)
        decision = _coerce(result.text, wants_speech(snapshot))
        return LLMOutput(result=decision, tokens_in=result.tokens_in, tokens_out=result.tokens_out)


def _coerce(text: str, speech_fallback: bool) -> dict[str, Any]:
    """Turn raw model text into a decision dict for the reducer to validate."""
    parsed = _extract_json(text)
    if isinstance(parsed, dict) and "action" in parsed:
        return parsed
    # Non-JSON prose: keep the user answered on speech cycles; stay silent on proactive ones.
    cleaned = text.strip()
    if speech_fallback and cleaned:
        return {"action": "speak", "message": cleaned}
    return {"action": "silence"}


def _extract_json(text: str) -> Any:
    """Best-effort JSON object extraction (tolerates ```json fences and surrounding prose)."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        # Strip a fenced block: ```json\n...\n```
        candidate = candidate.split("```", 2)[1] if candidate.count("```") >= 2 else candidate
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
