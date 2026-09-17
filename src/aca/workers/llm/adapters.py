"""Per-provider HTTP chat adapters.

Two request shapes cover the four providers: an OpenAI-style ``/chat/completions`` call (OpenAI,
Grok/xAI, Gemini's OpenAI-compatible endpoint) and Anthropic's ``/v1/messages``. Each adapter is
constructed with its resolved endpoint/credential/model and exposes one async ``complete`` that
returns text plus token usage. Network/HTTP errors propagate so the dispatcher's bounded retry +
terminal-failure handling applies (invariant 25). ``httpx`` is imported lazily so the package
imports without it when only the fake provider is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...errors import WorkerError

# GPT-5 family and o-series reasoning models reject the legacy ``max_tokens`` field on
# /chat/completions and require ``max_completion_tokens``; older chat models (gpt-4*, gpt-3.5) and
# the OpenAI-compatible third parties (Grok, Gemini) still take ``max_tokens``.
_MAX_COMPLETION_TOKENS_MODELS = re.compile(r"^(?:gpt-5|o[0-9])", re.IGNORECASE)


def _token_limit_param(model: str) -> str:
    return "max_completion_tokens" if _MAX_COMPLETION_TOKENS_MODELS.match(model) else "max_tokens"


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int
    # True when generation was cut off by the token cap (OpenAI finish_reason == "length",
    # Anthropic stop_reason == "max_tokens"). A truncated reply must not be treated as complete.
    truncated: bool = False


@runtime_checkable
class ChatAdapter(Protocol):
    async def complete(self, system: str, user: str) -> ChatResult:
        """Send one system+user turn and return the model's text and token usage."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleAdapter:
    """OpenAI-style ``/chat/completions`` — used by OpenAI, Grok (xAI), and Gemini (compat)."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int
    timeout_seconds: float

    async def complete(self, system: str, user: str) -> ChatResult:
        import httpx

        payload = {
            "model": self.model,
            _token_limit_param(self.model): self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                # Surface the provider's error body — a bare "400 Bad Request" hides the actual
                # cause (wrong param, unknown model, quota) from `aca logs`/last_error.
                raise WorkerError(f"chat/completions {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage") or {}
        return ChatResult(
            text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            truncated=choice.get("finish_reason") == "length",
        )


@dataclass(frozen=True, slots=True)
class AnthropicAdapter:
    """Anthropic Messages API (``/v1/messages``)."""

    base_url: str
    api_key: str
    model: str
    max_tokens: int
    timeout_seconds: float
    anthropic_version: str = "2023-06-01"

    async def complete(self, system: str, user: str) -> ChatResult:
        import httpx

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/v1/messages", json=payload, headers=headers
            )
            if resp.status_code >= 400:
                raise WorkerError(f"v1/messages {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return ChatResult(
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            truncated=data.get("stop_reason") == "max_tokens",
        )
