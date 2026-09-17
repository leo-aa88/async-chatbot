"""Build an ``LLMWorker`` from ``config.LLM`` (provider selection + credential resolution).

Providers resolve with no configuration beyond an API key in the environment (mirroring the
reference design). ``fake`` (alias ``mock``) is the deterministic offline default and needs
nothing. Real providers fail fast at startup with a clear error if their key env var is unset or
``httpx`` isn't installed, rather than only erroring on the first cognition cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ...config import LLM
from ...errors import ConfigError
from ..base import LLMWorker
from ..fake_llm import FakeLLMWorker
from .adapters import AnthropicAdapter, ChatAdapter, OpenAICompatibleAdapter
from .worker import ProviderLLMWorker


@dataclass(frozen=True, slots=True)
class _ProviderSpec:
    style: str  # "openai" | "anthropic"
    base_url: str
    key_env: str


# Endpoints are overridable via config.base_url if a provider moves them.
_PROVIDERS: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "grok": _ProviderSpec("openai", "https://api.x.ai/v1", "XAI_API_KEY"),
    "gemini": _ProviderSpec(
        "openai", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"
    ),
    "anthropic": _ProviderSpec("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
}
_ALIASES = {"claude": "anthropic", "xai": "grok", "google": "gemini", "mock": "fake"}


def key_env_for(config: LLM) -> str | None:
    """The single environment variable holding this provider's key (None for fake/unknown).

    Lets the caller load a ``.env`` scoped to exactly the credential this agent needs, instead of
    absorbing an unrelated directory's secrets into a long-running daemon.
    """
    provider = _ALIASES.get(config.provider, config.provider)
    if provider == "fake":
        return None
    spec = _PROVIDERS.get(provider)
    if spec is None:
        return None
    return config.api_key_env or spec.key_env


def build_llm_worker(config: LLM) -> LLMWorker:
    """Construct the generative worker for the configured provider."""
    provider = _ALIASES.get(config.provider, config.provider)
    if provider == "fake":
        return FakeLLMWorker()

    spec = _PROVIDERS.get(provider)
    if spec is None:
        known = ", ".join(["fake", *sorted(_PROVIDERS)])
        raise ConfigError(f"unknown llm.provider {config.provider!r} (known: {known})")
    if not config.model:
        raise ConfigError(f"llm.model is required for provider {provider!r}")

    key_env = config.api_key_env or spec.key_env
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ConfigError(f"{key_env} is not set (required for llm.provider {provider!r})")
    if not _httpx_available():
        raise ConfigError("the 'httpx' package is required for real LLM providers: pip install 'aca[llm]'")

    base_url = config.base_url or spec.base_url
    adapter = _build_adapter(spec.style, base_url, api_key, config)
    return ProviderLLMWorker(adapter)


def _build_adapter(style: str, base_url: str, api_key: str, config: LLM) -> ChatAdapter:
    if style == "anthropic":
        return AnthropicAdapter(base_url, api_key, config.model, config.max_tokens, config.timeout_seconds)
    return OpenAICompatibleAdapter(base_url, api_key, config.model, config.max_tokens, config.timeout_seconds)


def _httpx_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("httpx") is not None
