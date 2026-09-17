"""Build an ``EmbeddingWorker`` from ``config.Embedding`` (provider + credential resolution).

Mirrors the LLM factory: ``fake`` is the deterministic offline default and needs nothing; a real
provider fails fast at startup with a clear error if its key env var is unset or ``httpx`` isn't
installed, rather than only erroring on the first embedding call.
"""

from __future__ import annotations

import importlib.util
import os

from ..config import Embedding
from ..errors import ConfigError
from .base import EmbeddingWorker
from .embeddings import OpenAIEmbeddingAdapter, ProviderEmbeddingWorker
from .fake_embedding import FakeEmbeddingWorker

# Embedding endpoints are OpenAI-style; base_url is overridable for other compatible providers.
_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
}
_ALIASES = {"mock": "fake", "google": "gemini"}


def embedding_key_env_for(config: Embedding) -> str | None:
    """The env var holding this provider's key (None for fake/unknown)."""
    provider = _ALIASES.get(config.provider, config.provider)
    if provider == "fake" or provider not in _PROVIDERS:
        return None
    return config.api_key_env or _PROVIDERS[provider][1]


def build_embedding_worker(config: Embedding) -> EmbeddingWorker:
    """Construct the embedding worker for the configured provider."""
    provider = _ALIASES.get(config.provider, config.provider)
    if provider == "fake":
        return FakeEmbeddingWorker()

    spec = _PROVIDERS.get(provider)
    if spec is None:
        known = ", ".join(["fake", *sorted(_PROVIDERS)])
        raise ConfigError(f"unknown embedding.provider {config.provider!r} (known: {known})")
    if not config.model:
        raise ConfigError(f"embedding.model is required for provider {provider!r}")

    base_url, default_key_env = spec
    key_env = config.api_key_env or default_key_env
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ConfigError(f"{key_env} is not set (required for embedding.provider {provider!r})")
    if importlib.util.find_spec("httpx") is None:
        raise ConfigError("the 'httpx' package is required for real embeddings: pip install 'aca[llm]'")

    adapter = OpenAIEmbeddingAdapter(
        config.base_url or base_url, api_key, config.model, config.timeout_seconds
    )
    return ProviderEmbeddingWorker(adapter)
