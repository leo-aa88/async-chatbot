"""Provider-agnostic generative worker (DESIGN 22, 28.5).

A single ``ProviderLLMWorker`` implements the ``LLMWorker`` protocol and delegates the actual
network call to a small per-provider ``ChatAdapter``. Prompt construction is shared and
deterministic; adapters differ only in HTTP shape (OpenAI-style for OpenAI/Grok/Gemini, Messages
API for Anthropic). Whatever a provider returns is still validated and clamped by the reducer —
model output is data, never authority.
"""

from .factory import build_llm_worker
from .worker import ProviderLLMWorker

__all__ = ["ProviderLLMWorker", "build_llm_worker"]
