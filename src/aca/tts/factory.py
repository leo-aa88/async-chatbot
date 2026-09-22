"""Build a ``TtsEngine`` from ``config.Tts`` (provider selection).

Mirrors the LLM/embedding factories: ``none`` is the silent default and needs nothing; ``kokoro``
constructs the offline Kokoro-82M engine. Construction validates config eagerly (fails fast on an
unknown provider or an empty voice), while the heavy model/audio imports stay lazy inside the engine
so ``aca chat`` still starts without the ``tts`` extra until speech is actually used.
"""

from __future__ import annotations

from ..config import Tts
from ..errors import ConfigError
from .base import NullTtsEngine, TtsEngine
from .kokoro_engine import KokoroTtsEngine

_ALIASES = {"off": "none", "disabled": "none", "silent": "none", "": "none"}


def build_tts_engine(config: Tts) -> TtsEngine:
    """Construct the speech engine for the configured provider (silent ``NullTtsEngine`` default)."""
    provider = _ALIASES.get(config.provider, config.provider)
    if provider == "none":
        return NullTtsEngine()
    if provider == "kokoro":
        return KokoroTtsEngine(
            voice=config.voice,
            lang_code=config.lang_code,
            speed=config.speed,
            device=config.device,
        )
    raise ConfigError(f"unknown tts.provider {config.provider!r} (known: none, kokoro)")
