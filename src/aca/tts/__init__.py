"""Optional speech output for the chat client (DESIGN 2.2 secondary goal).

A client-side rendering layer: the delivered agent text the chat client already prints is also
spoken aloud. It sits entirely outside cognition and the reducer and never mutates durable state
(invariant 1). Silent by default (``tts.provider = "none"``); opt in with Kokoro-82M.
"""

from __future__ import annotations

from .base import NullTtsEngine, TtsEngine
from .controller import SpeechController
from .factory import build_tts_engine
from .kokoro_engine import KokoroTtsEngine

__all__ = [
    "TtsEngine",
    "NullTtsEngine",
    "KokoroTtsEngine",
    "SpeechController",
    "build_tts_engine",
]
