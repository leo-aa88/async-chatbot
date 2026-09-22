"""TTS engine abstraction (DESIGN 2.2 secondary goal: speech output).

Speech output is a *client-side rendering* concern. The chat client already prints each delivered
agent utterance; a ``TtsEngine`` renders that same text as audio. It lives entirely outside
cognition and the reducer, never mutates durable state (invariant 1), and so needs no injected
``Clock``/``Rng``. The default engine is silent — speech output is strictly opt-in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TtsEngine(ABC):
    """Renders delivered agent text to audio. Concrete engines own their own audio/model resources.

    ``enabled`` lets callers skip all speech machinery cheaply when output is off, without an
    ``isinstance`` check; the silent :class:`NullTtsEngine` sets it ``False``.
    """

    enabled: bool = True

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Render ``text`` and block until playback finishes (heavy work offloaded to a thread).

        Callers serialize invocations so utterances never overlap; an engine need not lock itself.
        """

    async def aclose(self) -> None:
        """Release audio/model resources. Idempotent; safe to call even if nothing was spoken."""
        return None


class NullTtsEngine(TtsEngine):
    """The silent default: speech output disabled. Does nothing and needs no dependencies."""

    enabled = False

    async def speak(self, text: str) -> None:
        return None
