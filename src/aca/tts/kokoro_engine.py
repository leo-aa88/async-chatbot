"""Kokoro-82M offline neural TTS (default voice ``am_onyx``).

Kokoro is a small (82M-parameter) open-weights TTS model that runs locally — no API, no per-call
cost, nothing leaves the host. The heavy dependencies (``kokoro`` + torch for synthesis,
``sounddevice`` + ``numpy`` for playback) are imported lazily inside the methods, so importing this
module — and the whole ``aca`` package — costs nothing until speech is actually enabled and used,
and ``aca chat`` still starts on a host without the ``tts`` extra installed.

Synthesis and playback are blocking / CPU-bound, so the async ``speak`` offloads them to a worker
thread. The caller (:class:`~aca.tts.controller.SpeechController`) already serializes calls, so
utterances never overlap and the engine needs no lock of its own.
"""

from __future__ import annotations

import asyncio

from ..errors import ConfigError
from .base import TtsEngine

# Kokoro's convention: a voice name's first letter is its language, the second its gender — e.g.
# ``am_onyx`` = American male, ``bf_emma`` = British female. The pipeline is built per language, so
# when ``lang_code`` is left blank we derive it from that first letter.
_LANG_BY_PREFIX = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Brazilian Portuguese
    "z": "z",  # Mandarin Chinese
}


def _resolve_lang_code(voice: str, lang_code: str) -> str:
    if lang_code:
        return lang_code
    return _LANG_BY_PREFIX.get(voice[:1].lower(), "a") if voice else "a"


class KokoroTtsEngine(TtsEngine):
    def __init__(
        self,
        *,
        voice: str,
        lang_code: str,
        speed: float,
        sample_rate: int,
        device: str | None = None,
    ) -> None:
        if not voice:
            raise ConfigError("tts.voice is required for tts.provider 'kokoro'")
        self._voice = voice
        self._lang_code = _resolve_lang_code(voice, lang_code)
        self._speed = speed
        self._sample_rate = sample_rate
        self._device = device
        self._pipeline = None  # lazily built on first speak (loads the model)

    async def speak(self, text: str) -> None:
        text = text.strip()
        if text:
            await asyncio.to_thread(self._render_and_play, text)

    async def aclose(self) -> None:
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as exc:  # pragma: no cover - only without the tts extra
                raise ConfigError(
                    "the 'kokoro' package is required for tts.provider 'kokoro': "
                    "pip install 'aca[tts]'"
                ) from exc
            self._pipeline = KPipeline(lang_code=self._lang_code)
        return self._pipeline

    def _render_and_play(self, text: str) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - only without the tts extra
            raise ConfigError(
                "the 'sounddevice' and 'numpy' packages are required to play Kokoro audio: "
                "pip install 'aca[tts]'"
            ) from exc

        pipeline = self._ensure_pipeline()
        for chunk in pipeline(text, voice=self._voice, speed=self._speed):
            # Kokoro yields (graphemes, phonemes, audio) tuples on older versions and Result objects
            # with an ``.audio`` attribute on newer ones — accept both.
            audio = chunk[2] if isinstance(chunk, tuple) else chunk.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):  # a torch tensor -> numpy on CPU
                audio = audio.detach().cpu().numpy()
            samples = np.asarray(audio, dtype="float32")
            sd.play(samples, self._sample_rate, device=self._device)
            sd.wait()
