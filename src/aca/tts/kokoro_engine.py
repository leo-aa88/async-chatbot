"""Kokoro-82M offline neural TTS (default voice ``am_onyx``).

Kokoro is a small (82M-parameter) open-weights TTS model. It runs locally: after a **one-time**
model download from Hugging Face on first use, no audio and no text leave the host. The heavy
dependencies (``kokoro`` + torch for synthesis, ``sounddevice`` + ``numpy`` for playback) are
imported lazily inside the methods, so importing this module — and the whole ``aca`` package —
costs nothing until speech is enabled and used, and ``aca chat`` still starts on a host without the
``tts`` extra installed.

Synthesis and playback are blocking / CPU-bound, so the async ``speak`` offloads them to a worker
thread. Two things that thread must *not* do are leak onto the terminal or refuse to stop:

* The chat ``ChatUI`` owns the terminal exclusively (one event-loop writer, no threads). The worker
  must therefore never reassign ``sys.stdout``/``sys.stderr`` — those are process-global, shared
  with the event loop, so redirecting them would send the main thread's keystroke echo and
  delivered messages to ``/dev/null`` for the whole utterance. Instead every noise source is
  silenced *at origin*: an explicit ``repo_id`` (so Kokoro doesn't print its default-repo warning),
  ``HF_HUB_DISABLE_PROGRESS_BARS`` (no tqdm download bars), Kokoro's own loguru logger disabled (no
  espeak-fallback warnings), and a ``warnings`` filter scoped to model construction only (torch's
  ``weight_norm`` ``FutureWarning``). None of that touches the terminal streams.
* Shutdown must interrupt in-flight audio. ``aclose`` sets a stop flag and calls ``sd.stop()``,
  which unblocks a running ``sd.wait()`` so the worker returns promptly instead of holding
  ``aca chat`` open (asyncio joins its executor threads at process exit) until the utterance ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import warnings

from ..errors import ConfigError
from .base import TtsEngine

# Disable Hugging Face download progress bars at the source — an env default, set before kokoro (and
# thus huggingface_hub) is imported. This keeps a cold-cache first run from drawing tqdm bars, and
# unlike a stdout redirect it never affects the main thread's terminal writes.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Kokoro's decoder output is fixed at 24 kHz (waveform samples, not a resampled buffer), so this is
# the player clock — a constant, never configuration. Playing at any other rate would pitch/speed
# shift the utterance rather than resample it.
KOKORO_SAMPLE_RATE = 24_000
# The published weights repo. Passing it explicitly stops Kokoro's KPipeline/KModel from printing a
# "Defaulting repo_id ..." warning to stdout from the worker thread.
_REPO_ID = "hexgrad/Kokoro-82M"

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
        device: str | None = None,
    ) -> None:
        if not voice:
            raise ConfigError("tts.voice is required for tts.provider 'kokoro'")
        self._voice = voice
        self._lang_code = _resolve_lang_code(voice, lang_code)
        self._speed = speed
        self._device = device
        self._pipeline = None  # lazily built on first speak (loads the model)
        self._stop = threading.Event()  # set by aclose() to abandon in-flight/queued synthesis

    async def speak(self, text: str) -> None:
        text = text.strip()
        if text and not self._stop.is_set():
            await asyncio.to_thread(self._render_and_play, text)

    async def aclose(self) -> None:
        # Interrupt any in-flight utterance so the worker thread returns promptly (and process exit
        # isn't held open joining it): stop the stream, then let the render loop see the flag.
        self._stop.set()
        with contextlib.suppress(Exception):
            import sounddevice as sd

            sd.stop()
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
            # Silence Kokoro's own loguru records (e.g. the espeak G2P-fallback warning) at their
            # source, so nothing is written to stderr from the worker thread. Targets kokoro's
            # logger only; it never reassigns sys.stdout/sys.stderr.
            with contextlib.suppress(Exception):
                from loguru import logger

                logger.disable("kokoro")
            # torch's weight_norm emits FutureWarnings while the model is built. Filter them for the
            # construction call only — a warnings filter, not a stream redirect, so the terminal is
            # untouched and playback below runs with warnings behaving normally.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Explicit repo_id suppresses Kokoro's default-repo warning print.
                self._pipeline = KPipeline(lang_code=self._lang_code, repo_id=_REPO_ID)
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
            if self._stop.is_set():  # aclose() was called — abandon the rest of the utterance
                break
            # Kokoro yields (graphemes, phonemes, audio) tuples on older versions and Result objects
            # with an ``.audio`` attribute on newer ones — accept both.
            audio = chunk[2] if isinstance(chunk, tuple) else chunk.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):  # a torch tensor -> numpy on CPU
                audio = audio.detach().cpu().numpy()
            samples = np.asarray(audio, dtype="float32")
            sd.play(samples, KOKORO_SAMPLE_RATE, device=self._device)
            sd.wait()  # sd.stop() from aclose() unblocks this so shutdown isn't held open
