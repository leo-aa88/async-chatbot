"""faster-whisper transcriber (CTranslate2), imported lazily behind the ``voice`` extra.

Wraps a single loaded ``WhisperModel`` and transcribes accumulated PCM. The recommended starting
point for a 4 GB GPU is ``small.en`` at ``int8_float16`` (~2 GB VRAM, English-only); the smaller,
faster ``distil-small.en`` trades a little accuracy for latency. Both load through this same class
— only ``config.voice.model`` changes.

The model is loaded once at construction (so the first utterance isn't slow) and each transcription
runs in a worker thread via ``asyncio.to_thread`` — a Whisper forward pass must never block the
client's event loop.
"""

from __future__ import annotations

import asyncio

from ..errors import ConfigError
from .transcriber import Transcript

_INT16_FULL_SCALE = 32768.0


class FasterWhisperTranscriber:
    """A ``Transcriber`` backed by ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        compute_type: str = "int8_float16",
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'faster-whisper' package is required for voice.provider 'faster-whisper': "
                "pip install 'aca[voice]'"
            ) from exc
        try:
            import numpy  # noqa: F401  (required to marshal PCM into the model)
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "the 'numpy' package is required for voice transcription: pip install 'aca[voice]'"
            ) from exc

        self._language = language or None
        self._beam_size = beam_size
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Transcript:
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE
        segments, info = self._model.transcribe(
            audio, language=self._language, beam_size=self._beam_size
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration = len(audio) / sample_rate if sample_rate else 0.0
        language = getattr(info, "language", None) or (self._language or "")
        return Transcript(text=text, language=language, duration_seconds=duration)
