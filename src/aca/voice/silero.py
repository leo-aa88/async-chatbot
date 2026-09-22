"""Silero VAD adapter, imported lazily behind the ``voice-silero`` extra.

A neural voice-activity detector — markedly more noise-robust than the energy gate — behind the
same ``SpeechDetector`` protocol. Silero operates on fixed-size 16 kHz chunks (512 samples), so
this adapter buffers incoming PCM and averages the model's per-chunk speech probability across the
frame, comparing it to ``threshold``. Optional: selected only by ``config.voice.vad = "silero"``.
"""

from __future__ import annotations

from ..errors import ConfigError
from .audio import AudioFrame

_SILERO_CHUNK = 512  # samples the model expects per call at 16 kHz
_INT16_FULL_SCALE = 32768.0


class SileroVad:
    """A ``SpeechDetector`` backed by the ``silero-vad`` model."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        try:
            import torch  # noqa: F401
            from silero_vad import load_silero_vad
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'silero-vad' package is required for voice.vad 'silero': pip install 'aca[voice-silero]'"
            ) from exc
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model = load_silero_vad()

    def is_speech(self, frame: AudioFrame) -> bool:
        import torch

        samples = torch.frombuffer(bytearray(frame.pcm), dtype=torch.int16).float() / _INT16_FULL_SCALE
        if samples.numel() < _SILERO_CHUNK:
            return False
        probs: list[float] = []
        for start in range(0, samples.numel() - _SILERO_CHUNK + 1, _SILERO_CHUNK):
            chunk = samples[start : start + _SILERO_CHUNK]
            probs.append(float(self._model(chunk, self._sample_rate).item()))
        return bool(probs) and (sum(probs) / len(probs)) >= self._threshold

    def reset(self) -> None:
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):
            reset()
