"""Silero VAD adapter, imported lazily behind the ``voice-silero`` extra.

A neural voice-activity detector — markedly more noise-robust than the energy gate — behind the
same ``SpeechDetector`` protocol. Silero v5 operates on a *fixed* window (512 samples at 16 kHz,
256 at 8 kHz), but a capture frame is ``round(sample_rate * frame_seconds)`` — 480 samples at the
defaults, which is *smaller* than the window. So this adapter **accumulates** incoming PCM across
frames and runs the model only once a full window exists, holding the last decision for frames that
don't complete a new window. (An earlier version compared the raw frame to the window size and
returned "not speech" for every default-sized frame, silently swallowing the microphone.)

The buffering/windowing is a pure state machine (``_WindowedDetector``) with no torch dependency,
so it is unit-testable with a stub scorer and default-sized frames.
"""

from __future__ import annotations

from collections.abc import Callable

from ..errors import ConfigError
from .audio import AudioFrame

_INT16_FULL_SCALE = 32768.0
_WINDOW_SAMPLES = {16000: 512, 8000: 256}  # Silero v5's fixed input sizes


class _WindowedDetector:
    """Accumulate PCM and classify on full windows; hold the last decision between windows.

    Pure and dependency-free: ``score`` maps one window's raw int16 bytes to a speech probability.
    ``push`` returns the current speech verdict for the frame just consumed — a fresh decision when
    one or more windows completed (their probabilities averaged), otherwise the held prior verdict.
    """

    def __init__(self, window_samples: int, threshold: float, score: Callable[[bytes], float]) -> None:
        self._window_bytes = window_samples * 2  # int16
        self._threshold = threshold
        self._score = score
        self._buffer = bytearray()
        self._last = False

    def push(self, pcm: bytes) -> bool:
        self._buffer += pcm
        probs: list[float] = []
        while len(self._buffer) >= self._window_bytes:
            window = bytes(self._buffer[: self._window_bytes])
            del self._buffer[: self._window_bytes]
            probs.append(self._score(window))
        if probs:
            self._last = (sum(probs) / len(probs)) >= self._threshold
        return self._last

    def reset(self) -> None:
        self._buffer.clear()
        self._last = False


class SileroVad:
    """A ``SpeechDetector`` backed by the ``silero-vad`` model (windowed, stateful)."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        window = _WINDOW_SAMPLES.get(sample_rate)
        if window is None:
            raise ConfigError(
                f"voice.vad 'silero' supports sample_rate 8000 or 16000, got {sample_rate}"
            )
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'silero-vad' package is required for voice.vad 'silero': pip install 'aca[voice-silero]'"
            ) from exc
        self._torch = torch
        self._sample_rate = sample_rate
        self._model = load_silero_vad()
        self._detector = _WindowedDetector(window, threshold, self._score)

    def _score(self, window: bytes) -> float:  # pragma: no cover - requires torch + model
        samples = self._torch.frombuffer(bytearray(window), dtype=self._torch.int16).float() / _INT16_FULL_SCALE
        return float(self._model(samples, self._sample_rate).item())

    def is_speech(self, frame: AudioFrame) -> bool:
        return self._detector.push(frame.pcm)

    def reset(self) -> None:
        self._detector.reset()
        reset = getattr(self._model, "reset_states", None)
        if callable(reset):  # pragma: no cover - requires the model
            reset()
