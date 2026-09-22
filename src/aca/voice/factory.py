"""Build the voice pipeline from ``config.Voice`` (provider + VAD selection, fail-fast).

Mirrors the LLM/embedding factories: ``fake``/``energy`` are the deterministic offline defaults and
need nothing; a real provider or neural VAD fails fast here with a clear, actionable error (missing
extra) rather than deep inside the capture loop. Frame-count derivation for the segmenter lives here
so the segmenter itself stays a pure, unit-testable state machine.
"""

from __future__ import annotations

from ..config import Voice
from ..errors import ConfigError
from .segmenter import Segmenter
from .transcriber import FakeTranscriber, Transcriber
from .vad import EnergyVad, SpeechDetector

_TRANSCRIBER_ALIASES = {"whisper": "faster-whisper", "faster_whisper": "faster-whisper", "mock": "fake"}
_VAD_ALIASES = {"rms": "energy"}


def _frames(seconds: float, frame_seconds: float) -> int:
    return int(round(seconds / frame_seconds)) if frame_seconds > 0 else 0


def resolve_transcriber_provider(provider: str) -> str:
    """Canonical provider name after aliasing (e.g. ``whisper`` -> ``faster-whisper``)."""
    return _TRANSCRIBER_ALIASES.get(provider, provider)


def build_transcriber(config: Voice) -> Transcriber:
    """Construct the transcriber for the configured provider."""
    provider = resolve_transcriber_provider(config.provider)
    if provider == "fake":
        return FakeTranscriber()
    if provider == "faster-whisper":
        from .whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(
            config.model, device=config.device, compute_type=config.compute_type,
            language=config.language, beam_size=config.beam_size,
        )
    raise ConfigError(f"unknown voice.provider {config.provider!r} (known: fake, faster-whisper)")


def build_speech_detector(config: Voice) -> SpeechDetector:
    """Construct the VAD speech detector for the configured strategy."""
    vad = _VAD_ALIASES.get(config.vad, config.vad)
    if vad == "energy":
        return EnergyVad(config.vad_threshold)
    if vad == "silero":
        from .silero import SileroVad

        return SileroVad(sample_rate=config.sample_rate)
    raise ConfigError(f"unknown voice.vad {config.vad!r} (known: energy, silero)")


def build_segmenter(config: Voice) -> Segmenter:
    """Derive frame counts from the configured timings and construct the segmenter."""
    fs = config.frame_seconds
    return Segmenter(
        start_frames=_frames(config.start_seconds, fs),
        silence_frames=_frames(config.silence_seconds, fs),
        min_speech_frames=_frames(config.min_utterance_seconds, fs),
        max_frames=_frames(config.max_utterance_seconds, fs),
        pre_roll_frames=_frames(config.pre_roll_seconds, fs),
    )
