"""Voice-activity detection: per-frame speech / not-speech classification.

The ``SpeechDetector`` protocol is the seam the segmenter depends on; it says nothing about *how*
speech is detected. ``EnergyVad`` is the dependency-free default (an RMS threshold), good enough
for a desktop mic in a quiet room. A more robust neural detector (Silero) plugs in behind the same
protocol via ``factory.build_speech_detector`` without the segmenter knowing the difference.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .audio import AudioFrame


@runtime_checkable
class SpeechDetector(Protocol):
    def is_speech(self, frame: AudioFrame) -> bool:
        """Classify one frame as speech (``True``) or not (``False``)."""

    def reset(self) -> None:
        """Drop any per-utterance internal state (called between utterances)."""


class EnergyVad:
    """Threshold the frame's RMS amplitude — stateless, dependency-free, deterministic.

    Not noise-robust (a loud fan reads as speech), but zero-dependency and perfectly adequate for a
    close desktop microphone. ``threshold`` is normalized RMS in ``[0, 1]``; tune via
    ``config.voice.vad_threshold``.
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def is_speech(self, frame: AudioFrame) -> bool:
        return frame.rms() >= self._threshold

    def reset(self) -> None:  # stateless — nothing to clear
        pass
