"""Audio frame primitive for the voice pipeline.

A frame is a small fixed-size chunk of 16-bit signed little-endian mono PCM — the unit the VAD
classifies and the segmenter accumulates. Deliberately dependency-free (stdlib ``array`` only) so
the segmentation core is unit-testable without ``numpy`` or an audio backend; the real transcriber
converts the accumulated PCM to whatever tensor its model wants.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

_INT16_FULL_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One chunk of 16-bit signed little-endian mono PCM at ``sample_rate`` Hz."""

    pcm: bytes
    sample_rate: int

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // 2

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate if self.sample_rate else 0.0

    def rms(self) -> float:
        """Root-mean-square amplitude normalized to ``[0, 1)`` (0 for an empty frame).

        A cheap, robust loudness proxy for the energy VAD: silence sits near 0, speech well above
        it. Computed from the raw int16 samples with the stdlib ``array`` module (no numpy).
        """
        if not self.pcm:
            return 0.0
        samples = array.array("h")
        samples.frombytes(self.pcm[: self.sample_count * 2])  # ignore a dangling odd byte
        if not samples:
            return 0.0
        mean_square = sum(s * s for s in samples) / len(samples)
        return math.sqrt(mean_square) / _INT16_FULL_SCALE
