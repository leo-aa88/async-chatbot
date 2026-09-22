"""Client-side voice perception for ``aca listen`` (DESIGN 13, 28.2).

Speech is just another human utterance. This package captures a microphone, segments it into
conversational turns with a lightweight VAD, transcribes each turn with Whisper, and hands the
text back to the caller — which injects it through the *same* durable ingress as typed input
(a ``HumanMessage`` tagged ``input_mode="voice"``). Nothing here is imported by the daemon core:
it lives entirely on the client, so it cannot affect the reducer's determinism or replay.

    microphone -> AudioFrame stream -> VAD -> Segmenter -> Utterance -> Transcriber -> text

The pure core (``audio``, ``vad``, ``segmenter``, ``transcriber.FakeTranscriber``) has no heavy
dependencies and is fully unit-tested. Real capture (``sounddevice``) and real transcription
(``faster-whisper``) are optional extras imported lazily, so ``import aca.voice`` never fails.
"""

from __future__ import annotations

from .audio import AudioFrame
from .segmenter import Segmenter, Utterance
from .transcriber import FakeTranscriber, Transcriber, Transcript
from .vad import EnergyVad, SpeechDetector

__all__ = [
    "AudioFrame",
    "Segmenter",
    "Utterance",
    "Transcript",
    "Transcriber",
    "FakeTranscriber",
    "SpeechDetector",
    "EnergyVad",
]
