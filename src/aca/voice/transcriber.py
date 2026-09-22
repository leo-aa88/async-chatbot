"""Transcriber protocol and the deterministic fake used in tests.

The real Whisper transcriber lives in ``whisper.py`` behind an optional import; both satisfy this
protocol so the session and CLI never depend on ``faster-whisper`` being installed. The transcribe
call is ``async`` because a real model is CPU/GPU-bound and offloads to a worker thread — the same
"heavy work never runs inline on the event loop" rule the daemon follows (AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Transcript:
    """The recognized text for one utterance plus optional metadata."""

    text: str
    language: str = ""
    duration_seconds: float = 0.0


@runtime_checkable
class Transcriber(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> Transcript:
        """Transcribe 16-bit mono PCM into text."""


class FakeTranscriber:
    """Deterministic offline transcriber for tests and dry runs.

    Emits canned text without touching an audio model, so the whole voice pipeline (capture ->
    VAD -> segmenter -> transcriber -> ingress) is exercisable in CI. With ``scripted`` it returns
    successive lines per utterance; once exhausted (or when unset) it falls back to a stable string
    derived from the PCM length, so output is always reproducible.
    """

    def __init__(self, scripted: list[str] | None = None) -> None:
        self._scripted = list(scripted or [])
        self._index = 0

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Transcript:
        if self._index < len(self._scripted):
            text = self._scripted[self._index]
            self._index += 1
        else:
            samples = len(pcm) // 2
            text = f"utterance of {samples} samples"
        duration = (len(pcm) // 2) / sample_rate if sample_rate else 0.0
        return Transcript(text=text, language="en", duration_seconds=duration)
