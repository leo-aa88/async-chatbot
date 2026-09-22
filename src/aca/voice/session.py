"""Voice session: wire capture -> VAD -> segmenter -> transcriber -> caller.

The orchestration seam. It owns no policy beyond "one complete utterance becomes one transcript";
what to *do* with a transcript (inject it as a ``HumanMessage``) is the caller's business, passed
in as ``on_transcript``. Keeping ingress out of here means the session has no dependency on the IPC
client and stays trivially testable with a fake source and fake transcriber.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .capture import AudioSource
from .segmenter import Segmenter, Utterance
from .transcriber import Transcriber
from .vad import SpeechDetector

OnTranscript = Callable[[str], Awaitable[None]]
OnError = Callable[[str, Exception], Awaitable[None]]


class VoiceSession:
    """Drive the pipeline until the audio source is exhausted or :meth:`stop` is called.

    The session runs as a background task, so an unhandled exception here would be a *silent*
    death — the prompt stays up while capture is gone. It therefore isolates the two failure
    surfaces: a per-utterance error (transcribe or ingress) is reported via ``on_error`` and the
    capture loop continues; a capture/device error (raised when the source opens the mic) is
    reported and ends the session, since the stream can't continue. ``on_error`` receives a stage
    label and the exception; when unset, errors are swallowed (test double behavior).
    """

    def __init__(
        self,
        source: AudioSource,
        detector: SpeechDetector,
        segmenter: Segmenter,
        transcriber: Transcriber,
        on_transcript: OnTranscript,
        *,
        min_chars: int = 1,
        on_error: OnError | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._segmenter = segmenter
        self._transcriber = transcriber
        self._on_transcript = on_transcript
        self._min_chars = min_chars
        self._on_error = on_error
        self._running = False

    async def run(self) -> None:
        self._running = True
        try:
            async for frame in self._source.frames():
                if not self._running:
                    break
                utterance = self._segmenter.push(frame, self._detector.is_speech(frame))
                if utterance is not None:
                    await self._emit(utterance)
            tail = self._segmenter.flush()
            if tail is not None:
                await self._emit(tail)
        except Exception as exc:  # capture/device failure — report and end (stream can't continue)
            await self._report("capture", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._source.close()

    async def _emit(self, utterance: Utterance) -> None:
        self._detector.reset()
        # A transcribe or ingress error must not kill capture: report it and keep listening.
        try:
            transcript = await self._transcriber.transcribe(utterance.pcm(), utterance.sample_rate)
            text = transcript.text.strip()
            if len(text) >= self._min_chars:
                await self._on_transcript(text)
        except Exception as exc:
            await self._report("transcription", exc)

    async def _report(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            await self._on_error(stage, exc)
