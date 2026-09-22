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
from .turn_assembler import VoiceTurnAssembler
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
        assembler: VoiceTurnAssembler | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._segmenter = segmenter
        self._transcriber = transcriber
        self._on_transcript = on_transcript
        self._min_chars = min_chars
        self._on_error = on_error
        # When set, transcribed utterances feed the assembler (which commits whole *turns*), and
        # trailing silence in the frame stream drives its floor-yield decision (DESIGN §36). When
        # None, the legacy behavior holds: each utterance is emitted directly as one transcript.
        self._assembler = assembler
        self._running = False

    async def run(self) -> None:
        self._running = True
        silence_seconds = 0.0  # trailing acoustic silence since the human last spoke (frame stream)
        try:
            async for frame in self._source.frames():
                if not self._running:
                    break
                speech = self._detector.is_speech(frame)
                silence_seconds = 0.0 if speech else silence_seconds + frame.duration_seconds
                utterance = self._segmenter.push(frame, speech)
                if utterance is not None:
                    await self._emit(utterance)
                # Drive the assembler's floor-yield decision from measured silence — never wall-clock
                # after ASR, so Whisper latency is not mistaken for the human pausing (DESIGN §36).
                if self._assembler is not None and not speech:
                    await self._assembler.on_silence(silence_seconds)
            tail = self._segmenter.flush()
            if tail is not None:
                await self._emit(tail)
            if self._assembler is not None:
                await self._assembler.flush()
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
            if len(text) < self._min_chars:
                return
            if self._assembler is not None:
                # Join into the in-progress turn; a commit happens later on floor-yield (on_silence).
                await self._assembler.add(text, utterance.duration_seconds)
            else:
                await self._on_transcript(text)
        except Exception as exc:
            await self._report("transcription", exc)

    async def _report(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            await self._on_error(stage, exc)
