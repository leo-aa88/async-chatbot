"""Turn segmentation: carve a frame stream into whole utterances (the VAD boundary logic).

This is the load-bearing "give Wolfy conversational turns" rule. We do *not* stream partial
transcripts as messages; we wait for a complete utterance — a run of speech bounded by silence —
and transcribe it as one unit, then inject it as a single ``HumanMessage``. This module owns only
the boundary decision and is a pure, deterministic state machine (frame counts in, whole
utterances out) with no clock, no I/O, and no dependencies, so it is exhaustively unit-testable.

    silence   "so I was thinking..."   silence
    ────────██████████████████████████────────
            ↑ onset (start_frames)     ↑ end (silence_frames of hangover)

Onset needs ``start_frames`` consecutive speech frames (debounces a single spurious blip). A
``pre_roll`` of frames kept before onset is prepended so the first phoneme isn't clipped. The
utterance closes after ``silence_frames`` consecutive non-speech frames (the hangover), or is
force-cut at ``max_frames`` so a stuck-open mic can't buffer without bound. An utterance shorter
than ``min_speech_frames`` of actual speech is dropped as a blip.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .audio import AudioFrame


@dataclass(frozen=True, slots=True)
class Utterance:
    """A complete spoken turn: the ordered frames from onset (incl. pre-roll) through end."""

    frames: tuple[AudioFrame, ...]
    sample_rate: int

    def pcm(self) -> bytes:
        return b"".join(f.pcm for f in self.frames)

    @property
    def duration_seconds(self) -> float:
        return sum(f.duration_seconds for f in self.frames)


class Segmenter:
    """Stateful onset/offset detector. Feed frames with :meth:`push`; drain the tail with
    :meth:`flush`. Both return a completed :class:`Utterance` or ``None``.

    Frame counts (not seconds) are passed in so the machine stays clock-free; the factory derives
    them from ``config.voice`` timings and the frame size.
    """

    def __init__(
        self,
        *,
        start_frames: int,
        silence_frames: int,
        min_speech_frames: int,
        max_frames: int,
        pre_roll_frames: int,
    ) -> None:
        self._start_frames = max(1, start_frames)
        self._silence_frames = max(1, silence_frames)
        self._min_speech_frames = max(1, min_speech_frames)
        self._max_frames = max(1, max_frames)
        self._pre_roll_frames = max(0, pre_roll_frames)
        # A rolling window of the most recent pre-speech frames, large enough to hold the pre-roll
        # context *plus* the whole onset run, so the seed buffer always includes both — even when
        # pre-roll is zero (the onset speech frames must never be lost).
        self._context: deque[AudioFrame] = deque(maxlen=self._pre_roll_frames + self._start_frames)
        self._in_speech = False
        self._buffer: list[AudioFrame] = []
        self._speech_run = 0        # consecutive speech frames seen while still in silence (onset)
        self._trailing_silence = 0  # consecutive non-speech frames while in speech (offset)
        self._speech_count = 0      # total speech frames captured in the current utterance

    def push(self, frame: AudioFrame, is_speech: bool) -> Utterance | None:
        """Consume one classified frame; return a finished utterance when a turn just closed."""
        if not self._in_speech:
            self._context.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._start_frames:
                self._begin_utterance()
            return None

        self._buffer.append(frame)
        if is_speech:
            self._speech_count += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        if self._trailing_silence >= self._silence_frames or len(self._buffer) >= self._max_frames:
            return self._finish()
        return None

    def flush(self) -> Utterance | None:
        """Close any in-progress utterance (end of the stream). Resets the machine."""
        if self._in_speech:
            return self._finish()
        self._reset_silence()
        return None

    # --- internals -----------------------------------------------------------------------
    def _begin_utterance(self) -> None:
        # Seed the buffer with up to ``pre_roll_frames`` of pre-speech context plus the confirmed
        # onset run, so the utterance starts a beat before speech was confirmed and no leading
        # phoneme is lost — regardless of the pre-roll size.
        self._in_speech = True
        keep = self._pre_roll_frames + self._speech_run
        context = list(self._context)
        self._buffer = context[-keep:] if keep > 0 else []
        self._speech_count = self._speech_run
        self._trailing_silence = 0

    def _finish(self) -> Utterance | None:
        frames = tuple(self._buffer)
        speech_count = self._speech_count
        sample_rate = frames[0].sample_rate if frames else 0
        self._reset_silence()
        if speech_count < self._min_speech_frames:
            return None  # a blip, not a turn — drop it
        return Utterance(frames=frames, sample_rate=sample_rate)

    def _reset_silence(self) -> None:
        self._in_speech = False
        self._buffer = []
        self._speech_run = 0
        self._trailing_silence = 0
        self._speech_count = 0
        self._context.clear()
