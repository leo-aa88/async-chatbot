"""Serialize speech so utterances never overlap, without blocking the chat event loop.

Agent messages arrive at arbitrary moments on the chat client's event loop. Synthesis + playback is
slow and blocking, so the controller runs it on a background task draining an ``asyncio.Queue``:
``submit`` is non-blocking (the message still prints instantly) and queued utterances are spoken
one at a time in arrival order. A disabled engine makes the whole controller a no-op, so the client
can wire it unconditionally.

Two failure rules keep TTS from ever disrupting the chat client:

* A synthesis/playback failure is reported once and then **latches** the controller off — a dead
  audio device or a missing package must not append an error for every subsequent message.
* If the ``on_error`` callback itself raises, that is swallowed: the speaker task must not die (a
  dead task would leave later ``submit`` calls to queue forever and quit to re-raise on ``aclose``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from .base import TtsEngine
from .text import clean_for_speech


class SpeechController:
    def __init__(
        self, engine: TtsEngine, *, on_error: Callable[[BaseException], None] | None = None
    ) -> None:
        self._engine = engine
        self._on_error = on_error
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._failed = False  # latched after the first engine failure; no further speech attempts
        self._inflight = 0  # submitted-but-not-yet-finished utterances (queued + the one speaking)

    @property
    def enabled(self) -> bool:
        return self._engine.enabled and not self._failed

    @property
    def speaking(self) -> bool:
        """Whether the agent currently holds the audio floor (an utterance is queued or playing).

        The half-duplex gate: while this is true the voice session drops microphone input so the
        agent's own speech is never captured and committed as a human turn (self-hearing). Counts
        from :meth:`submit` (synchronous) through the end of playback, so there is no window where a
        just-submitted utterance reads as not-speaking.
        """
        return self._inflight > 0

    def start(self) -> None:
        """Spawn the background speaker (no-op when speech is disabled or already started)."""
        if self._engine.enabled and self._task is None:
            self._task = asyncio.ensure_future(self._run())

    def submit(self, text: str) -> None:
        """Queue an agent message to be spoken. Non-blocking; safe to call from message handlers."""
        if not self.enabled:
            return
        spoken = clean_for_speech(text)
        if spoken:
            self._inflight += 1  # bumped here (sync) so `speaking` is true the instant we queue
            self._queue.put_nowait(spoken)

    def interrupt(self) -> None:
        """Barge-in: stop the current utterance and drop everything queued behind it.

        The human is reclaiming the floor. In-flight playback is abandoned via the engine's
        non-terminal :meth:`~aca.tts.base.TtsEngine.stop` (the speaker task stays alive and later
        submits play normally), and pending utterances are discarded so the agent doesn't resume a
        stale backlog after being cut off. Non-blocking; a no-op when speech is disabled.
        """
        if not self.enabled:
            return
        while True:  # drop the backlog; each drained item balances its submit() increment
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._inflight -= 1
        self._engine.stop()  # the in-flight utterance (still counted) unwinds and decrements in _run

    async def aclose(self) -> None:
        """Stop the speaker and release engine resources. Any in-flight utterance is abandoned.

        ``engine.aclose`` runs first so an engine that can interrupt playback (Kokoro calls
        ``sd.stop()``) unblocks its worker thread; then the task is cancelled. Cancelling alone does
        not stop a thread already inside ``to_thread``, which is why the engine, not cancellation,
        owns interruption.
        """
        await self._engine.aclose()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while True:
            text = await self._queue.get()  # outside the try: a cancel here dequeued nothing
            try:
                if self._failed:  # latched off after an earlier failure: drop without attempting
                    continue
                await self._engine.speak(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a TTS failure is surfaced once, then latches speech off
                self._failed = True
                self._notify_error(exc)
            finally:
                self._inflight -= 1  # one decrement per dequeued item (spoken, dropped, or cut off)

    def _notify_error(self, exc: BaseException) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:  # a broken error handler must never take the speaker task down
            pass
