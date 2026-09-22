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

    @property
    def enabled(self) -> bool:
        return self._engine.enabled and not self._failed

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
            self._queue.put_nowait(spoken)

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
            text = await self._queue.get()
            if self._failed:  # latched off after an earlier failure: drop without attempting
                continue
            try:
                await self._engine.speak(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a TTS failure is surfaced once, then latches speech off
                self._failed = True
                self._notify_error(exc)

    def _notify_error(self, exc: BaseException) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:  # a broken error handler must never take the speaker task down
            pass
