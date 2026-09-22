"""Serialize speech so utterances never overlap, without blocking the chat event loop.

Agent messages arrive at arbitrary moments on the chat client's event loop. Synthesis + playback is
slow and blocking, so the controller runs it on a background task draining an ``asyncio.Queue``:
``submit`` is non-blocking (the message still prints instantly) and queued utterances are spoken
one at a time in arrival order. A disabled engine makes the whole controller a no-op, so the client
can wire it unconditionally. A synthesis/playback failure is reported, never fatal — TTS must never
take down the chat client.
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

    @property
    def enabled(self) -> bool:
        return self._engine.enabled

    def start(self) -> None:
        """Spawn the background speaker (no-op when speech is disabled or already started)."""
        if self._engine.enabled and self._task is None:
            self._task = asyncio.ensure_future(self._run())

    def submit(self, text: str) -> None:
        """Queue an agent message to be spoken. Non-blocking; safe to call from message handlers."""
        if not self._engine.enabled:
            return
        spoken = clean_for_speech(text)
        if spoken:
            self._queue.put_nowait(spoken)

    async def aclose(self) -> None:
        """Stop the speaker and release engine resources. Any in-flight utterance is abandoned."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._engine.aclose()

    async def _run(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                await self._engine.speak(text)
            except Exception as exc:  # a TTS failure is surfaced, never fatal to the chat client
                if self._on_error is not None:
                    self._on_error(exc)
