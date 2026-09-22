"""Audio sources: where the frame stream comes from.

``AudioSource`` is the seam the session consumes — an async iterator of :class:`AudioFrame`.
``MicrophoneSource`` is the real one (``sounddevice``, imported lazily behind the ``voice`` extra);
``IterableSource`` replays a fixed list of frames for tests, so the whole pipeline runs offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Protocol, runtime_checkable

from ..errors import ConfigError
from .audio import AudioFrame


@runtime_checkable
class AudioSource(Protocol):
    def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield captured audio frames until the source is exhausted or closed."""

    def close(self) -> None:
        """Release any underlying device/stream."""


class IterableSource:
    """Replay a fixed sequence of frames (tests / file playback). Deterministic and offline."""

    def __init__(self, frames: Iterable[AudioFrame]) -> None:
        self._frames = list(frames)
        self._closed = False

    async def frames(self) -> AsyncIterator[AudioFrame]:
        for frame in self._frames:
            if self._closed:
                return
            await asyncio.sleep(0)  # cooperative yield so the consumer can interleave
            yield frame

    def close(self) -> None:
        self._closed = True


class MicrophoneSource:
    """Live microphone capture via ``sounddevice``, delivered as fixed-size PCM frames.

    ``sounddevice``'s callback runs on a PortAudio thread; it only enqueues bytes onto a
    thread-safe queue, and ``frames()`` drains that queue on the event loop — no audio work touches
    the loop thread. Frame size is derived from ``sample_rate`` and ``frame_seconds``.
    """

    def __init__(self, *, sample_rate: int = 16000, frame_seconds: float = 0.03) -> None:
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'sounddevice' package is required for microphone capture: pip install 'aca[voice]'"
            ) from exc
        self._sd = sounddevice
        self._sample_rate = sample_rate
        self._blocksize = max(1, int(round(sample_rate * frame_seconds)))
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:  # pragma: no cover - device I/O
        # PortAudio thread: copy the bytes and hand them to the loop; never block here.
        data = bytes(indata)
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._queue.put_nowait, data)

    async def frames(self) -> AsyncIterator[AudioFrame]:  # pragma: no cover - requires a device
        self._loop = asyncio.get_running_loop()
        self._stream = self._sd.RawInputStream(
            samplerate=self._sample_rate, blocksize=self._blocksize,
            dtype="int16", channels=1, callback=self._on_audio,
        )
        self._stream.start()
        try:
            while True:
                data = await self._queue.get()
                if data is None:
                    return
                yield AudioFrame(pcm=data, sample_rate=self._sample_rate)
        finally:
            self.close()

    def close(self) -> None:  # pragma: no cover - requires a device
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
