"""Single-instance advisory lock (DESIGN 29.1, invariant 3).

Exactly one agent service may own a data directory at a time. An OS advisory lock (``flock``)
enforces this across process lifetime and is released automatically on process death — so a
crash never leaves a stale lock the way a PID file would. SQLite transactions provide state
durability; they are not held open merely as a process mutex.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

from ..errors import ServiceAlreadyRunningError

try:
    import fcntl  # POSIX
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


class SingleInstanceLock:
    """A held-open advisory file lock scoped to one agent data directory."""

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / "aca.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                raise ServiceAlreadyRunningError(
                    f"another agent service already owns {self._path.parent}"
                ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode("ascii"))
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
