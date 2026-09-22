"""Single-threaded terminal line editor for ``aca chat``.

The chat client must do two things to the same terminal: read the human's keystrokes and print
unsolicited agent messages that arrive at any moment. The previous approach read input on a
worker thread (``input()``/GNU readline) while printing from the event loop — two threads writing
the same terminal, and readline is not thread-safe, so redraws corrupted the input line. No mutex
can fix that: readline's internal writes happen inside C code we can't lock.

This editor removes the race at the source: it reads keystrokes on the event loop via
``loop.add_reader`` and prints agent messages from that same loop, so keystroke echo and message
output are handled by a single thread and can never interleave. It does its own minimal line
editing (printable chars, Backspace, Enter, Ctrl-D) in cbreak mode, and always restores the
terminal on exit.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import sys
from collections.abc import AsyncIterator, Callable

try:  # POSIX terminal control; the agent's IPC is POSIX-oriented anyway.
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX
    termios = None
    tty = None

PROMPT = "> "


class ChatUI:
    """An event-loop-driven prompt: yields submitted lines, prints messages without clobbering."""

    def __init__(self, prompt: str = PROMPT, stamp: Callable[[], str] | None = None) -> None:
        self._prompt = prompt
        # Optional callable returning a formatted local timestamp; when set, a submitted input line
        # is redrawn with that stamp so the transcript shows when the human typed (symmetry with the
        # stamps on agent messages). Formatting/timezone stay in the CLI — the editor is display-only.
        self._stamp = stamp
        self._buffer = ""
        # How many physical terminal rows the current prompt+input occupies. A long line wraps
        # across several rows, and clearing only the cursor's row leaves the wrapped remainder on
        # screen (duplicated on the next redraw). Tracking the row count lets us clear the whole
        # block. Kept in sync on every echo/redraw.
        self._rendered_rows = 1
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._fd: int | None = None
        self._old_attrs = None
        self._reader_added = False
        # Incremental decoder so a multi-byte UTF-8 char split across os.read() calls is
        # reassembled instead of each half becoming U+FFFD.
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        # Escape-sequence state: None normally, "start" after ESC, "csi" while consuming CSI/SS3.
        self._esc: str | None = None

    def start(self) -> None:
        self._fd = sys.stdin.fileno()
        if termios is not None and os.isatty(self._fd):
            try:
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)  # char-at-a-time, no kernel echo; SIGINT still works
            except termios.error:  # pragma: no cover - unusual terminal
                self._old_attrs = None
        asyncio.get_running_loop().add_reader(self._fd, self._on_readable)
        self._reader_added = True
        self._draw_prompt()

    def stop(self) -> None:
        if self._reader_added and self._fd is not None:
            try:
                asyncio.get_running_loop().remove_reader(self._fd)
            except (ValueError, RuntimeError):  # pragma: no cover
                pass
            self._reader_added = False
        if self._old_attrs is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None
        sys.stdout.write("\n")
        sys.stdout.flush()

    async def lines(self) -> AsyncIterator[str]:
        """Yield submitted lines until EOF (Ctrl-D on an empty line, or stdin closed)."""
        while True:
            line = await self._queue.get()
            if line is None:
                return
            yield line

    def print_message(self, text: str, at: str | None = None) -> None:
        """Print an out-of-band agent message, preserving the in-progress input line.

        ``at`` is an already-formatted local timestamp (formatting/timezone live in the CLI, so the
        editor stays display-only); it is shown before the message when provided.
        """
        self._clear_block()  # wipe the whole (possibly wrapped) input block, not just one row
        stamp = f"[{at}] " if at else ""
        sys.stdout.write(f"{stamp}[agent] {text}\n")
        self._draw_prompt()

    def print_status(self, text: str) -> None:
        """Print an out-of-band *client* status line (not agent speech), preserving the input line.

        Used for local notices like a TTS failure — rendered without the ``[agent]`` prefix so it is
        never mistaken for something the agent said.
        """
        self._clear_block()
        sys.stdout.write(f"[client] {text}\n")
        self._draw_prompt()

    # --- internals -----------------------------------------------------------------------
    def _submit_line(self) -> None:
        """Finalize the current input line: echo it (stamped, if configured) and queue it."""
        self._clear_block()
        if self._stamp is not None and self._buffer:
            # Redraw the completed line with a leading timestamp, then a newline.
            sys.stdout.write(f"[{self._stamp()}] " + self._prompt + self._buffer + "\n")
        else:
            sys.stdout.write(self._prompt + self._buffer + "\n")
        sys.stdout.flush()
        line, self._buffer = self._buffer, ""
        self._queue.put_nowait(line)
        self._draw_prompt()

    def _cols(self) -> int:
        """Current terminal width; a huge value when stdout is not a tty (tests) so nothing wraps."""
        try:
            return os.get_terminal_size(sys.stdout.fileno()).columns or 80
        except (OSError, ValueError):
            return 1_000_000

    def _rows_for(self, cells: int) -> int:
        width = self._cols()
        return max(1, (cells + width - 1) // width)

    def _clear_block(self) -> None:
        """Erase the current prompt+input, spanning every wrapped row (cursor assumed at its end).

        Moves up to the first row of the block, returns to column 0, and clears to end of screen.
        (A line that ends exactly on the terminal width is the one ambiguous case for cursor row;
        the common long-line wrap is handled correctly.)
        """
        if self._rendered_rows > 1:
            sys.stdout.write(f"\033[{self._rendered_rows - 1}A")
        sys.stdout.write("\r\033[J")

    def _draw_prompt(self) -> None:
        """Draw the prompt+input on a freshly-cleared line and remember how many rows it spans."""
        sys.stdout.write(self._prompt + self._buffer)
        self._rendered_rows = self._rows_for(len(self._prompt) + len(self._buffer))
        sys.stdout.flush()

    def _on_readable(self) -> None:
        try:
            data = os.read(self._fd, 4096)
        except OSError:  # pragma: no cover
            data = b""
        if not data:  # stdin closed
            self._queue.put_nowait(None)
            return
        self._feed_bytes(data)

    def _feed_bytes(self, data: bytes) -> None:
        for ch in self._decoder.decode(data):
            self._handle_char(ch)

    def _handle_char(self, ch: str) -> None:
        if self._esc is not None:
            self._consume_escape(ch)
            return
        if ch == "\x1b":  # ESC: start of an arrow/Home/End/function-key sequence — swallow it
            self._esc = "start"
        elif ch in ("\n", "\r"):
            self._submit_line()
        elif ch in ("\x7f", "\b"):  # Backspace / Delete
            if self._buffer:
                self._clear_block()  # clear the old (possibly wrapped) line before shrinking it
                self._buffer = self._buffer[:-1]
                self._draw_prompt()  # full redraw: correct for any character width
        elif ch == "\x04":  # Ctrl-D: EOF only on an empty line
            if not self._buffer:
                self._queue.put_nowait(None)
        elif ch == "\x03":  # Ctrl-C, if delivered as a byte rather than SIGINT
            raise KeyboardInterrupt
        elif ch >= " ":  # printable character (echoed at its own terminal width)
            self._buffer += ch
            sys.stdout.write(ch)
            # Keep the row count current as the line grows and wraps, so a later out-of-band
            # message or submit clears every row (not just the cursor's).
            self._rendered_rows = self._rows_for(len(self._prompt) + len(self._buffer))
            sys.stdout.flush()

    def _consume_escape(self, ch: str) -> None:
        """Swallow an ANSI escape sequence (arrow keys, Home/End, Delete, function keys).

        Discarding the whole sequence keeps its bytes (``[``, ``A``, ``~``, ...) out of the input
        buffer, instead of them falling through as literal printable characters.
        """
        if self._esc == "start":
            # CSI ("ESC [") or SS3 ("ESC O") introduce a multi-char sequence; anything else is a
            # lone ESC or a two-char Meta combo — done after this char.
            self._esc = "csi" if ch in ("[", "O") else None
        elif "\x40" <= ch <= "\x7e":  # a CSI/SS3 final byte (@ .. ~) terminates the sequence
            self._esc = None
