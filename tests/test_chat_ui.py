"""Unit tests for the single-threaded chat line editor (no worker thread, no readline)."""

from __future__ import annotations

import io

import pytest

from aca.cli.chat_ui import ChatUI


def _feed(ui: ChatUI, text: str) -> None:
    for ch in text:
        ui._handle_char(ch)


@pytest.mark.asyncio
async def test_printable_chars_build_buffer_and_enter_submits(monkeypatch):
    ui = ChatUI()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    _feed(ui, "hello")
    assert ui._buffer == "hello"
    _feed(ui, "\n")
    assert ui._buffer == ""  # buffer cleared on submit
    assert await ui._queue.get() == "hello"


@pytest.mark.asyncio
async def test_backspace_edits_buffer(monkeypatch):
    ui = ChatUI()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    _feed(ui, "worl")
    _feed(ui, "\x7f")  # backspace
    _feed(ui, "d")
    assert ui._buffer == "word"


@pytest.mark.asyncio
async def test_ctrl_d_on_empty_line_signals_eof(monkeypatch):
    ui = ChatUI()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    _feed(ui, "\x04")
    assert await ui._queue.get() is None  # EOF sentinel


@pytest.mark.asyncio
async def test_ctrl_d_with_text_is_ignored(monkeypatch):
    ui = ChatUI()
    monkeypatch.setattr("sys.stdout", io.StringIO())
    _feed(ui, "hi\x04")
    assert ui._queue.empty()  # not EOF while there's input
    assert ui._buffer == "hi"


@pytest.mark.asyncio
async def test_message_preserves_in_progress_input(monkeypatch):
    ui = ChatUI()
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    _feed(ui, "half typed")
    ui.print_message("an autonomous thought")
    rendered = out.getvalue()
    assert "[agent] an autonomous thought" in rendered
    # The in-progress input is redrawn after the message (not lost or garbled).
    assert rendered.rstrip().endswith("> half typed")
    assert ui._buffer == "half typed"
