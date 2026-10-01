"""The em-dash rewriter (dashes.py), used for the tsundere persona's replies."""

from __future__ import annotations

import pytest

from aca.workers.llm.dashes import without_dashes


@pytest.mark.parametrize(("text", "expected"), [
    ("Keep it a likelihood—not a verdict.", "Keep it a likelihood, not a verdict."),
    ("Yeah, I got it—the “thanks ChatGPT” was sarcastic.", "Yeah, I got it, the “thanks ChatGPT” was sarcastic."),
    ("could help — as a cue — really", "could help, as a cue, really"),
    ("It's not— ...whatever. Keep going.", "It's not... whatever. Keep going."),
    ("It's not—", "It's not..."),
    ('"Wait—" she said', '"Wait..." she said'),
    ("— first\n— second", "first\nsecond"),
    ("this – that", "this, that"),
    ("pages 10–20", "pages 10–20"),  # a range keeps its en dash
    ("Wow—.", "Wow."),
    ("No dashes here.", "No dashes here."),
])
def test_dashes_become_ordinary_punctuation(text, expected):
    assert without_dashes(text) == expected


def test_code_blocks_are_left_as_written():
    text = "Fine—here:\n```python\nx = 'a—b'\n```\nDone—mostly."
    assert without_dashes(text) == "Fine, here:\n```python\nx = 'a—b'\n```\nDone, mostly."
