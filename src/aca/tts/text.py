"""Normalize a delivered agent message into spoken text (pure, deterministic).

The chat client prints the agent's message verbatim; this produces the *spoken* rendering of the
same body — not the identical string. Agent prose may carry light markdown (emphasis, inline/fenced
code, links), and reading the literal punctuation aloud ("star star", "backtick") sounds wrong, so
the markup is unwrapped to its text. The pass is markdown-aware, not a blunt character strip: it
only removes markers acting as markup, so ``C#``, ``a > b`` and ``snake_case`` are left intact.
Code fences keep their inner text (a fenced-only message becomes speech, never silent). Newlines are
preserved — Kokoro splits on ``\n+`` — while runs of spaces collapse. Pure ``str -> str``: no IO.
"""

from __future__ import annotations

import re

# Fenced block ```lang\n...\n``` -> keep the inner code text, drop the delimiters and info string.
_FENCE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)
# A one-line ```...``` that isn't a real (newline-delimited) block -> keep the inner text.
_TRIPLE_INLINE = re.compile(r"```(.+?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")  # `code` -> code
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")  # ![alt](url) -> alt
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)  # **x** / __x__ -> x
# Emphasis: markers must hug non-space content, so ``a * b`` (spaced) is not treated as italics.
_ITALIC_STAR = re.compile(r"\*(?!\s)(.+?)(?<!\s)\*", re.DOTALL)
# Underscore italics only at word boundaries, so ``snake_case`` / ``a_b`` are untouched.
_ITALIC_UNDERSCORE = re.compile(r"(?<![\w])_(?!\s)(.+?)(?<!\s)_(?![\w])", re.DOTALL)
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")  # line-leading "## " only (not mid-word ``C#``)
_QUOTE = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")  # line-leading "> " only (not mid-line ``a > b``)
_SPACES = re.compile(r"[ \t]+")
_BLANKLINES = re.compile(r"\n{2,}")


def clean_for_speech(text: str) -> str:
    """Unwrap light markdown to spoken text, keeping newlines and normalizing spaces."""
    text = _FENCE.sub(r"\1", text)
    text = _TRIPLE_INLINE.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _ITALIC_STAR.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANKLINES.sub("\n", text)
    return text.strip()
