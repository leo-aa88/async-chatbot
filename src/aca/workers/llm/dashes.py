"""No em dashes in anything the tsundere persona says (the tsundere.chat owner's rule, 2026-09-29).

The dash is the chatbot's signature punctuation, and the prompt alone doesn't stop it, so every reply
of a character persona is rewritten here before the reducer sees it (the default persona's replies
are untouched). A cut-off ("It's not— ...whatever") becomes an ellipsis, a dash at the start of a
line is dropped, and any other dash (a splice, or a pair around an aside) becomes a comma. A spaced
en dash used the same way ("this – that") goes too; a numeric range ("10–20") keeps its en dash,
and code inside ``` fences is left exactly as written.
"""

from __future__ import annotations

import re

_DASH = r"(?:—|(?<=\s)–(?=\s))"
_CUT_OFF = re.compile(rf"[ \t]*{_DASH}[ \t]*(?:\.\.\.|…)[ \t]*")  # "not— ...whatever"
_TRAILING = re.compile(rf"[ \t]*{_DASH}[ \t]*(?=$|[\"'”’)\]])", re.M)  # "It's not—" at the end
_LEADING = re.compile(rf"^([ \t]*){_DASH}[ \t]*", re.M)  # a line that opens with a dash
_SPLICE = re.compile(rf"[ \t]*{_DASH}[ \t]*")
_DOUBLED = re.compile(r",[ \t]*([,.;:!?])")  # "word,." left by a dash before punctuation
_FENCE = re.compile(r"(```.*?(?:```|$))", re.S)


def without_dashes(text: str) -> str:
    # Odd pieces of the split are code blocks: kept verbatim.
    parts = _FENCE.split(text)
    return "".join(p if i % 2 else _prose(p) for i, p in enumerate(parts)).strip()


def _prose(text: str) -> str:
    text = _CUT_OFF.sub("... ", text)
    text = _TRAILING.sub("...", text)
    text = _LEADING.sub(r"\1", text)
    text = _SPLICE.sub(", ", text)
    return _DOUBLED.sub(r"\1", text)
