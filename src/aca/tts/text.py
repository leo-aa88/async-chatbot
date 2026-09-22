"""Turn a delivered agent message into clean spoken text (pure, deterministic).

Agent utterances are prose but may carry light markdown — emphasis, code spans, links. Speaking the
literal punctuation ("star star", "backtick") sounds wrong, so the markup is stripped and
whitespace collapsed before synthesis. Pure ``str -> str``: trivially testable, no IO, no state.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)  # drop fenced code blocks entirely
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text (speak the label, not the URL)
_MARKUP = re.compile(r"[*_`#>~]+")  # inline emphasis / code / heading / quote / strike markers
_WHITESPACE = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Strip light markdown and collapse whitespace so an utterance reads naturally aloud."""
    text = _CODE_FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _MARKUP.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
