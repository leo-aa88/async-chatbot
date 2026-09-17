"""Cheap, deterministic text similarity for topic de-duplication (DESIGN 12.3).

The reducer must decide — deterministically, with no embeddings and no LLM call — whether a newly
enriched summary is a near-duplicate of an existing topic. This uses stdlib ``difflib`` over a
normalized (lowercased, punctuation-stripped, whitespace-collapsed) form, so the same inputs
always score the same (replay-safe). It is intentionally conservative: it reliably catches
near-identical phrasings, not loose paraphrases — that would need real embeddings.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, keep only alphanumeric tokens, join with single spaces."""
    return " ".join(_WORD.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two summaries (0 if either is empty after normalization)."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()
