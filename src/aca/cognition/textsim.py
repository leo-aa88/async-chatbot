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

# Negation / reversal markers. A single one of these is exactly the edit that flips a summary's
# meaning while preserving nearly all of its lexical overlap ("add" -> "remove", "wants" -> "does
# not want"). If one summary carries such a marker and the other does not, they mean opposite
# things and must NOT be merged even at high lexical similarity. This is a deliberately small,
# fixed list — it targets the common single-word-negation failure, not general paraphrase/meaning
# detection (which is deferred to embeddings). Matched as whole words on the lowercased text so
# contractions ("don't", "won't") and stems ("removing", "disabled") are caught.
_NEGATION = re.compile(
    r"\b(?:no|not|never|none|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|"
    r"wouldn't|shouldn't|couldn't|no\s+longer|"
    r"stop\w*|remov\w*|disabl\w*|abandon\w*|cancel\w*|drop\w*|reject\w*|declin\w*|"
    r"dislik\w*|unhappy|against|without|instead\s+of)\b"
)


def normalize(text: str) -> str:
    """Lowercase, keep only alphanumeric tokens, join with single spaces."""
    return " ".join(_WORD.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two summaries (0 if either is empty after normalization)."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def has_negation(text: str) -> bool:
    """Whether the text carries a negation/reversal marker (see ``_NEGATION``)."""
    return _NEGATION.search(text.lower()) is not None


def polarity_conflict(a: str, b: str) -> bool:
    """True when exactly one of the two summaries is negated — they likely mean opposite things.

    Vetoes an otherwise-high lexical match so "add dark mode" and "remove dark mode" (or "wants X"
    and "does not want X") never merge into one reinforced topic.
    """
    return has_negation(a) != has_negation(b)
