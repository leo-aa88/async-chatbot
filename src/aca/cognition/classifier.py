"""Conservative rule-based human-message classification (DESIGN 13, 20).

The classifier is deliberately **recall-biased for tasks** (DESIGN 20.1): the costly error is
silently ignoring a real task, not answering unnecessarily. Ambiguous task/social input routes
to the response-required path. A re-prompt after intentional silence is forced onto the
response-required path and flagged as possible prior misclassification (DESIGN 13.4).

This is heuristics-first (DESIGN 28.3). A learned classifier is a later addition once logs
provide a labeled mistake set.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..domain.enums import MessageClass

_WORD = re.compile(r"[a-z0-9']+")

# Very short acknowledgements safe to treat as trivial for *embedding* purposes only. The raw
# event is still persisted in every case (DESIGN 12.1).
_TRIVIAL_ACKS = frozenset(
    {"ok", "k", "kk", "yeah", "yep", "yup", "no", "nope", "lol", "haha", "ty", "thx", "nice"}
)

_ACK_TOKENS = _TRIVIAL_ACKS | frozenset(
    {"cool", "fair", "sure", "great", "thanks", "gotcha", "right", "makes", "sense", "word"}
)
_CLOSER_TOKENS = frozenset(
    {"bye", "goodbye", "goodnight", "gn", "cya", "gtg", "later", "night"}
)
# Confident social questions that do not demand a task response.
_SOCIAL_QUESTION_PREFIXES = (
    "how are you",
    "how's it going",
    "hows it going",
    "what's up",
    "whats up",
    "wyd",
    "how was your",
    "how have you been",
)
_TASK_KEYWORDS = frozenset(
    {
        "explain", "fix", "debug", "rewrite", "refactor", "implement", "write", "build",
        "create", "generate", "find", "solve", "compute", "calculate", "translate",
        "summarize", "review", "analyze", "error", "bug", "stack", "trace", "exception",
        "why", "how", "what", "which", "where", "when",
    }
)
_IMPERATIVE_VERBS = frozenset(
    {
        "explain", "fix", "debug", "rewrite", "refactor", "implement", "write", "build",
        "create", "generate", "find", "solve", "compute", "calculate", "translate",
        "summarize", "review", "analyze", "show", "list", "make", "add", "remove", "give",
        "tell", "check", "run", "help",
    }
)
_REPROMPT_TOKENS = frozenset({"?", "??", "hello?", "hi?", "you there?", "u there?", "still there?"})


@dataclass(frozen=True, slots=True)
class ClassificationContext:
    """Signals about the immediately preceding turn, used for re-prompt recovery."""

    previous_human_text: str | None = None
    previous_turn_was_silent: bool = False
    seconds_since_previous_human: float | None = None


@dataclass(frozen=True, slots=True)
class Classification:
    """The outcome of cheap classification."""

    message_class: MessageClass
    response_required: bool
    embedding_eligible: bool
    is_reprompt: bool
    possible_prior_miss: bool


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _is_single_emoji(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 4:
        return False
    return all(unicodedata.category(ch).startswith("S") or not ch.isalnum() for ch in stripped)


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]; used as a cheap lexical proximity signal."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_trivial(text: str) -> bool:
    """Obviously trivial content that may skip embedding (DESIGN 12.2, 28.4)."""
    stripped = text.strip()
    if not stripped:
        return True
    if _is_single_emoji(stripped):
        return True
    tokens = _tokens(stripped)
    return len(tokens) <= 1 and (not tokens or tokens[0] in _TRIVIAL_ACKS)


def _looks_like_reprompt(text: str, context: ClassificationContext) -> bool:
    stripped = text.strip().lower()
    if not context.previous_turn_was_silent:
        return False
    if stripped in _REPROMPT_TOKENS or stripped.rstrip("?!") in {"hello", "hi", "you there"}:
        return True
    if stripped in {"?", "??", "???"}:
        return True
    if context.previous_human_text is not None:
        similarity = jaccard_similarity(text, context.previous_human_text)
        if similarity >= 0.6:
            return True
    return False


def _base_class(text: str) -> MessageClass:
    stripped = text.strip()
    lowered = stripped.lower()
    tokens = _tokens(stripped)

    if not tokens:
        return MessageClass.LOW_INFORMATION

    # Acknowledgements / closers first (short, low-obligation).
    if len(tokens) <= 3:
        if tokens[0] in _CLOSER_TOKENS or (len(tokens) == 2 and tokens[-1] in _CLOSER_TOKENS):
            return MessageClass.CONVERSATION_CLOSER
        if all(t in _ACK_TOKENS for t in tokens):
            return MessageClass.ACKNOWLEDGEMENT

    is_question = stripped.endswith("?")
    starts_imperative = tokens[0] in _IMPERATIVE_VERBS

    if starts_imperative and not is_question:
        return MessageClass.DIRECT_TASK

    if is_question:
        if any(lowered.startswith(prefix) for prefix in _SOCIAL_QUESTION_PREFIXES):
            return MessageClass.SOCIAL_QUESTION
        # Any other question is treated as a task question (recall-biased, DESIGN 13.2, 20.1).
        return MessageClass.TASK_QUESTION

    # Non-question statements: a task keyword or an imperative verb anywhere still leans task.
    if starts_imperative or any(t in _TASK_KEYWORDS for t in tokens[:3]):
        return MessageClass.DIRECT_TASK

    return MessageClass.HIGH_INFORMATION if len(tokens) >= 4 else MessageClass.STATEMENT


def classify(text: str, context: ClassificationContext | None = None) -> Classification:
    """Classify a human message conservatively."""
    context = context or ClassificationContext()

    if _looks_like_reprompt(text, context):
        # Re-prompts always force the response-required path (DESIGN 13.4, invariant 13).
        return Classification(
            message_class=MessageClass.REPROMPT,
            response_required=True,
            embedding_eligible=not is_trivial(text),
            is_reprompt=True,
            possible_prior_miss=True,
        )

    message_class = _base_class(text)
    return Classification(
        message_class=message_class,
        response_required=message_class.response_required,
        embedding_eligible=not is_trivial(text),
        is_reprompt=False,
        possible_prior_miss=False,
    )
