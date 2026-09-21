"""Discourse-focus gate (DESIGN §34, v0.7): is this proactive thought about the conversation *now*?

Distinct from ``continuity.py`` (which answers "did I already say this?"). This module answers
"does this candidate belong to the subject the human made this conversation about?" and, **only
while conversation mode is `IDLE`**, suppresses a proactive candidate that does not — an ORPHAN.
`ACTIVE` is already fully suppressed by the mode gate; `DORMANT` is left open so autonomous
resurfacing is preserved. Everything here is deterministic (cosine of stored vectors), computed by
code — never LLM authority (invariant 5) — and kept separate from the observational metrics.

Two independent pieces:

* the **focus predicate** (`is_focus_setting`): does a human turn assert a new subject? This is its
  own conservative decision, NOT the ingress task/social class or memory status (§34.4).
* the **affinity gate** (`is_discourse_orphan`): given the current focus subject, is a candidate an
  ORPHAN, and is the gate active (IDLE)?
"""

from __future__ import annotations

from enum import Enum

from ..cognition import vectors
from ..domain.enums import ConversationMode, MessageClass
from .context import ReducerContext
from .continuity import candidate_vector
from .support import infer_mode_for


class DiscourseRelation(str, Enum):
    CONTINUE = "continue"   # direct continuation of the live subject
    BRIDGE = "bridge"       # a related move from it
    ORPHAN = "orphan"       # no conversational reason to say this now
    UNJUDGED = "unjudged"   # no comparable vector (missing/cross-model) — never suppressed


# Structural signals from ingress that are *reliable* about subject-ness (not the length split):
# a task/question/reprompt always asserts a subject; an acknowledgement/closer/low-info never does.
_SUBJECT_CLASSES = frozenset({
    MessageClass.DIRECT_TASK, MessageClass.TASK_QUESTION, MessageClass.REPROMPT,
    MessageClass.SOCIAL_QUESTION,
})
_NON_SUBJECT_CLASSES = frozenset({
    MessageClass.ACKNOWLEDGEMENT, MessageClass.CONVERSATION_CLOSER, MessageClass.LOW_INFORMATION,
})

# For the *plain-statement* bucket (HIGH_INFORMATION / STATEMENT — the same kind of turn split only
# by length at ingress, so length is NOT a subject signal), decide independently: a turn is an
# acknowledgement iff, after dropping function/filler words, it has no content beyond acknowledgement
# words. This separates "I see your point now" (ack) from "Robots are next" (subject) without the
# ingress length class. Deliberately imperfect; calibration is DESIGN §31.15.
_ACK_WORDS = frozenset({
    "ok", "okay", "k", "kk", "yeah", "yep", "yup", "no", "nope", "nah", "lol", "haha", "lmao",
    "ty", "thx", "thanks", "nice", "cool", "fair", "sure", "great", "gotcha", "right", "makes",
    "sense", "word", "see", "point", "agreed", "understood", "true", "correct", "exactly",
    "totally", "indeed", "noted", "alright", "aight", "fine", "good", "got", "np",
})
_FILLER_WORDS = frozenset({
    "i", "you", "your", "my", "we", "they", "it", "the", "a", "an", "that", "this", "these",
    "those", "is", "are", "was", "were", "now", "then", "so", "well", "oh", "ah", "to", "of",
    "and", "but", "do", "me", "us", "am",
})


def _is_acknowledgement(text: str) -> bool:
    """Independent (non-length) check: does this plain statement carry no subject beyond ack words?

    True when every content token (after removing function/filler words) is an acknowledgement
    word — or there is no content at all. False as soon as a genuinely new content word appears.
    """
    tokens = [t for t in text.strip().lower().replace("'", " ").split() if t.isalpha()]
    content = [t for t in tokens if t not in _FILLER_WORDS]
    if not content:
        return True
    return all(t in _ACK_WORDS for t in content)


def is_focus_setting(text: str, message_class: MessageClass) -> bool:
    """Whether a human turn asserts a new conversational subject (DESIGN §34.4).

    Its own discourse decision, not the ingress length class: structural task/question classes
    assert a subject; acknowledgement/closer/low-info never do; and a *plain statement* asserts a
    subject only if it is not an acknowledgement by content (`_is_acknowledgement`). Uncertainty in
    the plain-statement bucket resolves toward the content check, so an acknowledgement phrased as a
    long statement ("I see your point now") does not become the focus while a short real shift
    ("Robots are next") does.
    """
    if message_class in _NON_SUBJECT_CLASSES:
        return False
    if message_class in _SUBJECT_CLASSES:
        return True
    return not _is_acknowledgement(text)


def _focus_vector(ctx: ReducerContext):
    conversation = ctx.stores.state.load_conversation()
    fid = conversation.focus_memory_id
    if not fid:
        return None
    return ctx.stores.memory.embeddings_by_memory([fid]).get(fid)


def gate_active(ctx: ReducerContext, now) -> bool:
    """The gate enforces only while conversation mode is `IDLE` (recomputed now, not the stored
    value, since mode drifts with time). `ACTIVE` is mode-suppressed already; `DORMANT` stays open
    for resurfacing (§34.7)."""
    conversation = ctx.stores.state.load_conversation()
    return infer_mode_for(ctx, conversation, now) is ConversationMode.IDLE


def assess(ctx: ReducerContext, kind: str, candidate_id: str) -> DiscourseRelation:
    """Relation of a candidate to the current focus subject by cosine band. UNJUDGED when either
    vector is missing or from a different model (drift-safe; never an orphan, §34.5)."""
    bridge = ctx.config.memory.discourse_bridge_cosine
    if bridge <= 0.0:
        return DiscourseRelation.UNJUDGED  # gate disabled
    focus = _focus_vector(ctx)
    cand = candidate_vector(ctx, kind, candidate_id)
    if focus is None or cand is None or focus[0] != cand[0]:
        return DiscourseRelation.UNJUDGED
    affinity = vectors.cosine(focus[1], cand[1])
    # Check the ORPHAN boundary (bridge) FIRST so the suppression cut is exactly discourse_bridge_
    # cosine regardless of the CONTINUE threshold — the label can't silently move the gate (§34.5).
    # (Config also enforces bridge <= continue, so this ordering and that invariant agree.)
    if affinity < bridge:
        return DiscourseRelation.ORPHAN
    if affinity >= ctx.config.memory.discourse_continue_cosine:
        return DiscourseRelation.CONTINUE
    return DiscourseRelation.BRIDGE


def is_discourse_orphan(ctx: ReducerContext, kind: str, candidate_id: str, now) -> bool:
    """True iff the gate is active (IDLE) AND the candidate is an ORPHAN of the current focus.

    Fail-open everywhere else: inactive outside `IDLE`, and UNJUDGED (missing focus/candidate
    vector, cross-model, or gate disabled) is never an orphan (§34.5, §34.7).
    """
    if not gate_active(ctx, now):
        return False
    return assess(ctx, kind, candidate_id) is DiscourseRelation.ORPHAN
