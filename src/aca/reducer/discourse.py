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


# A subject is asserted, approximately, by a structural cue that introduces something to discuss —
# a task/imperative or a question. This is a *content-free approximation*, deliberately separate
# from response obligation: `REPROMPT` ("You there?") requires a response but never asserts a new
# subject, so it is excluded. Two undecidable classes remain and are accepted, documented residuals
# (DESIGN §34.4), not solved here: a bare declarative ("The robot needs better balance" vs an ack
# "I hear you loud and clear") and a confirmation/check-in question ("Is that clear?") — neither is
# separable from a real subject deterministically, and both, if mishandled, over-suppress (silence)
# rather than voice an orphan. Faithful subject detection needs an LLM-proposed relation (§31.15).
_SUBJECT_CLASSES = frozenset({
    MessageClass.DIRECT_TASK, MessageClass.TASK_QUESTION, MessageClass.SOCIAL_QUESTION,
})


def is_focus_setting(text: str, message_class: MessageClass) -> bool:
    """Approximate whether a human turn introduces a new conversational subject (DESIGN §34.4).

    Focus-setting is a separate axis from response obligation: a task/question introduces something
    to discuss, but a `REPROMPT` (obligation without a new subject) does not and is excluded. Plain
    declaratives never set the focus (undecidable vs an acknowledgement). It is only an
    approximation — a confirmation question ("Is that clear?") is a known residual that can still
    become the focus, over-suppressing for one IDLE band; faithful detection is deferred to §31.15.
    ``text`` is unused in v0.7 but kept for that future content-aware / LLM-proposed refinement.
    """
    return message_class in _SUBJECT_CLASSES


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
