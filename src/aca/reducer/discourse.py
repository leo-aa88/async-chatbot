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


# Turns that do not assert a subject even when ingress makes a RAW memory for them. The ingress
# class is insufficient (``noted``/``alright`` are STATEMENT), so this is its own token set plus the
# low-substance classes (§34.4). Conservative by design; calibration is DESIGN §31.15.
_BACKCHANNEL_TOKENS = frozenset({
    "ok", "okay", "k", "kk", "noted", "alright", "aight", "lol", "lmao", "sure", "yep", "yup",
    "yeah", "nah", "cool", "fine", "right", "gotcha", "got it", "thanks", "thx", "ty", "np",
})
_LOW_SUBSTANCE_CLASSES = frozenset({
    MessageClass.ACKNOWLEDGEMENT, MessageClass.CONVERSATION_CLOSER, MessageClass.LOW_INFORMATION,
})


def is_focus_setting(text: str, message_class: MessageClass) -> bool:
    """Whether a human turn asserts a new conversational subject (DESIGN §34.4).

    Its own conservative predicate — a low-substance class, or a turn whose whole content is a
    backchannel token, does not set the focus. Uncertainty biases toward *not* focus-setting so a
    guessed subject is never installed (the transition then defers to mode; §34.4).
    """
    if message_class in _LOW_SUBSTANCE_CLASSES:
        return False
    normalized = text.strip().lower().rstrip(".!").strip()
    return normalized not in _BACKCHANNEL_TOKENS


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
    if affinity >= ctx.config.memory.discourse_continue_cosine:
        return DiscourseRelation.CONTINUE
    if affinity >= bridge:
        return DiscourseRelation.BRIDGE
    return DiscourseRelation.ORPHAN


def is_discourse_orphan(ctx: ReducerContext, kind: str, candidate_id: str, now) -> bool:
    """True iff the gate is active (IDLE) AND the candidate is an ORPHAN of the current focus.

    Fail-open everywhere else: inactive outside `IDLE`, and UNJUDGED (missing focus/candidate
    vector, cross-model, or gate disabled) is never an orphan (§34.5, §34.7).
    """
    if not gate_active(ctx, now):
        return False
    return assess(ctx, kind, candidate_id) is DiscourseRelation.ORPHAN
