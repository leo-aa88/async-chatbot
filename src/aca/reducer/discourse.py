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

import math
from enum import Enum

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
# (DESIGN §34.4), not solved here — and they fail in BOTH directions, not only toward silence:
#   * a bare declarative first subject is missed -> no focus -> the gate fails open and an unrelated
#     (orphan) candidate CAN be voiced;
#   * a declarative mid-conversation shift is missed -> stale focus retained -> over-suppresses the
#     new-subject candidate; and a confirmation question ("Is that clear?") can become the focus ->
#     over-suppresses for one IDLE band.
# Faithful subject detection needs an LLM-proposed relation (§31.15), not another text heuristic.
_SUBJECT_CLASSES = frozenset({
    MessageClass.DIRECT_TASK, MessageClass.TASK_QUESTION, MessageClass.SOCIAL_QUESTION,
})

# Presence pings and confirmation/check-in questions carry response obligation but assert no new
# subject. The ingress class does not separate them: after a delivered agent turn `"You there?"` is
# a `TASK_QUESTION`, not a `REPROMPT` (which requires prior silence). So the *utterance* is filtered
# here (a small, conservative, subtractive set — not proof of subject). Unlisted variants remain a
# documented over-suppression residual, deferred to the LLM-proposed relation (DESIGN §34.4, §31.15).
_CHECKIN_PHRASES = frozenset({
    "you there", "are you there", "you still there", "still there", "you there still",
    "anyone there", "anybody there", "you around", "you online", "you awake", "hello", "hey", "hi",
    "is that clear", "does that make sense", "make sense", "makes sense", "got it", "right", "ok",
    "okay", "clear", "understood", "capisce", "yeah", "sure",
})


def _is_checkin(text: str) -> bool:
    """Whether the whole utterance is a presence ping / confirmation check (no new subject)."""
    normalized = " ".join(text.strip().lower().split()).rstrip("?!.").strip()
    return normalized in _CHECKIN_PHRASES


def is_focus_setting(text: str, message_class: MessageClass) -> bool:
    """Approximate whether a human turn introduces a new conversational subject (DESIGN §34.4).

    Focus-setting is a separate axis from response obligation: a task/question introduces something
    to discuss, but a `REPROMPT` and a presence/confirmation check-in ("You there?", "Is that
    clear?") carry obligation without a new subject, so they are excluded — the check-in filter runs
    on the *utterance* because the ingress class alone can't tell `"You there?"` (a `TASK_QUESTION`
    after a delivered turn) from a real question. Plain declaratives never set the focus (undecidable
    vs an acknowledgement). It remains an approximation: an unlisted check-in/confirmation phrasing
    is a documented over-suppression residual, deferred to §31.15.
    """
    return message_class in _SUBJECT_CLASSES and not _is_checkin(text)


def _subject_vector(ctx: ReducerContext, memory_id: str | None):
    """The stored ``(model_version, vector)`` of a subject memory (focus or closed), or ``None``."""
    if not memory_id:
        return None
    return ctx.stores.memory.embeddings_by_memory([memory_id]).get(memory_id)


def gate_active(ctx: ReducerContext, now) -> bool:
    """The gate enforces only while conversation mode is `IDLE` (recomputed now, not the stored
    value, since mode drifts with time). `ACTIVE` is mode-suppressed already; `DORMANT` stays open
    for resurfacing (§34.7)."""
    conversation = ctx.stores.state.load_conversation()
    return infer_mode_for(ctx, conversation, now) is ConversationMode.IDLE


def _affinity(a: list[float], b: list[float]) -> float | None:
    """A numerically stable cosine in [-1, 1], or ``None`` when the pair is not comparable.

    ``None`` (⇒ UNJUDGED, fail open, §34.5) for a mismatched length, an empty/zero vector, a
    non-finite input, or a non-finite result. Each vector is scaled by its max-abs component before
    the dot/norms, so magnitudes that would overflow (`1e308`) or underflow (`1e-300`) a plain
    ``sum(x*x)`` are handled: opposite huge vectors give −1 (ORPHAN), identical ones give +1
    (CONTINUE) — never a silent NaN that slips into the BRIDGE band. Replaces the earlier guard,
    whose validity predicate could disagree with the cosine it gated.
    """
    if len(a) != len(b) or not a:
        return None
    if not all(math.isfinite(x) for x in a) or not all(math.isfinite(y) for y in b):
        return None
    sa = max(abs(x) for x in a)
    sb = max(abs(y) for y in b)
    if sa == 0.0 or sb == 0.0:  # a zero vector — no direction to compare
        return None
    dot = sum((x / sa) * (y / sb) for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum((x / sa) ** 2 for x in a))
    nb = math.sqrt(sum((y / sb) ** 2 for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    val = dot / (na * nb)
    if not math.isfinite(val):
        return None
    return max(-1.0, min(1.0, val))


def assess_against(
    ctx: ReducerContext, subject_memory_id: str | None, kind: str, candidate_id: str
) -> DiscourseRelation:
    """Relation of a candidate to a given *subject* memory by cosine band. UNJUDGED when either vector
    is missing, from a different model, or not usably comparable (drift-/degeneracy-safe; never an
    orphan, §34.5). Used for both the current focus and the closed subject (§34.11)."""
    bridge = ctx.config.memory.discourse_bridge_cosine
    if bridge <= 0.0:
        return DiscourseRelation.UNJUDGED  # gate disabled
    subject = _subject_vector(ctx, subject_memory_id)
    cand = candidate_vector(ctx, kind, candidate_id)
    if subject is None or cand is None or subject[0] != cand[0]:
        return DiscourseRelation.UNJUDGED
    affinity = _affinity(subject[1], cand[1])
    if affinity is None:  # not a measurement (empty/zero/non-finite/overflow/mismatch) — fail open
        return DiscourseRelation.UNJUDGED
    # Check the ORPHAN boundary (bridge) FIRST so the suppression cut is exactly discourse_bridge_
    # cosine regardless of the CONTINUE threshold — the label can't silently move the gate (§34.5).
    # (Config also enforces bridge <= continue, so this ordering and that invariant agree.)
    if affinity < bridge:
        return DiscourseRelation.ORPHAN
    if affinity >= ctx.config.memory.discourse_continue_cosine:
        return DiscourseRelation.CONTINUE
    return DiscourseRelation.BRIDGE


def assess(ctx: ReducerContext, kind: str, candidate_id: str) -> DiscourseRelation:
    """Relation of a candidate to the *current focus* subject (§34.5)."""
    focus_id = ctx.stores.state.load_conversation().focus_memory_id
    return assess_against(ctx, focus_id, kind, candidate_id)


def is_discourse_orphan(ctx: ReducerContext, kind: str, candidate_id: str, now) -> bool:
    """True iff a proactive candidate should be muted by the discourse gate (IDLE only).

    Three cases while IDLE:
    * a **focus is set** — the candidate is an ORPHAN of it (cosine below the bridge, §34.5): the
      open-focus rule — *unrelated* is muted.
    * **no focus, a subject was just closed** (§34.11) — the gate **inverts** against the closed
      subject: a candidate *related* to it (CONTINUE/BRIDGE) is muted (don't reopen a resolved
      thread), while a genuinely *unrelated* (ORPHAN, or UNJUDGED) thought still speaks. This is the
      one place a missing focus can suppress, and only for the closed thread itself; scoped to `IDLE`
      (`DORMANT` leaves the gate inactive, so a later lull still resurfaces, §34.7).
    * **no focus, nothing closed** — the declarative residual, or a fresh conversation: fail open.

    Fail-open everywhere else: inactive outside `IDLE`; UNJUDGED (missing/degenerate/cross-model
    vector, or gate disabled) is never an orphan (§34.5, §34.7).
    """
    if not gate_active(ctx, now):
        return False
    conversation = ctx.stores.state.load_conversation()
    if conversation.focus_memory_id is not None:
        return assess_against(ctx, conversation.focus_memory_id, kind, candidate_id) is DiscourseRelation.ORPHAN
    if conversation.closed_focus_memory_id is not None:
        # Inverted rule: renag *the closed thread* only — related is muted, unrelated speaks.
        return assess_against(ctx, conversation.closed_focus_memory_id, kind, candidate_id) in (
            DiscourseRelation.CONTINUE, DiscourseRelation.BRIDGE)
    return False


def advancement_suppresses(ctx: ReducerContext, relation: str | None, now) -> bool:
    """Whether a proposed advancement relation mutes the proactive speak (§35.4, invariant 42a).

    Suppress-only, and scoped by *kind*:
    * ``REPEAT`` mutes in **any** mode — never restate something already said (mirrors §16.2).
    * ``ORPHAN`` mutes **only while a live thread actually exists** — mode `IDLE` (the discourse
      gate's window) *and* a focus subject set (`focus_memory_id is not None`). "Off the current
      thread" is meaningful only against a current subject, so both parts are required: `DORMANT`
      has no live window, and `IDLE` with no focus (e.g. only check-ins were exchanged, so no turn
      ever set the focus) has no thread either — in both an "orphan" is a legitimate *resurfacing*
      that §34.7 preserves, and the LLM's `ORPHAN` label must not suppress it. The guard is focus
      *existence*, not its vector: an LLM semantic judgement must not hinge on whether an embedding
      happened to be available (that fail-open belongs to §34.5's cosine gate, not here).
    Forward moves (`ADVANCE`/`EVIDENCE`/`REVISE`/`CLOSE`/`REOPEN`) and a missing/unrecognized
    relation (`None`) fall open to §34, so the current v0.7 worker is gated exactly as before.
    """
    if relation == "REPEAT":
        return True
    if relation == "ORPHAN":
        conversation = ctx.stores.state.load_conversation()
        return conversation.focus_memory_id is not None and gate_active(ctx, now)
    return False


def resolve_focus_transition(
    ctx: ReducerContext, transition: str | None, memory_id: str | None, prior_focus: str | None
) -> tuple[bool, str | None]:
    """Resolve a human-turn focus transition (§35.3, invariant 42b) into an optional **override** of
    the §34.4 deterministic provisional focus. Returns ``(override, new_focus_memory_id)``:

    * ``None`` / unrecognized ⇒ ``(False, _)`` — **no override**: the deterministic provisional focus
      (written at reduction as a pre-result placeholder) stands. This is the v0.7 backward-compat
      floor (§35.9 cases 3, 8): the current worker emits no ``focus`` and is unaffected.
    * ``KEEP`` ⇒ ``(True, prior_focus)`` — the subject is unchanged *from before this turn*, so the
      focus reverts to the pre-turn focus. This is what makes `KEEP` **fix** the check-in residual:
      an unlisted check-in ("was that understandable?") that §34.4 wrongly anchored to this turn's
      memory is reverted to the real subject (§35.3), not merely left on the wrong anchor.
    * ``CLEAR`` ⇒ ``(True, None)`` — nothing is on the floor; the gate then fails open (a deliberate
      widening, §35.6), not a suppression.
    * ``REPLACE`` naming a **validated existing provisional-memory** id ⇒ ``(True, id)``; any other
      id (a ``topic_id``, an unknown/missing id) is **treated as KEEP** ⇒ ``(True, prior_focus)``
      (§35.9 case 6). Validation is provisional-memory *existence* — the sole LLM input to the focus,
      writing no other state; it is deliberately independent of whether an embedding exists (that
      cosine fail-open is §34.5's concern), so the focus is a *memory* the §34 gate can resolve.
    """
    if transition is None:
        return False, None
    if transition == "CLEAR":
        return True, None
    if transition == "REPLACE" and memory_id and ctx.stores.memory.get_memory(memory_id) is not None:
        return True, memory_id
    return True, prior_focus  # KEEP, or a REPLACE that failed validation (treated as KEEP)
