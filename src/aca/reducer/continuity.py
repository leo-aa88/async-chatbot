"""Semantic continuity check (DESIGN 6, 16): don't re-voice, reworded, a thought just voiced.

Shared by two current-state checkpoints, the same way the mode gate is re-evaluated at both:

* the wake handler, pre-dispatch — a near-repeat candidate is dispatched enrichment-only, not spoken;
* the LLMResult handler, pre-outbox — re-checked against *current* state before ``create_outbound``,
  so a near-repeat that became one after dispatch (another expression landed, or a paraphrase went
  in-flight in the interim) is not voiced.

Observational metrics (advance-rate, dominance) are never consulted here — feeding them back into a
gate would be a Goodhart coupling. The signal is raw cosine against the same very-high
``topic_dedup_cosine`` used for dedup, so only genuine re-voicings are caught; an *advance* (same
subject, new content) sits below it and passes.
"""

from __future__ import annotations

from datetime import timedelta

from ..cognition import vectors
from ..domain.enums import CandidateKind
from .context import ReducerContext


def candidate_vector(ctx: ReducerContext, kind: str, candidate_id: str):
    """A candidate's ``(model_version, vector)`` — a topic via its summary embedding, a memory via
    its own — or None if it has none."""
    if kind == CandidateKind.TOPIC.value:
        return ctx.stores.memory.topic_embeddings_by_id([candidate_id]).get(candidate_id)
    if kind == CandidateKind.PROVISIONAL_MEMORY.value:
        return ctx.stores.memory.embeddings_by_memory([candidate_id]).get(candidate_id)
    return None


def is_semantic_repeat(ctx: ReducerContext, kind: str, candidate_id: str, now) -> bool:
    """Whether the candidate near-paraphrases something already expressed or in-flight.

    Compares (same-model, drift-safe) the candidate's vector against every candidate voiced within
    the repeat window *and* every candidate with a still-undelivered proactive item — so two ≥dedup
    paraphrases can't both reach the outbox before either is delivered. Cosine ≥ ``topic_dedup_cosine``
    is a repeat. A threshold of 0 disables the check; a missing/cross-model vector fails open (speak).
    """
    dedup = ctx.config.memory.topic_dedup_cosine
    if dedup <= 0.0:
        return False
    current = candidate_vector(ctx, kind, candidate_id)
    if current is None:
        return False
    since = now - timedelta(seconds=ctx.config.memory.repeat_suppression_seconds)
    prior = set(ctx.stores.memory.recent_expressions(since))
    prior |= ctx.stores.outbox.in_flight_proactive_candidates()
    for other_kind, cid in prior:
        if cid == candidate_id:
            continue  # exact re-selection is handled by pool-level repeat-suppression in selection
        other = candidate_vector(ctx, other_kind, cid)
        if other is None or other[0] != current[0]:  # missing / different model -> not comparable
            continue
        if vectors.cosine(current[1], other[1]) >= dedup:
            return True
    return False
