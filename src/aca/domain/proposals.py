"""LLM action contract and validation (DESIGN 22).

    Language-model output is data, never authority.

A worker returns a raw dict. This module parses it into a *validated, clamped* ``LLMDecision``.
Only whitelisted proposal types survive; numeric deltas are hard-capped; message size is
bounded; unknown fields are dropped. Anything structurally invalid raises ``ValidationError``
and the caller treats it as "no authorized change" (DESIGN 22.3). This is a pure function of
its input — it performs no IO and no current-state checks (those happen later in the reducer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ValidationError
from .enums import ActionKind

# Hard caps applied to untrusted model output.
MAX_MESSAGE_CHARS = 4000
MAX_PROPOSALS = 16
MAX_ACTIVATION_DELTA = 0.5
MAX_SUMMARY_CHARS = 500
MAX_TAGS = 12
MAX_TAG_CHARS = 60

_WHITELISTED_PROPOSAL_TYPES = frozenset(
    {
        "ENRICH_PROVISIONAL_MEMORY",
        "ADJUST_TOPIC_ACTIVATION",
        "CREATE_DEFERRED_INTENT",
        "RESOLVE_DEFERRED_INTENT",
        "USER_ENGAGEMENT_OBSERVATION",
    }
)

# Advancement relation a proactive result may propose for its own message (DESIGN §35.4). Parsed
# here (validate + whitelist) only; the suppression *policy* (which values mute a speak) lives in
# the reducer. An absent or unrecognized value parses to ``None`` and the reducer falls open to the
# §34 decision — never a parse failure (§35.9 case 2), so the current v0.7 worker is unaffected.
_ADVANCEMENT_RELATIONS = frozenset(
    {"ADVANCE", "EVIDENCE", "REVISE", "CLOSE", "REOPEN", "ORPHAN", "REPEAT"}
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"proposal field {name!r} must be numeric") from exc


@dataclass(frozen=True, slots=True)
class Proposal:
    """A single whitelisted, already-clamped typed proposal."""

    type: str
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """A validated worker result: a bounded action plus clamped proposals."""

    action: ActionKind
    message: str | None
    proposals: tuple[Proposal, ...]
    useful_enrichment: bool = False
    # A proactive message's proposed relation to the current thread (§35.4); ``None`` when absent or
    # unrecognized. Advisory data the reducer gates on — never authority (invariant 42a).
    relation: str | None = None


def _clean_str(value: Any, limit: int) -> str:
    text = str(value).strip()
    return text[:limit]


def _validate_proposal(raw: Any) -> Proposal | None:
    """Validate and clamp one proposal. Returns ``None`` for a non-whitelisted type."""
    if not isinstance(raw, dict):
        raise ValidationError("each proposal must be an object")
    ptype = raw.get("type")
    if ptype not in _WHITELISTED_PROPOSAL_TYPES:
        return None  # silently drop non-whitelisted proposals (DESIGN 22.1)

    fields: dict[str, Any] = {}
    if ptype == "ENRICH_PROVISIONAL_MEMORY":
        if not raw.get("provisional_memory_id"):
            raise ValidationError("ENRICH_PROVISIONAL_MEMORY needs provisional_memory_id")
        fields["provisional_memory_id"] = str(raw["provisional_memory_id"])
        fields["topic_summary"] = _clean_str(raw.get("topic_summary", ""), MAX_SUMMARY_CHARS)
        tags = raw.get("tags", []) or []
        fields["tags"] = tuple(_clean_str(t, MAX_TAG_CHARS) for t in tags[:MAX_TAGS])
    elif ptype == "ADJUST_TOPIC_ACTIVATION":
        if not raw.get("topic_id"):
            raise ValidationError("ADJUST_TOPIC_ACTIVATION needs topic_id")
        fields["topic_id"] = str(raw["topic_id"])
        delta = _as_float(raw.get("delta", 0.0), "delta")
        fields["delta"] = _clamp(delta, -MAX_ACTIVATION_DELTA, MAX_ACTIVATION_DELTA)
    elif ptype == "CREATE_DEFERRED_INTENT":
        fields["intent"] = _clean_str(raw.get("intent", ""), MAX_SUMMARY_CHARS)
        if not fields["intent"]:
            raise ValidationError("CREATE_DEFERRED_INTENT needs a non-empty intent")
        fields["topic_id"] = (
            str(raw["topic_id"]) if raw.get("topic_id") else None
        )
        fields["provisional_memory_id"] = (
            str(raw["provisional_memory_id"]) if raw.get("provisional_memory_id") else None
        )
    elif ptype == "RESOLVE_DEFERRED_INTENT":
        if not raw.get("intent_id"):
            raise ValidationError("RESOLVE_DEFERRED_INTENT needs intent_id")
        fields["intent_id"] = str(raw["intent_id"])
    elif ptype == "USER_ENGAGEMENT_OBSERVATION":
        if not raw.get("target_action_id"):
            raise ValidationError("USER_ENGAGEMENT_OBSERVATION needs target_action_id")
        fields["target_action_id"] = str(raw["target_action_id"])
        fields["classification"] = _clean_str(raw.get("classification", ""), 64)

    return Proposal(type=str(ptype), fields=fields)


def parse_decision(raw: Any) -> LLMDecision:
    """Parse and clamp an untrusted worker result into a validated ``LLMDecision``.

    Raises ``ValidationError`` on structurally invalid input. The action string is normalized
    case-insensitively; an unrecognized action is a validation failure rather than a silent
    default, because a wrong action could otherwise masquerade as intentional silence.
    """
    if not isinstance(raw, dict):
        raise ValidationError("worker result must be an object")

    action_raw = str(raw.get("action", "")).strip().upper()
    try:
        action = ActionKind(action_raw)
    except ValueError as exc:
        raise ValidationError(f"unknown action: {action_raw!r}") from exc

    message: str | None = None
    if action == ActionKind.SPEAK:
        text = _clean_str(raw.get("message", ""), MAX_MESSAGE_CHARS)
        if not text:
            raise ValidationError("SPEAK requires a non-empty message")
        message = text

    raw_proposals = raw.get("proposals", []) or []
    if not isinstance(raw_proposals, list):
        raise ValidationError("proposals must be a list")

    proposals: list[Proposal] = []
    for raw_proposal in raw_proposals[:MAX_PROPOSALS]:
        validated = _validate_proposal(raw_proposal)
        if validated is not None:
            proposals.append(validated)

    useful_enrichment = any(
        p.type in ("ENRICH_PROVISIONAL_MEMORY", "CREATE_DEFERRED_INTENT") for p in proposals
    )
    relation_raw = str(raw.get("relation", "")).strip().upper()
    relation = relation_raw if relation_raw in _ADVANCEMENT_RELATIONS else None
    return LLMDecision(
        action=action,
        message=message,
        proposals=tuple(proposals),
        useful_enrichment=useful_enrichment or bool(raw.get("useful_enrichment", False)),
        relation=relation,
    )
