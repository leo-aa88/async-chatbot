"""Unit tests for the LLM action contract (DESIGN 22): data, never authority."""

from __future__ import annotations

import pytest

from aca.domain.enums import ActionKind
from aca.domain.proposals import MAX_ACTIVATION_DELTA, MAX_MESSAGE_CHARS, parse_decision
from aca.errors import ValidationError


def test_speak_requires_message():
    with pytest.raises(ValidationError):
        parse_decision({"action": "speak"})


def test_unknown_action_is_rejected_not_defaulted():
    with pytest.raises(ValidationError):
        parse_decision({"action": "delete_everything"})


def test_activation_delta_is_clamped():
    decision = parse_decision(
        {"action": "silence",
         "proposals": [{"type": "ADJUST_TOPIC_ACTIVATION", "topic_id": "t1", "delta": 999}]}
    )
    assert decision.proposals[0].fields["delta"] == MAX_ACTIVATION_DELTA


def test_non_whitelisted_proposals_are_dropped():
    decision = parse_decision(
        {"action": "silence",
         "proposals": [{"type": "WIRE_MONEY", "amount": 1000},
                       {"type": "ADJUST_TOPIC_ACTIVATION", "topic_id": "t1", "delta": 0.1}]}
    )
    assert [p.type for p in decision.proposals] == ["ADJUST_TOPIC_ACTIVATION"]


def test_message_is_truncated():
    decision = parse_decision({"action": "speak", "message": "x" * (MAX_MESSAGE_CHARS + 500)})
    assert decision.action is ActionKind.SPEAK
    assert len(decision.message) == MAX_MESSAGE_CHARS


def test_malformed_input_raises():
    with pytest.raises(ValidationError):
        parse_decision("not a dict")
    with pytest.raises(ValidationError):
        parse_decision({"action": "silence", "proposals": "nope"})


def test_enrichment_proposal_marks_useful():
    decision = parse_decision(
        {"action": "silence",
         "proposals": [{"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": "m1",
                        "topic_summary": "x"}]}
    )
    assert decision.useful_enrichment is True
