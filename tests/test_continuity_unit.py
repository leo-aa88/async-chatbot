"""Unit tests for the continuity corpus scorer (DESIGN §27): pure confusion-matrix logic.

No reducer — synthetic ``DecisionOutcome``s isolate the two-sided scoring (persistence recall,
restraint rate, and the balanced-accuracy headline). The corpus cases that drive a real scenario
live in ``tests/adversarial/test_continuity_corpus.py``.
"""

from __future__ import annotations

import pytest

from aca.eval.continuity import (
    ContinuityCategory,
    DecisionOutcome,
    as_worth,
    format_continuity,
    score,
)

_CAT = ContinuityCategory.CONTINUATION


def _o(should_speak: bool, did_speak: bool, category: ContinuityCategory = _CAT) -> DecisionOutcome:
    return DecisionOutcome("c", category, should_speak=should_speak, did_speak=did_speak)


def test_outcome_kind_quadrants():
    assert _o(True, True).kind == "correct_continuation"
    assert _o(True, False).kind == "missed_continuation"
    assert _o(False, False).kind == "correct_abandonment"
    assert _o(False, True).kind == "intrusion"
    assert _o(True, True).correct and not _o(True, False).correct


def test_two_sided_rates_and_balanced_accuracy():
    # 3 should-speak (2 spoke) -> recall 2/3; 2 should-silence (1 silent) -> abandonment 1/2.
    sc = score([
        _o(True, True), _o(True, True), _o(True, False),
        _o(False, False), _o(False, True),
    ])
    assert sc.continuation_recall == 2 / 3
    assert sc.abandonment_rate == 1 / 2
    assert sc.balanced_accuracy == (2 / 3 + 1 / 2) / 2
    assert sc.accuracy == 3 / 5


def test_rates_are_none_without_cases_on_a_side():
    # Only should-speak cases: the restraint side has no evidence, so its rate — and the balanced
    # headline that depends on it — must be None, never a misleading 0/1.
    sc = score([_o(True, True), _o(True, False)])
    assert sc.continuation_recall == 1 / 2
    assert sc.abandonment_rate is None
    assert sc.balanced_accuracy is None


def test_by_category_rollup_flags_failures():
    sc = score([
        _o(True, True, ContinuityCategory.CONTINUATION),
        _o(False, True, ContinuityCategory.SHIFT),          # an intrusion
        _o(True, False, ContinuityCategory.RESUMPTION),     # a missed continuation
    ])
    rows = sc.by_category()
    assert rows["continuation"] == {"decisions": 1, "correct": 1, "missed_continuation": 0, "intrusion": 0}
    assert rows["shift"]["intrusion"] == 1 and rows["shift"]["correct"] == 0
    assert rows["resumption"]["missed_continuation"] == 1
    assert "interruption" not in rows  # categories with no decisions are omitted


def test_as_worth_derives_a_gating_oracle_from_labels():
    # The corpus's own labels become gating's worth(candidate_id) — one source of truth, no re-encoding.
    outcomes = [
        DecisionOutcome("c1", _CAT, should_speak=True, did_speak=True, candidate_id="advance"),
        DecisionOutcome("c2", _CAT, should_speak=False, did_speak=True, candidate_id="old_thread"),
    ]
    worth = as_worth(outcomes)
    assert worth("advance") is True
    assert worth("old_thread") is False


def test_as_worth_is_strict_on_unknown_candidates():
    # "No label" is not "low-value": an unknown lookup must raise, not default to False, or missing
    # corpus coverage would silently flatter the gating scores (over-speech / suppression precision).
    worth = as_worth([DecisionOutcome("c", _CAT, should_speak=True, did_speak=True, candidate_id="x")])
    with pytest.raises(KeyError):
        worth("never_labelled")


def test_as_worth_rejects_missing_and_duplicate_candidate_ids():
    # candidate_id is required to build the oracle...
    with pytest.raises(ValueError):
        as_worth([DecisionOutcome("c", _CAT, should_speak=True, did_speak=True)])
    # ...and must be unique — should_speak is candidate-in-context, so a duplicate would collapse two
    # conflicting labels into one (last-wins) silently.
    with pytest.raises(ValueError):
        as_worth([
            DecisionOutcome("a", _CAT, should_speak=True, did_speak=True, candidate_id="dup"),
            DecisionOutcome("b", _CAT, should_speak=False, did_speak=False, candidate_id="dup"),
        ])


def test_format_continuity_renders_headline_and_flags():
    lines = format_continuity(score([_o(False, True, ContinuityCategory.SHIFT)]))
    text = "\n".join(lines)
    assert "continuity corpus: 1 decisions" in text
    assert "balanced accuracy" in text
    assert "shift" in text and "intrusion=1" in text
    # An n/a rate renders without crashing (no should-speak cases here).
    assert "n/a" in text
