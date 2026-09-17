"""Unit tests for deterministic topic-similarity scoring (DESIGN 12.3)."""

from __future__ import annotations

from aca.cognition.textsim import has_negation, normalize, polarity_conflict, similarity


def test_normalize_strips_case_and_punctuation():
    assert normalize("  ACA's Logs, and silence!! ") == "aca s logs and silence"


def test_near_identical_summaries_score_high():
    a = "AI chatbot experiment logs with silence scenarios"
    b = "AI chatbot experiment logs with silence and scenarios"
    assert similarity(a, b) >= 0.8


def test_unrelated_summaries_score_low():
    a = "Future robot embodiment and unexpected text-to-speech behavior"
    b = "Reducing sycophancy through non-mandatory responses"
    assert similarity(a, b) < 0.5


def test_empty_is_zero_and_identical_is_one():
    assert similarity("", "anything") == 0.0
    assert similarity("same text", "same text") == 1.0


def test_negation_swap_is_high_similarity_but_polarity_conflict():
    # The reviewer's reproduced case: opposite requests score above the 0.8 default threshold.
    a = "User wants to add dark mode to the settings page"
    b = "User wants to remove dark mode from the settings page"
    assert similarity(a, b) >= 0.8
    assert polarity_conflict(a, b) is True  # so the merge is vetoed despite the high score


def test_polarity_conflict_covers_common_reversals():
    assert polarity_conflict("enable notifications", "disable notifications") is True
    assert polarity_conflict("decided to ship it", "decided not to ship it") is True
    assert has_negation("we won't continue the project") is True
    # Same polarity (both plain, or both negated) is not a conflict.
    assert polarity_conflict("add dark mode", "add a dark theme") is False
    assert polarity_conflict("stop the sync", "no longer syncing") is False
