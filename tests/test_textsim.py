"""Unit tests for deterministic topic-similarity scoring (DESIGN 12.3)."""

from __future__ import annotations

from aca.cognition.textsim import normalize, similarity


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
