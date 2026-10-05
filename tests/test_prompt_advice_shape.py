"""Advice is sized and spoken, and a term of endearment is never adopted (the system prompt's rules).

Measured live with gpt-5.6-luna on four advice questions and a "sweetie" jab; see the PR. These pin the
rules, so a later prompt edit can't drop them unnoticed.
"""

from __future__ import annotations

from aca.cognition.snapshot import Snapshot
from aca.workers.llm.prompt import build_prompt


def _system() -> str:
    return " ".join(build_prompt(Snapshot("c", "w", 1, "t", {}))[0].split())


def test_advice_is_a_take_and_a_few_points_not_a_list():
    system = _system()
    assert "Asked for ideas, advice, or an approach" in system
    assert "two or three points that matter most" in system
    assert "No numbered or bulleted list unless they asked for steps" in system


def test_its_example_is_off_topic_so_it_cant_be_parroted():
    # An example on the measured topic got copied word for word; this one is about deploys.
    assert "Cache the dependencies first" in _system()


def test_a_term_of_endearment_is_never_adopted():
    assert "is something to react to, not something you start calling them" in _system()
