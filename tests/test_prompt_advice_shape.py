"""Prompt-contract tests: the system prompt says advice is sized and spoken, and that a term of
endearment is not an invitation to mirror it.

These check what the prompt says, not what the model does. The behavioral evidence (gpt-5.6-luna on
four advice questions and a "sweetie" jab) is in the PR; these keep a later edit from dropping the rules.
"""

from __future__ import annotations

from aca.cognition.snapshot import Snapshot
from aca.workers.llm.prompt import build_prompt


def _system() -> str:
    return " ".join(build_prompt(Snapshot("c", "w", 1, "t", {}))[0].split())


def test_prompt_requests_concise_unstructured_advice():
    system = _system()
    assert "Asked for ideas, advice, or an approach" in system
    assert "two or three points that matter most" in system
    assert "Don't default to numbered or bulleted lists unless they asked for steps, options, a list" in system


def test_prompt_advice_example_is_off_topic():
    # An example on the measured topic got copied word for word; this one is about deploys.
    assert "Cache the dependencies first" in _system()


def test_prompt_does_not_encourage_endearment_mirroring():
    system = _system()
    assert "not an invitation to mirror it back" in system
    assert "merely because they used it for you" in system
