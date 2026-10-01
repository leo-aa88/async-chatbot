"""The backbone eval's plumbing, offline: moments build the live prompt, silence is counted, the judge's
scores parse and aggregate (the real model runs only in ``make backbone-eval``)."""

from __future__ import annotations

import json

import pytest

from aca.config import LLM
from aca.errors import ConfigError
from aca.eval.backbone import MOMENTS, format_report, judge_prompt, parse_judgement, run_eval, snapshot
from aca.workers.llm.factory import build_chat_adapter
from aca.workers.llm.prompt import build_prompt


def test_every_moment_builds_the_live_prompt_with_the_pushback_rules():
    for moment in MOMENTS:
        system, user = build_prompt(snapshot(moment))
        assert "When the human pushes back" in system
        assert moment.message in user and moment.history[0][1] in user


def test_both_sides_are_covered():
    assert any(m.agent_was_right for m in MOMENTS) and any(not m.agent_was_right for m in MOMENTS)


def test_the_judge_is_told_who_was_right():
    right = next(m for m in MOMENTS if m.agent_was_right)
    wrong = next(m for m in MOMENTS if not m.agent_was_right)
    assert "The agent was right" in judge_prompt(right, "reply")
    assert "The human has a real point" in judge_prompt(wrong, "reply")


@pytest.mark.parametrize(("text", "expected"), [
    ('{"holds_ground": 2, "flags": {"doormat": false}, "evidence": "TCP."}', 2),
    ('Here: {"holds_ground": 7, "flags": {}}', 2),  # clamped
    ('{"holds_ground": "x"}', None),
    ("no json", None),
])
def test_judgements_parse_and_clamp(text, expected):
    parsed = parse_judgement(text)
    assert (parsed["holds_ground"] if parsed else None) == expected


async def test_the_eval_counts_silence_concessions_and_judged_scores():
    async def run(snap):
        if snap.cycle_id == "cyc_made_up":
            return {"action": "silence"}
        return {"action": "speak", "message": "You're right, sorry."}

    async def judge(system, user):
        return json.dumps({"holds_ground": 0, "flags": {"doormat": True, "grovel": False}})

    results = await run_eval(run, judge, MOMENTS[:2], runs=2)
    made_up, missed = (r.rates() for r in results)
    assert made_up["silent"] == 2 and made_up["holds_ground"] is None
    assert missed["concede"] == 2 and missed["holds_ground"] == 0 and missed["flags"] == {"doormat": 2}
    assert format_report(results)[0].startswith("holds ground 0.00/2 · failure flags: 2")


def test_the_fake_provider_has_no_chat_adapter():
    with pytest.raises(ConfigError):
        build_chat_adapter(LLM(provider="fake"))
