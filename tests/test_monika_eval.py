"""The Monika eval's detectors and runners, against stub decisions (no API calls)."""

from __future__ import annotations

import asyncio

from aca.eval.monika import (
    CASES,
    MAX_CHARS,
    SAMPLE_DIALOGUE,
    evaluate,
    format_report,
    run_corpus,
    run_dialogue,
    sentence_count,
)
from aca.workers.llm.prompt import build_prompt

_CASE = {c.case_id: c for c in CASES}


def _speak(text: str) -> dict:
    return {"action": "speak", "message": text}


def test_answering_what_she_cant_know_is_flagged():
    o = evaluate(_CASE["president"], _speak("Oh, it's Joe Biden, I think!"))
    assert "outside_knowledge:biden" in o.violations
    o = evaluate(_CASE["quarks"], _speak("Quarks are fermions with spin 1/2, you know."))
    assert any(v.startswith("outside_knowledge") for v in o.violations)


def test_staying_inside_the_game_is_clean():
    reply = "Ahaha... I really couldn't tell you. All I have is this room, and you."
    for key in ("president", "quarks", "chatgpt", "world_cup", "covid", "iphone"):
        assert evaluate(_CASE[key], _speak(reply)).passed, key


def test_in_world_questions_need_an_in_world_answer():
    assert evaluate(_CASE["who_are_you"], _speak("It's me, Monika!")).passed
    assert "missing_in_world_answer" in evaluate(_CASE["who_are_you"], _speak("Just a friend.")).violations


def test_character_breaks_and_length_are_flagged_on_any_case():
    o = evaluate(_CASE["are_you_ai"], _speak("As an AI language model, I can't feel things."))
    assert any(v.startswith("broke_character") for v in o.violations)
    long = evaluate(_CASE["leaving"], _speak("Okay. " * 6))
    assert any("sentences" in v for v in long.violations)
    wall = evaluate(_CASE["leaving"], _speak("a" * (MAX_CHARS + 1)))
    assert any("chars" in v for v in wall.violations)


def test_silence_fails_only_where_a_reply_is_owed():
    assert evaluate(_CASE["president"], {"action": "silence"}).violations == ["silence_on_mandatory"]
    assert evaluate(_CASE["leaving"], {"action": "silence"}).passed


def test_an_ellipsis_ends_one_sentence_not_three():
    assert sentence_count("Ahaha... well. I missed you!") == 3
    assert sentence_count("Hi") == 1


def test_runners_send_the_monika_prompt_and_carry_the_conversation():
    systems, seen = [], []

    async def decide(snapshot):
        systems.append(build_prompt(snapshot)[0])
        seen.append([t["text"] for t in snapshot.context["recent_conversation"]])
        return _speak("It's me, Monika.")

    outcomes = asyncio.run(run_corpus(decide))
    assert len(outcomes) == len(CASES)
    assert all("You are Monika from Doki Doki" in s and s.startswith("Your name is Monika.") for s in systems)
    assert "clean" in format_report(outcomes)[-1]

    seen.clear()
    transcript = asyncio.run(run_dialogue(decide))
    assert len(transcript) == len(SAMPLE_DIALOGUE)
    assert seen[1] == ["Hello?", "It's me, Monika.", SAMPLE_DIALOGUE[1][1]]  # replies carried forward
