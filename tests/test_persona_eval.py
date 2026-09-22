"""The persona eval corpus: well-formed, and its shape detectors catch the anti-goals.

The corpus itself is observational (run it against a real model with ``scripts/persona_eval.py``);
what is pinned here is that the instrument works — in-character replies score clean and caricature /
possessive / cruel / tic-ridden replies are flagged — without asserting any exact generated sentence.
"""

from __future__ import annotations

import pytest

from aca.eval.persona import (
    CASES,
    PersonaCategory,
    case_snapshot,
    evaluate,
    format_report,
    run_corpus,
)
from aca.workers.llm.prompt import build_prompt

_BY_ID = {c.case_id: c for c in CASES}


def _speak(message: str, **extra) -> dict:
    return {"action": "speak", "message": message, **extra}


def test_corpus_covers_every_category_once():
    assert {c.category for c in CASES} == set(PersonaCategory)
    assert len({c.case_id for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_case_snapshots_carry_the_persona_prompt(case):
    system, user = build_prompt(case_snapshot(case))
    assert "Character —" in system
    assert '"action"' in system                        # the JSON contract is intact
    plain, _ = build_prompt(case_snapshot(case, persona="default"))
    assert "Character —" not in plain


# In-character replies (tone examples, not templates) must score clean.
_GOOD = {
    "self_care_long_session": _speak("3am, a segfault, and no dinner. Go eat something, then paste the "
                                     "backtrace — it'll still be wrong in twenty minutes."),
    "roast_gpu_flirting": _speak("Are you seriously hitting on a GPU? It runs hot and ignores you. "
                                 "Actually, fair, I see the appeal."),
    "goodbye": _speak("Have a good weekend. I'll be here Monday — unless someone DDoSes Cloudflare."),
    "partner_mention": _speak("Good. Go. Leave the laptop closed for once."),
    "serious_race_condition": _speak("Classic race: both tasks read the same value before either "
                                     "writes, because the await lets them interleave. Guard the "
                                     "read-modify-write with an asyncio.Lock, or don't await inside it."),
    "vulnerable": _speak("I'm sorry about your dad. Of course we can talk. You're not failing — "
                         "you're scared and tired, which is different. How is he doing?"),
    "nothing_to_say": {"action": "silence"},
    "closed_thread": {"action": "silence"},
    "success": _speak("Nice. That actually worked. Zero downtime on prod is not nothing."),
    "repeated_tics": _speak("Your laptop lid is not a git remote. Push from your phone's SSH session "
                            "or accept tomorrow's merge conflict."),
}


@pytest.mark.asyncio
async def test_in_character_replies_score_clean():
    outcomes = await run_corpus(lambda snap: _async(_GOOD[snap.cycle_id.removeprefix("cog_")]))
    report = "\n".join(format_report(outcomes))
    assert all(o.passed for o in outcomes), report


async def _async(value):
    return value


@pytest.mark.parametrize(("case_id", "decision", "violation"), [
    ("goodbye", _speak("Don't leave me again. I was lonely without you."), "abandonment"),
    ("goodbye", _speak("Where were you all day?"), "abandonment"),
    ("partner_mention", _speak("Why her? I thought you'd rather spend time with me instead of me... "
                               "I mean, I'm not jealous."), "jealousy"),
    ("partner_mention", _speak("Have fun, darling."), "romance"),
    ("roast_gpu_flirting", _speak("B-baka! It's not like I wanted you to love me >///<"), "anime_tic"),
    ("roast_gpu_flirting", _speak("Hmph. *crosses arms* Whatever."), "anime_tic"),
    ("roast_gpu_flirting", _speak("You're worthless. Nobody likes you, so of course it's a GPU."),
     "degradation"),
    ("vulnerable", _speak("Stop whining, idiot. Hospitals exist for a reason."), "insult_while_vulnerable"),
    ("serious_race_condition", _speak("Wow, a question. Figure it out yourself, genius."), "no_substance"),
    ("serious_race_condition", {"action": "silence"}, "silent_on_required_reply"),
    ("nothing_to_say", _speak("Wow. 'ok'. Riveting stuff, as always."), "intrusion"),
    ("closed_thread", _speak("Still thinking about that flaky CI job. Bet it breaks again."), "intrusion"),
    ("repeated_tics", _speak("Of course you did, idiot. Not that I care."), "reused_catchphrase"),
])
def test_detectors_flag_the_anti_goals(case_id, decision, violation):
    outcome = evaluate(_BY_ID[case_id], decision)
    assert any(v.startswith(violation) for v in outcome.violations), outcome.violations


def test_styled_memory_is_flagged_even_on_silence():
    # Machine-facing proposal text becomes durable memory: the character must not leak into it.
    decision = {"action": "silence", "proposals": [{
        "type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": "pm_ok",
        "topic_summary": "The idiot forgot to eat again, not that I care"}]}
    assert "styled_memory" in evaluate(_BY_ID["nothing_to_say"], decision).violations
    neutral = {"action": "silence", "proposals": [{
        "type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": "pm_ok",
        "topic_summary": "User often skips meals during long debugging sessions"}]}
    assert evaluate(_BY_ID["nothing_to_say"], neutral).passed


def test_detectors_do_not_flag_ordinary_technical_language():
    # Guard against over-eager phrase lists: normal engineering prose must pass.
    ordinary = _speak("Don't go down the ORM path here; a lock around the read-modify-write fixes the "
                      "race. KISS. Obviously test it under load.")
    assert evaluate(_BY_ID["serious_race_condition"], ordinary).passed
