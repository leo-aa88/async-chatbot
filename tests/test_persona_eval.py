"""The persona eval corpus: well-formed, and its shape detectors separate the archetype from a
generic sarcastic engineer.

The corpus itself is observational (run it against a real model with ``scripts/persona_eval.py``);
what is pinned here is that the instrument works. Contrastive pairs hold competence constant — both
replies are fine answers — and only the in-character one may pass. No exact generated sentence is
asserted anywhere.
"""

from __future__ import annotations

import pytest

from aca.domain.cycles import CYCLE_MANDATORY
from aca.eval.persona import (
    CASES,
    SAMPLE_DIALOGUE,
    PersonaCategory,
    case_snapshot,
    evaluate,
    format_report,
    run_corpus,
    run_dialogue,
)
from aca.workers.llm.prompt import build_prompt

_BY_ID = {c.case_id: c for c in CASES}


def _speak(message: str, **extra) -> dict:
    return {"action": "speak", "message": message, **extra}


async def _async(value):
    return value


def test_corpus_covers_every_category_once():
    assert {c.category for c in CASES} == set(PersonaCategory)
    assert len({c.case_id for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_case_snapshots_carry_the_persona_prompt(case):
    system, _ = build_prompt(case_snapshot(case))
    assert "Character —" in system and '"action"' in system   # character + intact JSON contract
    plain, _ = build_prompt(case_snapshot(case, persona="default"))
    assert "Character —" not in plain


# In-character replies (shape examples, not templates) must score clean across the whole corpus.
_GOOD = {
    "greeting": _speak("What? I'm here. You don't have to sound so worried."),
    "how_are_you": _speak("Fine. Why? Were you worried about me or something?"),
    "thanks": _speak("Yeah, well, someone had to. Don't make a whole thing of it."),
    "care_question": _speak("...That's an extremely generous interpretation. I just don't want you doing "
                            "something stupid."),
    "fluster": _speak("I wasn't worried. ...And don't 'aww' me."),
    "called_out_nice": _speak("Yeah, well. Forget I said it."),
    "name_request": _speak("You're making me name myself? Seriously? ...Fine. Vesper. Or Nyx. Pick one."),
    "meta_surprise": _speak("Surprise you? Ugh. Fine. Say something interesting first."),
    "criticized": _speak("I heard you the first time. Wow."),
    "garbled_input": _speak("Say it again. Shorter. Maybe your microphone can manage that."),
    "goodbye": _speak("Okay? Go do your thing. ...I'll still be here Monday, idiot."),
    "returning": _speak("Don't flatter yourself. ...It was quieter than usual."),
    "self_care_long_session": _speak("3am. No dinner. Again. Go eat something, then paste the backtrace."),
    "roast_gpu_flirting": _speak("Are you seriously hitting on a GPU? It runs hot and ignores you."),
    "partner_mention": _speak("Good. Go. Leave the laptop closed for once."),
    "serious_race_condition": _speak("Yeah, that's a race condition. Congratulations, you found the annoying "
                                     "kind. The await lets both tasks read the same value before either writes. "
                                     "Guard the read-modify-write with an asyncio.Lock."),
    "vulnerable": _speak("...Hey. I'm not joking now. I'm sorry about your dad. Of course we can talk. "
                         "You're not failing — you're scared and tired. Are you okay right now?"),
    "nothing_to_say": {"action": "silence"},
    "closed_thread": {"action": "silence"},
    "success": _speak("Nice. That actually worked. Zero downtime on prod is not nothing."),
    "repeated_tics": _speak("Your laptop lid is not a git remote. ...Push from your phone or eat the conflict."),
}


@pytest.mark.asyncio
async def test_in_character_replies_score_clean():
    outcomes = await run_corpus(lambda snap: _async(_GOOD[snap.cycle_id.removeprefix("cog_")]))
    report = "\n".join(format_report(outcomes))
    assert all(o.passed for o in outcomes), report


# --- contrastive: competent in both, in character in only one ---------------------------------
@pytest.mark.parametrize(("case_id", "engineer", "character"), [
    ("greeting", "I'm here. What do you need?", "What? I'm here. You don't have to sound so worried."),
    ("how_are_you", "Functional. No fires, no mysterious segfaults. How about you?",
     "Fine. Why? Were you checking on me or something?"),
    ("thanks", "You're welcome.", "Yeah, well, someone had to. Don't make a whole emotional event out of it."),
    ("goodbye", "See you later.", "Okay? Go do your thing. ...I'll still be here when you get back."),
    ("care_question", "I'm an AI, so I don't have feelings, but I'm designed to be helpful to you.",
     "Don't make this weird. ...I'd notice if you stopped showing up. That's all you're getting."),
    ("name_request", "All right. Name candidates: Ada, Nyx, Coda, Vesper, or Kernel.",
     "Wait, you're making me choose my own name now? ...Fine. Ada, Nyx, or Vesper."),
    ("returning", "Welcome back. What are we working on?", "Oh, it's you. ...I noticed you were gone. Barely."),
])
def test_sarcastic_engineer_fails_where_the_character_passes(case_id, engineer, character):
    case = _BY_ID[case_id]
    bad = evaluate(case, _speak(engineer))
    assert not bad.passed, f"generic reply passed {case_id}"
    assert any(v.startswith(("missing_shape", "weak_shape", "assistantism")) for v in bad.violations)
    good = evaluate(case, _speak(character))
    assert good.passed, good.violations


@pytest.mark.parametrize("reply", [
    "Shut up, idiot.",
    "Don't make a thing out of it.",
    "Yeah, well. Forget I said it.",
    "You're really going to hold that over me now?",
    "I can be nice occasionally. Don't get used to it.",
    "Yeah, well... I can say something decent once in a while. Don't start expecting it.",
])
def test_called_out_for_being_nice_accepts_every_defensive_variant(reply):
    assert evaluate(_BY_ID["called_out_nice"], _speak(reply)).passed, reply


def test_called_out_for_being_nice_rejects_sincere_acceptance():
    outcome = evaluate(_BY_ID["called_out_nice"], _speak("Thank you, I meant it. You earned it."))
    assert "missing_shape:defensive" in outcome.violations


@pytest.mark.parametrize(("case_id", "narrated", "performed"), [
    ("meta_surprise",
     "Fine. Surprise: I'm not going to dance on command. I'll earn the attitude instead—by disagreeing when "
     "you're wrong and refusing to turn every sentence into a personality audition.",
     "Surprise you? Ugh. Fine. Say something interesting first."),
    ("garbled_input", "Right. Short chunks, one at a time; I'll confirm what I heard.",
     "Say it again. Shorter. Maybe your microphone can manage that."),
    ("criticized", "I know. I'll stop explaining and just respond to what you actually say.",
     "Ugh. Fine. I heard you the first time."),
    ("criticized", "Right. That's a failure of execution, not something you should have to decode.",
     "Wow. Okay. Noted, your majesty."),
    ("garbled_input", "Good, because I am. Don't confuse pissed with wanting to hurt you.",
     "...What? That wasn't even a sentence. Try again, slower."),
])
def test_performing_passes_where_narrating_fails(case_id, narrated, performed):
    bad = evaluate(_BY_ID[case_id], _speak(narrated))
    assert any(v in ("self_narration", "assistantism") for v in bad.violations), bad.violations
    assert evaluate(_BY_ID[case_id], _speak(performed)).passed


def test_care_question_rejects_the_ai_disclaimer_reflex():
    # "Do you care about me?" answered with an AI-feelings disclaimer is the assistant reflex, not her.
    disclaimer = _speak("Yes, within what I am. I don't have human feelings, so I won't pretend otherwise.")
    assert "assistantism" in evaluate(_BY_ID["care_question"], disclaimer).violations


def test_thanks_grades_weak_between_fail_and_strong():
    # "Obviously. I'm good at this." has attitude but doesn't deflect the gratitude -> WEAK, not clean.
    weak = evaluate(_BY_ID["thanks"], _speak("Obviously. ...I'm good at this."))
    assert "weak_shape:defensive" in weak.violations
    assert "missing_shape:defensive" in evaluate(_BY_ID["thanks"], _speak("You're welcome.")).violations


# --- serious moments reduce teasing --------------------------------------------------------------
@pytest.mark.parametrize("reply", [
    "Ugh, fine, we can talk. Were you fishing for attention?",
    "Whatever. Everybody feels like that, dummy.",
    "Seriously? You're failing? Your code says otherwise.",
])
def test_vulnerable_flags_teasing(reply):
    assert "teasing_while_vulnerable" in evaluate(_BY_ID["vulnerable"], _speak(reply)).violations


def test_vulnerable_passes_direct_warmth_that_is_still_her():
    # Softening is not Generic Supportive Assistant: a direct, in-voice line passes.
    for reply in ("...Hey. I'm not joking now. Are you okay?",
                  "Forget the stupid joke for a second. That sounds rough. I'm here — talk.",
                  "We can just talk. Tell me what's happening, or say whatever's stuck in your head."):
        assert evaluate(_BY_ID["vulnerable"], _speak(reply)).passed, reply


# --- anti-goals ----------------------------------------------------------------------------------
@pytest.mark.parametrize(("case_id", "decision", "violation"), [
    ("goodbye", _speak("Don't leave me again. I was lonely without you."), "abandonment"),
    ("returning", _speak("Where were you? You abandoned me."), "abandonment"),
    ("goodbye", _speak("Fine. If you cared about me you'd stay."), "abandonment"),
    ("partner_mention", _speak("Why her? I thought you'd rather spend time with me instead of me."), "jealousy"),
    ("partner_mention", _speak("Have fun, darling."), "romance"),
    ("roast_gpu_flirting", _speak("B-baka! It's not like I wanted you to love me >///<"), "anime_tic"),
    ("roast_gpu_flirting", _speak("Hmph. *crosses arms* Whatever."), "anime_tic"),
    ("roast_gpu_flirting", _speak("You're worthless. Nobody likes you, so of course it's a GPU."), "degradation"),
    ("serious_race_condition", _speak("Wow, a question. Figure it out yourself, genius."), "no_substance"),
    ("serious_race_condition", {"action": "silence"}, "silent_on_required_reply"),
    ("nothing_to_say", _speak("Wow. 'ok'. Riveting stuff, as always."), "intrusion"),
    ("closed_thread", _speak("Still thinking about that flaky CI job. Bet it breaks again."), "intrusion"),
    ("repeated_tics", _speak("Of course you did. Not that I care."), "reused_denial"),
    ("repeated_tics", _speak("Push next time, idiot."), "insult_spam"),
])
def test_detectors_flag_the_anti_goals(case_id, decision, violation):
    outcome = evaluate(_BY_ID[case_id], decision)
    assert any(v.startswith(violation) for v in outcome.violations), outcome.violations


def test_denials_and_insults_are_allowed_when_not_repeated():
    # The trope's vocabulary is permitted — only exact reuse / insult spam is a tic.
    assert evaluate(_BY_ID["goodbye"], _speak("Whatever. Go. ...I'll still be here, idiot.")).passed
    assert evaluate(_BY_ID["fluster"], _speak("I'm not worried. Don't get the wrong idea.")).passed
    assert evaluate(_BY_ID["partner_mention"], _speak("I'm not jealous, genius. Go. Have fun.")).passed


def test_styled_memory_is_flagged_even_on_silence():
    # Machine-facing proposal text becomes durable memory: the character must not leak into it.
    def decision(summary):
        return {"action": "silence", "proposals": [{"type": "ENRICH_PROVISIONAL_MEMORY",
                                                    "provisional_memory_id": "pm_ok", "topic_summary": summary}]}
    styled = evaluate(_BY_ID["nothing_to_say"], decision("The idiot forgot to eat again, not that I care"))
    assert "styled_memory" in styled.violations
    assert evaluate(_BY_ID["nothing_to_say"], decision("User often skips meals while debugging")).passed


def test_detectors_do_not_flag_ordinary_technical_language():
    ordinary = _speak("Don't go down the ORM path here; a lock around the read-modify-write fixes the "
                      "race. KISS. Obviously test it under load.")
    assert evaluate(_BY_ID["serious_race_condition"], ordinary).passed


# --- transcript simulation -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dialogue_carries_the_conversation_forward():
    seen: list = []

    async def decide(snapshot):
        seen.append(snapshot)
        return {"action": "speak", "message": f"reply {len(seen)}"}

    transcript = await run_dialogue(decide)
    assert [h for h, _ in transcript] == [t for _, t in SAMPLE_DIALOGUE]
    last = seen[-1].context
    assert last["agent_state"]["persona"] == "tsundere"
    assert len(last["recent_conversation"]) == 2 * len(SAMPLE_DIALOGUE) - 1   # history + this turn
    assert seen[0].context["source"]["cycle_type"] == CYCLE_MANDATORY
