"""Her voice work from tsundere.chat: the pet-warning and silence notes, and no em dashes.

Everything here is the tsundere persona's only: the default persona's prompts and replies are
unchanged (its golden prompt is checked in test_persona.py).
"""

from __future__ import annotations

import asyncio

from aca.cognition.snapshot import Snapshot
from aca.domain.cycles import CYCLE_MANDATORY
from aca.persona import catchphrase_note, silence_note
from aca.workers.llm.adapters import ChatResult
from aca.workers.llm.prompt import build_prompt
from aca.workers.llm.worker import ProviderLLMWorker


def _agent(text):
    return {"role": "agent", "text": text}


def _human(text, outcome=None):
    return {"role": "human", "text": text, **({"outcome": outcome} if outcome else {})}


def test_a_pet_warning_already_used_is_named():
    note = catchphrase_note([_agent("Fine. Don't get smug about it."), _agent("Sure.")])
    assert note and "'smug'" in note
    assert catchphrase_note([_agent("Fine."), _agent("Sure.")]) is None


def test_a_chosen_silence_is_hers_to_carry_and_a_failure_is_never_her_choice():
    chose = silence_note([_human("shipped the migration", "chose_silence"), _human("why did you ignore me?")])
    assert chose and "on purpose" in chose and "Never an apology" in chose
    failed = silence_note([_human("shipped the migration", "failed"), _human("hello?")])
    assert failed and "not your choice" in failed and "Don't claim you ignored it" in failed
    assert silence_note([_human("hi", "superseded"), _human("hello")]) is None
    assert silence_note([_human("hi")]) is None


def _snapshot(persona, recent):
    state = {"persona": persona} if persona else {}
    return Snapshot("c", "w", 1, "t", {"agent_state": state, "recent_conversation": recent,
                                        "source": {"cycle_type": CYCLE_MANDATORY, "text": "hello?"}})


def test_the_notes_reach_her_prompt_only():
    recent = [_agent("Don't look so pleased."), _human("shipped it", "chose_silence"), _human("hello?")]
    _, hers = build_prompt(_snapshot("tsundere", recent))
    _, default = build_prompt(_snapshot(None, recent))
    assert "Give that family a rest" in hers and "on purpose" in hers
    assert "style_note" not in default


class _Adapter:
    async def complete(self, system, user):
        return ChatResult(text='{"action": "speak", "message": "Fine—it works."}', tokens_in=1, tokens_out=1)


def _reply(persona):
    return asyncio.run(ProviderLLMWorker(_Adapter()).run(_snapshot(persona, []))).result["message"]


def test_she_never_writes_an_em_dash_and_the_default_persona_is_untouched():
    assert _reply("tsundere") == "Fine, it works."
    assert _reply(None) == "Fine—it works."
