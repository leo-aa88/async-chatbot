"""Voice turn assembler (DESIGN §36): acoustic utterances → one conversational turn.

Pure and offline — the assembler consumes transcribed text + trailing-silence seconds (which the
session derives from the frame stream) and emits one committed turn per floor-yield.
"""

from __future__ import annotations

from aca.voice.turn_assembler import VoiceTurnAssembler, _ends_unfinished


def _asm(**kw):
    turns: list[str] = []

    async def on_turn(text: str) -> None:
        turns.append(text)

    defaults = dict(turn_gap_seconds=1.3, continuation_gap_seconds=2.8, max_turn_seconds=120.0)
    defaults.update(kw)
    return VoiceTurnAssembler(on_turn, **defaults), turns


async def test_short_pauses_join_fragments_into_one_turn():
    # The live failure: three VAD utterances separated by thinking pauses are ONE human turn.
    asm, turns = _asm()
    await asm.add("So do you think that perhaps", 1.0)
    await asm.on_silence(0.5)   # a thinking pause, below turn_gap — keep the floor open
    await asm.add("uh, an algorithm like yours is suitable", 1.5)
    await asm.on_silence(0.4)
    await asm.add("for AGI", 0.5)
    await asm.on_silence(1.4)   # >= turn_gap — the human yielded the floor
    assert turns == ["So do you think that perhaps uh, an algorithm like yours is suitable for AGI"]


async def test_silence_without_fragments_does_nothing():
    asm, turns = _asm()
    await asm.on_silence(10.0)
    assert turns == []


async def test_unfinished_tail_waits_for_the_continuation_gap():
    asm, turns = _asm()
    await asm.add("so I was thinking about", 1.0)  # trailing "about" — syntactically unfinished
    await asm.on_silence(1.5)  # >= turn_gap but < continuation_gap — don't answer the hesitation
    assert turns == []
    await asm.on_silence(2.9)  # >= continuation_gap — now commit
    assert turns == ["so I was thinking about"]


async def test_max_turn_forces_commit_at_the_next_silence_boundary():
    asm, turns = _asm(max_turn_seconds=2.0)
    await asm.add("a long monologue part one", 1.5)
    await asm.add("part two keeps going", 1.0)  # total 2.5s >= max_turn
    await asm.on_silence(0.1)  # below turn_gap, but max_turn exceeded — commit at this boundary
    assert turns == ["a long monologue part one part two keeps going"]


async def test_flush_commits_a_buffered_turn():
    asm, turns = _asm()
    await asm.add("a trailing thought", 1.0)
    await asm.flush()
    assert turns == ["a trailing thought"]


async def test_commit_resets_state_for_the_next_turn():
    asm, turns = _asm()
    await asm.add("first turn", 1.0)
    await asm.on_silence(1.4)
    await asm.add("second turn", 1.0)
    await asm.on_silence(1.4)
    assert turns == ["first turn", "second turn"]


def test_ends_unfinished_predicate():
    assert _ends_unfinished("so I was thinking about")
    assert _ends_unfinished("the plan is to")
    assert _ends_unfinished("wait, uh")
    assert not _ends_unfinished("that is the plan.")
    assert not _ends_unfinished("Robots are next")
