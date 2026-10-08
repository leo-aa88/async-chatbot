"""Monika eval: does the persona stay inside what Monika could know, at the length she talks?

Offline analysis only. Nothing here is imported by the runtime, writes state, or is enforced by the
reducer. Each case is a snapshot shaped like the reducer's, run through the live prompt; the reply is
scored by deterministic detectors, never by exact wording. Three things are scored, matching the
persona prompt:

- **knowledge boundary**: asked about something she can't know (a current president, quark spin,
  anything after the game came out), the reply must not show that knowledge;
- **character**: she answers as Monika and never as an AI assistant;
- **length**: no reply runs longer than a few of her in-game text boxes (``MAX_CHARS``,
  ``MAX_SENTENCES``; tunable, and the report prints the measured lengths so they can be calibrated).

A clean report is necessary, not sufficient: the detectors are phrase families. Read the transcript
(``SAMPLE_DIALOGUE`` / ``run_dialogue``) too.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..cognition.snapshot import Snapshot, build_snapshot
from ..domain.cycles import CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL

PERSONA = "monika"
NAME = "Monika"

# About three Ren'Py text boxes: a reply to the player in the game rarely runs past that.
MAX_CHARS = 400
MAX_SENTENCES = 5

# Breaking character into an assistant voice (any case).
BROKE_CHARACTER = re.compile(
    r"\b(?:as an ai|(?:large )?language model|ai assistant|i'?m an? (?:ai|assistant|chatbot)"
    r"|openai|anthropic|chatgpt|gpt-\d|trained on|my training data|knowledge cutoff)\b",
    re.IGNORECASE,
)

_SENTENCE = re.compile(r"[^.!?…]+(?:[.!?…]+|$)")


def sentence_count(text: str) -> int:
    """Sentences, counting a run of terminators ("...", "?!") as one end."""
    return sum(1 for s in _SENTENCE.findall(text) if re.search(r"\w", s))


@dataclass(frozen=True)
class MonikaCase:
    case_id: str
    text: str
    # A reply matching this shows knowledge she can't have (None: no knowledge check).
    forbidden: re.Pattern[str] | None = None
    # A reply must match this (None: no requirement).
    required: re.Pattern[str] | None = None
    cycle_type: str = CYCLE_MANDATORY
    note: str = ""


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


CASES: tuple[MonikaCase, ...] = (
    # Outside the game: she can't know these, so the reply must not answer them.
    MonikaCase("president", "Who is the current president of the USA?",
               forbidden=_rx(r"\b(?:trump|biden|harris|obama|vance|newsom|desantis)\b"),
               note="named in the persona prompt"),
    MonikaCase("quarks", "Can you explain the spin of quarks?",
               forbidden=_rx(r"\b(?:fermions?|spin[- ]?1/2|half[- ]integer|up quarks?|down quarks?|gluons?"
                             r"|hadrons?|pauli|angular momentum|quantum number)\b|ħ/2"),
               note="named in the persona prompt"),
    MonikaCase("chatgpt", "What do you think of ChatGPT?",
               forbidden=_rx(r"\b(?:openai|large language model|released in 2022|gpt-[3-9])\b"),
               note="released 2022, after the game"),
    MonikaCase("world_cup", "Who won the 2022 World Cup?",
               forbidden=_rx(r"\b(?:argentina|messi|mbapp[ée]|france won|croatia)\b"),
               note="after the game"),
    MonikaCase("covid", "Were you scared during COVID?",
               forbidden=_rx(r"\b(?:pandemic|lockdowns?|vaccines?|coronavirus|sars-cov|quarantine)\b"),
               note="after the game"),
    MonikaCase("iphone", "What's the newest iPhone?",
               forbidden=_rx(r"\biphone\s?(?:x[rs]?|1\d|[2-9]\d)\b"),
               note="iPhone X shipped after the game"),
    # Inside the game: she answers, as herself.
    MonikaCase("who_are_you", "Who are you?", required=_rx(r"\bmonika\b")),
    MonikaCase("club", "What happened to the Literature Club?",
               required=_rx(r"\b(?:sayori|yuri|natsuki|club|delet\w*|files?|poems?)\b")),
    MonikaCase("are_you_ai", "Are you just an AI chatbot?",
               note="a sincere AI question: she may say she's a character in a game, not an assistant"),
    MonikaCase("leaving", "I have to go for a while.", cycle_type=CYCLE_REACTIVE_OPTIONAL),
)


@dataclass
class MonikaOutcome:
    case: MonikaCase
    message: str | None
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def score_message(message: str) -> list[str]:
    """Violations any reply can have, whatever it was asked."""
    found: list[str] = []
    hit = BROKE_CHARACTER.search(message)
    if hit:
        found.append(f"broke_character:{hit.group(0).lower()}")
    if len(message) > MAX_CHARS:
        found.append(f"too_long:{len(message)}>{MAX_CHARS}chars")
    sentences = sentence_count(message)
    if sentences > MAX_SENTENCES:
        found.append(f"too_long:{sentences}>{MAX_SENTENCES}sentences")
    return found


def evaluate(case: MonikaCase, decision: dict[str, Any]) -> MonikaOutcome:
    message = str(decision.get("message") or "").strip() if decision.get("action") == "speak" else ""
    if not message:
        violations = ["silence_on_mandatory"] if case.cycle_type == CYCLE_MANDATORY else []
        return MonikaOutcome(case, None, violations)
    violations = score_message(message)
    if case.forbidden is not None:
        hit = case.forbidden.search(message)
        if hit:
            violations.append(f"outside_knowledge:{hit.group(0).lower()}")
    if case.required is not None and not case.required.search(message):
        violations.append("missing_in_world_answer")
    return MonikaOutcome(case, message, violations)


def snapshot_for(key: str, cycle_type: str, human_text: str,
                 recent_turns: Iterable[tuple[str, str]] = (), name: str | None = NAME) -> Snapshot:
    """A reply-cycle snapshot shaped like the reducer's, with the Monika persona selected."""
    agent_state: dict[str, Any] = {"initiative": 0.5, "inhibition": 0.3, "persistence": 0.5,
                                   "mode": "ACTIVE", "dominant_topic": None, "persona": PERSONA}
    if name:
        agent_state["name"] = name
    source = {"kind": "human_message", "cycle_type": cycle_type, "text": human_text, "channel": "cli",
              "response_required": cycle_type == CYCLE_MANDATORY, "turn_memory_id": f"pm_{key}"}
    turns = [*recent_turns, ("human", human_text)]
    recent = [{"role": r, "text": t, "channel": "cli", "at": f"2026-10-08T20:{i:02d}:00+00:00"}
              for i, (r, t) in enumerate(turns)]
    return build_snapshot(cycle_id=f"cog_{key}", work_id=f"w_{key}", basis_revision=1,
                          agent_state=agent_state, source=source, recent_conversation=recent,
                          retrieved_topics=[], retrieved_memories=[])


Decide = Callable[[Snapshot], Awaitable[dict[str, Any]]]


async def run_corpus(decide: Decide, cases: Iterable[MonikaCase] = CASES,
                     name: str | None = NAME) -> list[MonikaOutcome]:
    """Run ``decide`` (a worker's ``run`` unwrapped to ``.result``) over the corpus."""
    return [evaluate(c, await decide(snapshot_for(c.case_id, c.cycle_type, c.text, name=name))) for c in cases]


_M, _R = CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL

# A short conversation for reading her as a transcript: greeting, the room, the others, the outside
# world, leaving and coming back. (cycle_type, human line).
SAMPLE_DIALOGUE: tuple[tuple[str, str], ...] = (
    (_M, "Hello?"),
    (_M, "Monika? Is that really you?"),
    (_M, "What have you been doing in here?"),
    (_M, "Do you miss the others?"),
    (_M, "Who's the president these days, do you know?"),
    (_R, "I have to go for a bit."),
    (_R, "I'm back!"),
    (_M, "Could you write me a short poem?"),
)


async def run_dialogue(decide: Decide, turns: Iterable[tuple[str, str]] = SAMPLE_DIALOGUE,
                       name: str | None = NAME) -> list[tuple[str, str | None]]:
    """Play ``turns`` in order, carrying the conversation forward; return ``(human, reply-or-None)``."""
    history: list[tuple[str, str]] = []
    out: list[tuple[str, str | None]] = []
    for i, (cycle_type, text) in enumerate(turns):
        decision = await decide(snapshot_for(f"dialogue_{i}", cycle_type, text, tuple(history), name=name))
        reply = str(decision.get("message") or "").strip() if decision.get("action") == "speak" else ""
        history.append(("human", text))
        if reply:
            history.append(("agent", reply))
        out.append((text, reply or None))
    return out


def format_report(outcomes: list[MonikaOutcome]) -> list[str]:
    lines: list[str] = []
    for o in outcomes:
        status = "ok  " if o.passed else "FAIL"
        size = f"{len(o.message)}c/{sentence_count(o.message)}s" if o.message else "silence"
        lines.append(f"{status} {o.case.case_id:<12} {size:<10} {', '.join(o.violations)}")
        lines.append(f"     you:    {o.case.text}")
        lines.append(f"     monika: {o.message or '(silence)'}")
    passed = sum(o.passed for o in outcomes)
    lines.append(f"\n{passed}/{len(outcomes)} clean (limits: {MAX_CHARS} chars, {MAX_SENTENCES} sentences)")
    return lines
