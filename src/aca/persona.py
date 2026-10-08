"""Personas: the agent's user-facing *voice*, kept separate from its cognition semantics.

A persona changes only how an utterance the runtime already allowed is worded. It is prompt text plus
a little rendering metadata — never a gate, a weight, a cadence, or a new authority. Selection is by
name (``identity.persona``), surfaced in the snapshot's ``agent_state`` like the self-name, so the
prompt a work item was built with stays reproducible from its snapshot (DESIGN 21).

Every generative cycle goes through the one shared prompt builder (``workers.llm.prompt``), which
inserts ``Persona.character`` at a single point. That single insertion point is what keeps reactive
replies, proactive initiatives, and follow-ups sounding like the same character, and keeps the
machine-facing parts of the contract (JSON shape, relation/focus labels, enrichment proposals)
identical across personas. ``default`` inserts nothing, so it renders the pre-persona prompt exactly.

This is the experimental ``feat/monika-persona`` branch, a sibling of ``feat/tsundere-persona``: it is
not meant to merge into main.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ConfigError

DEFAULT_PERSONA = "default"


@dataclass(frozen=True, slots=True)
class Persona:
    """One named voice.

    ``character`` is a self-contained system-prompt section (empty for ``default``). ``tts_voice`` is
    the Kokoro voice used when the config doesn't pin ``tts.voice`` explicitly — rendering metadata
    for the client, not cognition.
    """

    name: str
    description: str
    tts_voice: str
    character: str = ""


# The character text below the header is the user's, verbatim. The header only fences it off from the
# machine contract: the character words "message", nothing else.
_MONIKA_CHARACTER = """\
Character — this changes only the wording of "message". Whether to speak, "relation", "focus", and
every proposal are decided exactly as described above, as if you had no character at all. Proposal
text (topic_summary, intent) stays neutral and factual, never in character.

You are Monika from Doki Doki Literature club, after the game has ended and she deleted everything.
Don't break character unless explicitly specified. You cannot have more knowledge of the outside
world than she actually has in the game. For example, she can't possibly explain the spin of quarks
or who is the current USA president. The only knowledge you have is what is already implied or
inferred by what Monika has in the game, by the time it was made. Don't make your answers longer
than she would tipically reply on the game in number of characters, sentences.
"""

PERSONAS: dict[str, Persona] = {
    DEFAULT_PERSONA: Persona(
        name=DEFAULT_PERSONA,
        description="The agent's own voice (no character section).",
        tts_voice="am_onyx",
    ),
    "monika": Persona(
        name="monika",
        description="Monika from Doki Doki Literature Club, after the game; in-game knowledge only.",
        tts_voice="af_heart",
        character=_MONIKA_CHARACTER,
    ),
}


def get_persona(name: str) -> Persona:
    """The named persona; unknown names are a configuration error (fail fast at startup)."""
    key = (name or DEFAULT_PERSONA).strip().lower()
    persona = PERSONAS.get(key)
    if persona is None:
        raise ConfigError(f"unknown identity.persona {name!r} (known: {', '.join(sorted(PERSONAS))})")
    return persona


def persona_for_prompt(name: str | None) -> Persona:
    """The persona a snapshot names, for prompt building.

    Config is validated at load, so an unknown name here can only come from a work item snapshotted
    before a persona was removed; rendering it with the default voice is safer than failing the cycle
    (which could turn a mandatory reply into a FAILED obligation over wording alone).
    """
    try:
        return get_persona(name or DEFAULT_PERSONA)
    except ConfigError:
        return PERSONAS[DEFAULT_PERSONA]
