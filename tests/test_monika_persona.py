"""The Monika persona: voice-only, inserted at one point, default prompt untouched."""

from __future__ import annotations

from pathlib import Path

import pytest

from aca.cognition.snapshot import Snapshot
from aca.config import Config
from aca.errors import ConfigError
from aca.persona import DEFAULT_PERSONA, PERSONAS, get_persona, persona_for_prompt
from aca.workers.llm.prompt import build_prompt, build_system_prompt

_FIXTURE = Path(__file__).parent / "fixtures" / "system_prompt_default.txt"
_USER_TEXT = "You are Monika from Doki Doki Literature club, after the game has ended and she deleted everything."


def _snapshot(**agent_state) -> Snapshot:
    return Snapshot("c", "w", 1, "t", {"agent_state": agent_state})


def test_default_prompt_is_byte_identical_to_main():
    assert build_system_prompt() == _FIXTURE.read_text()
    assert build_system_prompt(DEFAULT_PERSONA) == _FIXTURE.read_text()
    assert build_prompt(_snapshot())[0] == _FIXTURE.read_text()


def test_monika_text_sits_between_the_voice_rules_and_the_machine_contract():
    system = build_system_prompt("monika")
    assert _USER_TEXT in system
    assert "Don't break character unless explicitly specified." in system
    voice, character, enrichment = (system.index(s) for s in ("Voice —", "Character —", "Enrichment —"))
    assert voice < character < enrichment
    # Everything outside the character section is the default prompt, unchanged.
    default = _FIXTURE.read_text()
    assert system.replace(PERSONAS["monika"].character + "\n", "") == default


def test_character_never_owns_the_machine_fields():
    assert "Proposal\ntext (topic_summary, intent) stays neutral" in PERSONAS["monika"].character


def test_name_and_persona_reach_the_prompt():
    system, _ = build_prompt(_snapshot(name="Monika", persona="monika"))
    assert system.startswith("Your name is Monika.\n\n")
    assert _USER_TEXT in system


def test_unknown_persona_fails_at_config_and_falls_back_in_prompt():
    with pytest.raises(ConfigError):
        get_persona("natsuki")
    with pytest.raises(ConfigError):
        Config.from_mapping({"identity": {"persona": "natsuki"}})
    assert persona_for_prompt("natsuki").name == DEFAULT_PERSONA  # a stale snapshot never fails a cycle


def test_monika_speaks_with_af_heart_unless_a_voice_is_pinned():
    assert Config.from_mapping({"identity": {"persona": "monika"}}).tts.voice == "af_heart"
    assert Config.from_mapping({"identity": {"persona": "Monika"}}).identity.persona == "monika"
    pinned = {"identity": {"persona": "monika"}, "tts": {"voice": "af_bella"}}
    assert Config.from_mapping(pinned).tts.voice == "af_bella"
    assert Config.from_mapping({}).tts.voice == "am_onyx"
