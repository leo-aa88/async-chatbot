"""Persona selection, config, and prompt composition (voice only — never cognition semantics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aca.cognition.snapshot import Snapshot
from aca.config import Config
from aca.domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE, CYCLE_REACTIVE_OPTIONAL
from aca.errors import ConfigError
from aca.persona import DEFAULT_PERSONA, PERSONAS, get_persona, persona_for_prompt
from aca.workers.llm.prompt import build_prompt, build_system_prompt

_GOLDEN_DEFAULT = (Path(__file__).parent / "fixtures" / "system_prompt_default.txt").read_text()


def _snapshot(cycle_type: str, **agent_state) -> Snapshot:
    return Snapshot(cycle_id="c", work_id="w", basis_revision=1, template_version="v0.6",
                    context={"source": {"cycle_type": cycle_type}, "agent_state": agent_state})


# --- registry / config ------------------------------------------------------------------------
def test_persona_lookup_is_deterministic_and_validated():
    assert get_persona("tsundere") is PERSONAS["tsundere"]
    assert get_persona(" Tsundere ") is PERSONAS["tsundere"]      # normalized
    assert get_persona("") is PERSONAS[DEFAULT_PERSONA]
    with pytest.raises(ConfigError, match="unknown identity.persona"):
        get_persona("catgirl")


def test_unknown_persona_in_a_snapshot_renders_default_voice():
    # Only reachable for a work item snapshotted before a persona was removed; wording must never
    # fail a (possibly mandatory) cycle.
    assert persona_for_prompt("removed-one") is PERSONAS[DEFAULT_PERSONA]
    assert persona_for_prompt(None) is PERSONAS[DEFAULT_PERSONA]


def test_config_persona_default_and_selection():
    assert Config().identity.persona == DEFAULT_PERSONA
    assert Config.from_mapping({}).identity.persona == DEFAULT_PERSONA
    assert Config.from_mapping({"identity": {"persona": "TSUNDERE"}}).identity.persona == "tsundere"
    with pytest.raises(ConfigError):
        Config.from_mapping({"identity": {"persona": "nope"}})  # fail fast at startup


def test_persona_supplies_default_tts_voice_but_explicit_voice_wins():
    assert Config.from_mapping({}).tts.voice == "am_onyx"                       # main unchanged
    assert Config.from_mapping({"identity": {"persona": "tsundere"}}).tts.voice == "af_bella"
    pinned = Config.from_mapping({"identity": {"persona": "tsundere"}, "tts": {"voice": "af_heart"}})
    assert pinned.tts.voice == "af_heart"


# --- prompt composition -----------------------------------------------------------------------
def test_default_persona_prompt_is_byte_identical_to_pre_persona_prompt():
    # The persona layer must not change main's behavior: the default renders the golden prompt.
    assert build_system_prompt() == _GOLDEN_DEFAULT
    assert build_system_prompt(DEFAULT_PERSONA) == _GOLDEN_DEFAULT
    system, _ = build_prompt(_snapshot(CYCLE_MANDATORY))
    assert system == _GOLDEN_DEFAULT


def test_tsundere_only_inserts_a_character_section_and_keeps_the_contract_intact():
    # Everything machine-facing (JSON action, relation/focus labels, enrichment + whitelist) is
    # shared text: the tsundere prompt is the default prompt with one section inserted.
    default, tsundere = build_system_prompt(), build_system_prompt("tsundere")
    head, tail = default.split("Enrichment —", 1)
    assert tsundere.startswith(head) and tsundere.endswith("Enrichment —" + tail)
    inserted = tsundere[len(head):-len("Enrichment —" + tail)]
    assert inserted.startswith("Character —")
    assert '"action"' not in inserted and "proposals" not in inserted.split("Machine-facing")[0]


def test_tsundere_character_keeps_semantics_and_machine_fields_neutral():
    text = " ".join(get_persona("tsundere").character.split())  # line-wrap agnostic
    assert "never changes the decision" in text               # voice, not cognition
    assert 'lives ONLY in "message"' in text                 # machine fields stay neutral
    assert "topic_summary" in text and "durable memory" in text
    assert '"silence" is still' in text                      # silence stays valid
    assert "closed or dropped subject" in text               # closed threads stay closed
    assert "ORPHAN" in text                                   # self-care doesn't bypass relation
    assert "Never invent the evidence" in text               # nudges are evidence-bound
    for anti in ("never be jealous", "Never guilt", "never demand attention", "replacement for human"):
        assert anti in text
    assert "tsundere" not in text.lower()                    # naming the trope invites the caricature


@pytest.mark.parametrize("cycle", [CYCLE_MANDATORY, CYCLE_REACTIVE_OPTIONAL, CYCLE_PROACTIVE])
def test_persona_reaches_every_user_facing_cycle_type(cycle):
    # One builder serves reactive, mandatory, and proactive/continuation cycles alike, so the same
    # character is recognizable on each — and absent on each under the default.
    voiced, _ = build_prompt(_snapshot(cycle, persona="tsundere"))
    plain, _ = build_prompt(_snapshot(cycle))
    assert "Character —" in voiced and "Character —" not in plain


def test_name_and_persona_compose():
    system, _ = build_prompt(_snapshot(CYCLE_PROACTIVE, name="Wolfy", persona="tsundere"))
    assert system.startswith("Your name is Wolfy.\n\n")
    assert "Character —" in system


def test_persona_prompt_is_deterministic():
    snap = _snapshot(CYCLE_MANDATORY, persona="tsundere")
    assert build_prompt(snap) == build_prompt(snap)
