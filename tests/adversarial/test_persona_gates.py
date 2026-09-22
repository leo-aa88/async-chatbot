"""A persona is voice, never authority: it must not move any gate.

Two guarantees, driven through the real reducer:

1. **Injection** — the configured persona reaches the generative snapshot (and therefore the prompt)
   on reply cycles and on proactive cycles (the latter checked inside the invariance cases); the
   default config leaves snapshots untouched.
2. **Invariance** — every continuity-corpus decision (continue / near-repeat / interruption /
   resumption / closed-thread / unrelated-after-close / backchannel / dormant resurfacing) comes out
   identically with the tsundere persona as with the default. The model is held constant (always
   speaks), so any difference could only come from the persona leaking into a gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import Harness

from aca.clock import ManualClock
from aca.config import Config
from aca.workers.base import LLMOutput
from aca.workers.llm.prompt import build_prompt

from . import test_continuity_corpus as corpus


class _RecordingSpeaker:
    """Always speaks, recording the system prompt each cycle would have been sent."""

    def __init__(self) -> None:
        self.systems: list[tuple[str, str]] = []

    async def run(self, snapshot) -> LLMOutput:
        system, _ = build_prompt(snapshot)
        self.systems.append((str((snapshot.context.get("source") or {}).get("cycle_type")), system))
        return LLMOutput(result={"action": "speak", "message": "Following that thread —"},
                         tokens_in=8, tokens_out=5)


def _with_persona(config: Config, persona: str) -> Config:
    return config.with_overrides(identity=config.identity.__class__(name=config.identity.name,
                                                                      persona=persona))


def test_persona_reaches_reply_snapshots_and_prompt(tmp_path):
    llm = _RecordingSpeaker()
    config = _with_persona(Config.from_mapping({"rng_seed": 3}), "tsundere")
    h = Harness(tmp_path, config, ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)), llm=llm)
    h.send_human("Can you explain how the reducer serializes writes?")
    pending = [w for w in h.pending_work() if "context" in w.snapshot]  # generative (not embedding) work
    assert pending, "a mandatory reply must create generative work"
    # The persisted snapshot itself carries the persona, so the prompt is replayable from it.
    assert all(w.snapshot["context"]["agent_state"]["persona"] == "tsundere" for w in pending)
    h.run_all_pending()
    assert llm.systems and all("Character —" in system for _, system in llm.systems)


def test_default_config_leaves_snapshots_without_persona(tmp_path):
    seen: list[dict] = []

    class _Spy(_RecordingSpeaker):
        async def run(self, snapshot):
            seen.append(snapshot.context.get("agent_state") or {})
            return await super().run(snapshot)

    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 3}),
                ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)), llm=_Spy())
    h.send_human("What's the capital of France?")
    h.run_all_pending()
    assert seen and all("persona" not in state for state in seen)


@pytest.mark.parametrize("case", corpus.CASES, ids=lambda c: c.__name__)
def test_persona_does_not_move_any_continuity_decision(case, tmp_path, monkeypatch):
    baseline = case(tmp_path / "default")

    recorder = _RecordingSpeaker()
    base_config = corpus._config
    monkeypatch.setattr(corpus, "_config", lambda: _with_persona(base_config(), "tsundere"))
    monkeypatch.setattr(corpus, "_SpeakLLM", lambda: recorder)
    voiced = case(tmp_path / "tsundere")

    assert voiced.did_speak == baseline.did_speak, f"persona changed the {voiced.case_id} decision"
    # Whenever a proactive cycle did reach the model, it was the persona prompt that reached it —
    # so the invariance above is not vacuous.
    proactive = [system for cycle_type, system in recorder.systems if cycle_type == "proactive"]
    if voiced.did_speak:
        assert proactive, "a proactive message was committed without the model being called"
    assert all("Character —" in system for system in proactive)
