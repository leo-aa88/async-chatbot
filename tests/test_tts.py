"""Speech output (DESIGN 2.2): config parsing, text cleanup, engine factory, serialized playback.

Everything here is client-side rendering — no reducer, no durable state, no injected clock/rng.
The Kokoro model itself is never loaded: construction is validated eagerly but synthesis stays
lazy, so these tests run offline without the ``tts`` extra installed.
"""

from __future__ import annotations

import asyncio

import pytest

from aca.config import Config, Tts
from aca.errors import ConfigError
from aca.tts import NullTtsEngine, SpeechController, build_tts_engine
from aca.tts.base import TtsEngine
from aca.tts.kokoro_engine import KokoroTtsEngine, _resolve_lang_code
from aca.tts.text import clean_for_speech


# --- config ---------------------------------------------------------------------------------
def test_tts_defaults_are_silent_and_onyx():
    cfg = Config()
    assert cfg.tts.provider == "none"
    assert cfg.tts.voice == "am_onyx"
    assert cfg.tts.speed == 1.0
    assert cfg.tts.sample_rate == 24_000


def test_tts_from_mapping_normalizes_provider_and_reads_fields():
    cfg = Config.from_mapping({"tts": {"provider": "Kokoro", "voice": "bf_emma", "speed": 1.2}})
    assert cfg.tts.provider == "kokoro"  # lower-cased
    assert cfg.tts.voice == "bf_emma"
    assert cfg.tts.speed == 1.2


@pytest.mark.parametrize("bad", [{"speed": 0}, {"speed": -1}, {"sample_rate": 0}])
def test_tts_rejects_nonpositive(bad):
    with pytest.raises(ConfigError):
        Tts.from_mapping(bad)


# --- text cleanup ---------------------------------------------------------------------------
def test_clean_for_speech_strips_markdown_and_keeps_link_label():
    assert clean_for_speech("**Hi** there `code` and [the link](http://x)") == "Hi there code and the link"


def test_clean_for_speech_drops_fenced_code_and_collapses_whitespace():
    assert clean_for_speech("say this ```print(1)``` now") == "say this now"
    assert clean_for_speech("  a\n\n  b  ") == "a b"


def test_clean_for_speech_can_be_empty():
    assert clean_for_speech("   ```only code```  ") == ""


# --- factory --------------------------------------------------------------------------------
def test_factory_none_returns_disabled_null_engine():
    engine = build_tts_engine(Tts())
    assert isinstance(engine, NullTtsEngine)
    assert engine.enabled is False


def test_factory_aliases_map_to_silent():
    assert isinstance(build_tts_engine(Tts(provider="off")), NullTtsEngine)


def test_factory_unknown_provider_raises():
    with pytest.raises(ConfigError):
        build_tts_engine(Tts(provider="festival"))


def test_factory_kokoro_builds_without_loading_model():
    engine = build_tts_engine(Tts(provider="kokoro", voice="am_onyx"))
    assert isinstance(engine, KokoroTtsEngine)
    assert engine.enabled is True  # constructing the engine must not import kokoro/torch


def test_kokoro_requires_a_voice():
    with pytest.raises(ConfigError):
        KokoroTtsEngine(voice="", lang_code="", speed=1.0, sample_rate=24_000)


def test_resolve_lang_code_derives_from_voice_prefix():
    assert _resolve_lang_code("am_onyx", "") == "a"  # American
    assert _resolve_lang_code("bf_emma", "") == "b"  # British
    assert _resolve_lang_code("am_onyx", "j") == "j"  # explicit override wins
    assert _resolve_lang_code("", "") == "a"  # safe fallback


# --- controller (serialized, non-blocking, fault-tolerant) ----------------------------------
class _RecordingEngine(TtsEngine):
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.closed = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def aclose(self) -> None:
        self.closed = True


async def _wait_for(cond, tries: int = 100) -> None:
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0)


async def test_controller_speaks_cleaned_text_in_order():
    engine = _RecordingEngine()
    ctrl = SpeechController(engine)
    ctrl.start()
    ctrl.submit("**one**")
    ctrl.submit("two")
    ctrl.submit("   ")  # whitespace-only -> nothing to say, dropped
    await _wait_for(lambda: len(engine.spoken) == 2)
    await ctrl.aclose()
    assert engine.spoken == ["one", "two"]
    assert engine.closed is True


async def test_controller_is_noop_when_engine_disabled():
    ctrl = SpeechController(NullTtsEngine())
    ctrl.start()
    ctrl.submit("hello")
    await asyncio.sleep(0)
    await ctrl.aclose()
    assert ctrl.enabled is False


async def test_controller_reports_engine_error_without_crashing():
    class _Boom(TtsEngine):
        async def speak(self, text: str) -> None:
            raise RuntimeError("no audio device")

    errors: list[BaseException] = []
    ctrl = SpeechController(_Boom(), on_error=errors.append)
    ctrl.start()
    ctrl.submit("x")
    await _wait_for(lambda: bool(errors))
    await ctrl.aclose()
    assert errors and isinstance(errors[0], RuntimeError)
