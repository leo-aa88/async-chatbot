"""Speech output (DESIGN 2.2 / 6.1): config, spoken-text normalization, factory, playback control.

Everything here is client-side rendering — no reducer, no durable state, no injected clock/rng.
The Kokoro model itself is never loaded: construction is validated eagerly but synthesis stays
lazy, so these tests run offline without the ``tts`` extra installed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

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


def test_tts_from_mapping_normalizes_provider_and_reads_fields():
    cfg = Config.from_mapping({"tts": {"provider": "Kokoro", "voice": "bf_emma", "speed": 1.2}})
    assert cfg.tts.provider == "kokoro"  # lower-cased
    assert cfg.tts.voice == "bf_emma"
    assert cfg.tts.speed == 1.2


def test_tts_rejects_nonpositive_speed():
    with pytest.raises(ConfigError):
        Tts.from_mapping({"speed": 0})


def test_tts_has_no_sample_rate_knob():
    # Kokoro's rate is a fixed constant, not configuration (see KOKORO_SAMPLE_RATE); a stray key is
    # ignored rather than becoming a mis-clocking playback knob.
    assert not hasattr(Tts(), "sample_rate")
    assert Tts.from_mapping({"sample_rate": 48000}) == Tts()


# --- spoken-text normalization --------------------------------------------------------------
def test_clean_unwraps_emphasis_code_and_links():
    assert clean_for_speech("**Hi** there `code` and [the link](http://x)") == "Hi there code and the link"
    assert clean_for_speech("_stressed_ and *also*") == "stressed and also"


def test_clean_leaves_non_markdown_punctuation_intact():
    # The named regressions from review: mid-word '#', a spaced '>', and snake_case underscores.
    assert clean_for_speech("I use C# daily") == "I use C# daily"
    assert clean_for_speech("if a > b then") == "if a > b then"
    assert clean_for_speech("call read_file_content now") == "call read_file_content now"


def test_clean_strips_line_leading_heading_and_quote_markers():
    assert clean_for_speech("# Title\n> quoted") == "Title\nquoted"


def test_clean_keeps_fenced_code_content_never_silent():
    assert clean_for_speech("```py\nprint(1)\n```") == "print(1)"
    assert clean_for_speech("```only inline```") == "only inline"


def test_clean_preserves_newlines_but_collapses_spaces():
    # Kokoro splits on newlines; keep them, collapse only intra-line runs of spaces.
    assert clean_for_speech("one\ntwo   three") == "one\ntwo three"
    assert clean_for_speech("a\n\n\nb") == "a\nb"


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
        KokoroTtsEngine(voice="", lang_code="", speed=1.0)


def test_resolve_lang_code_derives_from_voice_prefix():
    assert _resolve_lang_code("am_onyx", "") == "a"  # American
    assert _resolve_lang_code("bf_emma", "") == "b"  # British
    assert _resolve_lang_code("am_onyx", "j") == "j"  # explicit override wins
    assert _resolve_lang_code("", "") == "a"  # safe fallback


async def test_render_never_redirects_terminal_streams(monkeypatch):
    # Regression guard: the worker thread must not reassign the process-global sys.stdout/sys.stderr
    # that ChatUI writes from — redirecting them would swallow keystroke echo and delivered messages
    # for the whole utterance. Inject fake libs and assert the streams are untouched during synthesis
    # (and that repo_id is passed and Kokoro's logger disabled, the at-source suppressions).
    original_stdout, original_stderr = sys.stdout, sys.stderr
    seen: dict = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def __call__(self, text, voice, speed):
            seen["stdout_ok"] = sys.stdout is original_stdout
            seen["stderr_ok"] = sys.stderr is original_stderr
            yield ("g", "p", [0.0, 0.1])

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=_FakePipeline))
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(asarray=lambda a, dtype=None: a))
    monkeypatch.setitem(
        sys.modules, "sounddevice",
        types.SimpleNamespace(play=lambda *a, **k: None, wait=lambda: None, stop=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules, "loguru",
        types.SimpleNamespace(logger=types.SimpleNamespace(disable=lambda n: seen.__setitem__("disabled", n))),
    )

    engine = KokoroTtsEngine(voice="am_onyx", lang_code="", speed=1.0)
    await engine.speak("hello")

    assert seen["kwargs"] == {"lang_code": "a", "repo_id": "hexgrad/Kokoro-82M"}
    assert seen["stdout_ok"] is True and seen["stderr_ok"] is True  # no global redirect held
    assert seen["disabled"] == "kokoro"  # Kokoro's loguru records silenced at source
    assert sys.stdout is original_stdout and sys.stderr is original_stderr  # untouched afterward


async def test_kokoro_aclose_stops_playback(monkeypatch):
    # aclose must call sd.stop() so an in-flight sd.wait() is interrupted at shutdown. Inject a fake
    # sounddevice (the real one isn't installed) and assert stop() is invoked.
    stopped = threading.Event()
    fake_sd = types.SimpleNamespace(stop=stopped.set)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    engine = KokoroTtsEngine(voice="am_onyx", lang_code="", speed=1.0)
    await engine.aclose()
    assert stopped.is_set()


# --- controller (serialized, non-blocking, interruptible, fault-latching) -------------------
class _RecordingEngine(TtsEngine):
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.closed = False

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def aclose(self) -> None:
        self.closed = True


async def _wait_for(cond, tries: int = 200) -> None:
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


async def test_shutdown_interrupts_in_flight_speak():
    # A worker blocked inside speak (mirroring Kokoro's to_thread + sd.wait) must be released by
    # aclose, not left to finish on its own — otherwise Ctrl-D hangs until the utterance ends.
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class _Blocking(TtsEngine):
        async def speak(self, text: str) -> None:
            await asyncio.to_thread(self._render)

        def _render(self) -> None:
            started.set()
            release.wait(timeout=5)  # ONLY aclose should free us; a timeout means aclose failed to
            finished.set()

        async def aclose(self) -> None:
            release.set()  # engine owns interruption (Kokoro's sd.stop() analogue)

    ctrl = SpeechController(_Blocking())
    ctrl.start()
    ctrl.submit("x")
    await _wait_for(lambda: started.is_set())
    await asyncio.wait_for(ctrl.aclose(), timeout=2)
    assert release.is_set()  # aclose actively signaled the stop
    await _wait_for(lambda: finished.is_set())  # and the worker actually unwound


async def test_controller_latches_off_after_engine_failure():
    # A persistently broken engine must surface ONE error and then stop, not error per message.
    errors: list[BaseException] = []

    class _Broken(TtsEngine):
        calls = 0

        async def speak(self, text: str) -> None:
            type(self).calls += 1
            raise RuntimeError("no audio device")

    engine = _Broken()
    ctrl = SpeechController(engine, on_error=errors.append)
    ctrl.start()
    ctrl.submit("first")
    await _wait_for(lambda: bool(errors))
    assert ctrl.enabled is False  # latched off
    ctrl.submit("second")  # dropped by submit (disabled)
    await asyncio.sleep(0)
    await ctrl.aclose()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert engine.calls == 1  # never retried the dead device


async def test_speaker_survives_a_raising_error_callback():
    # If on_error itself raises, the speaker task must not die and quitting must not re-raise.
    def _boom(_exc: BaseException) -> None:
        raise ValueError("handler is broken")

    class _Broken(TtsEngine):
        async def speak(self, text: str) -> None:
            raise RuntimeError("device gone")

    ctrl = SpeechController(_Broken(), on_error=_boom)
    ctrl.start()
    ctrl.submit("x")
    await _wait_for(lambda: ctrl.enabled is False)  # failure latched despite the broken callback
    assert ctrl._task is not None and not ctrl._task.done()  # task still alive, not crashed
    await asyncio.wait_for(ctrl.aclose(), timeout=2)  # clean shutdown, no traceback escaping
