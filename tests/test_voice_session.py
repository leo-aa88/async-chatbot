"""End-to-end voice pipeline (offline): source -> VAD -> segmenter -> transcriber -> callback.

Uses the deterministic ``IterableSource`` + ``FakeTranscriber`` so the whole flow runs in CI with
no microphone and no Whisper model. Also covers the ``build_*`` factory selection and fail-fast.
"""

from __future__ import annotations

import struct

import pytest

from aca.config import Voice
from aca.errors import ConfigError
from aca.voice.audio import AudioFrame
from aca.voice.capture import IterableSource
from aca.voice.factory import build_segmenter, build_speech_detector, build_transcriber
from aca.voice.segmenter import Segmenter
from aca.voice.session import VoiceSession
from aca.voice.transcriber import FakeTranscriber
from aca.voice.vad import EnergyVad

SR = 16000
N = 160


def frame(amp: int) -> AudioFrame:
    return AudioFrame(pcm=struct.pack(f"<{N}h", *([amp] * N)), sample_rate=SR)


SIL, LOUD = frame(0), frame(8000)


def _session(frames, transcriber, *, min_chars=1, collect=None):
    seg = Segmenter(start_frames=1, silence_frames=2, min_speech_frames=1, max_frames=100, pre_roll_frames=1)
    return VoiceSession(
        IterableSource(frames), EnergyVad(0.02), seg, transcriber, collect, min_chars=min_chars
    )


async def test_session_transcribes_each_utterance():
    frames = [SIL, SIL, LOUD, LOUD, LOUD, SIL, SIL, SIL, LOUD, LOUD, SIL, SIL]
    got: list[str] = []

    async def on_transcript(text: str) -> None:
        got.append(text)

    await _session(frames, FakeTranscriber(["first", "second"]), collect=on_transcript).run()
    assert got == ["first", "second"]


async def test_session_flushes_trailing_speech_at_eof():
    frames = [SIL, LOUD, LOUD, LOUD]  # stream ends mid-utterance; flush must still emit it
    got: list[str] = []

    async def on_transcript(text: str) -> None:
        got.append(text)

    await _session(frames, FakeTranscriber(["tail"]), collect=on_transcript).run()
    assert got == ["tail"]


async def test_empty_transcript_is_not_injected():
    frames = [LOUD, LOUD, SIL, SIL]
    got: list[str] = []

    async def on_transcript(text: str) -> None:
        got.append(text)

    # An all-whitespace transcript (VAD false positive / non-speech noise) must not become a turn.
    await _session(frames, FakeTranscriber(["   "]), collect=on_transcript).run()
    assert got == []


async def test_transcribe_error_is_reported_and_capture_continues():
    # A failure on one utterance must be reported and NOT kill the background capture loop.
    class BoomOnce(FakeTranscriber):
        def __init__(self):
            super().__init__(["ok-second"])
            self.calls = 0

        async def transcribe(self, pcm, sample_rate):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model boom")
            return await super().transcribe(pcm, sample_rate)

    frames = [LOUD, LOUD, SIL, SIL, LOUD, LOUD, SIL, SIL]  # two utterances
    got: list[str] = []
    errors: list[tuple[str, str]] = []

    async def on_transcript(text: str) -> None:
        got.append(text)

    async def on_error(stage: str, exc: Exception) -> None:
        errors.append((stage, str(exc)))

    seg = Segmenter(start_frames=1, silence_frames=2, min_speech_frames=1, max_frames=100, pre_roll_frames=1)
    session = VoiceSession(IterableSource(frames), EnergyVad(0.02), seg, BoomOnce(), on_transcript, on_error=on_error)
    await session.run()
    assert errors == [("transcription", "model boom")]
    assert got == ["ok-second"]  # second utterance still transcribed -> loop stayed alive


async def test_capture_error_is_reported():
    class BoomSource:
        def frames(self):
            async def gen():
                raise RuntimeError("device busy")
                yield  # pragma: no cover - unreachable, marks this an async generator

            return gen()

        def close(self):
            pass

    errors: list[tuple[str, str]] = []

    async def on_transcript(text: str) -> None:
        pass

    async def on_error(stage: str, exc: Exception) -> None:
        errors.append((stage, str(exc)))

    seg = Segmenter(start_frames=1, silence_frames=2, min_speech_frames=1, max_frames=10, pre_roll_frames=0)
    session = VoiceSession(BoomSource(), EnergyVad(0.02), seg, FakeTranscriber(), on_transcript, on_error=on_error)
    await session.run()  # must not raise
    assert errors == [("capture", "device busy")]


# --- factory --------------------------------------------------------------------------------
def test_build_transcriber_default_is_fake():
    assert isinstance(build_transcriber(Voice()), FakeTranscriber)


def test_build_transcriber_rejects_unknown_provider():
    with pytest.raises(ConfigError):
        build_transcriber(Voice(provider="bogus"))


def test_build_speech_detector_default_is_energy():
    assert isinstance(build_speech_detector(Voice()), EnergyVad)


def test_build_speech_detector_rejects_unknown_vad():
    with pytest.raises(ConfigError):
        build_speech_detector(Voice(vad="bogus"))


def test_build_segmenter_derives_frame_counts():
    seg = build_segmenter(Voice(frame_seconds=0.03, start_seconds=0.15, silence_seconds=0.6))
    assert seg._start_frames == 5     # 0.15 / 0.03
    assert seg._silence_frames == 20  # 0.60 / 0.03


# --- Whisper compute-type / device resolution (pure, no WhisperModel) ----------------------
def test_compute_type_auto_matches_resolved_device():
    from aca.voice.whisper import _resolve_compute_type

    assert _resolve_compute_type("auto", "cuda") == "int8_float16"
    assert _resolve_compute_type("auto", "cpu") == "int8"  # CPU-runnable default, not a CUDA type


def test_compute_type_cuda_only_on_cpu_is_rejected():
    from aca.voice.whisper import _resolve_compute_type

    # The exact failure the reviewer flagged: int8_float16 on a CPU device. Reject cleanly (before
    # any model download) rather than let CTranslate2 raise a raw ValueError mid-load.
    for cuda_only in ("int8_float16", "float16", "bfloat16"):
        with pytest.raises(ConfigError):
            _resolve_compute_type(cuda_only, "cpu")


def test_compute_type_explicit_cpu_type_passes_through():
    from aca.voice.whisper import _resolve_compute_type

    assert _resolve_compute_type("int8", "cpu") == "int8"
    assert _resolve_compute_type("float32", "cpu") == "float32"


def test_resolve_device_passes_explicit_through():
    from aca.voice.whisper import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"


def test_build_model_auto_falls_back_to_cpu_when_cuda_load_fails(monkeypatch):
    # A present-but-unusable GPU: device auto -> cuda, but the CUDA model load raises (old driver /
    # missing libcublas). _build_model must retry on CPU rather than crash --voice.
    from aca.voice import whisper

    monkeypatch.setattr(whisper, "_resolve_device", lambda d: "cuda" if d == "auto" else d)
    attempts = []

    def build(model, device, compute):
        attempts.append((device, compute))
        if device == "cuda":
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        return f"model@{device}/{compute}"

    result, resolved = whisper._build_model(build, "small.en", "auto", "auto")
    assert resolved == "cpu"
    assert result == "model@cpu/int8"
    assert [d for d, _ in attempts] == ["cuda", "cpu"]  # tried CUDA, then fell back


def test_build_model_explicit_cuda_reraises(monkeypatch):
    # An explicit device="cuda" is the user asking for CUDA — surface the load error, don't silently
    # downgrade to CPU.
    from aca.voice import whisper

    monkeypatch.setattr(whisper, "_resolve_device", lambda d: d)

    def build(model, device, compute):
        raise RuntimeError("libcublas missing")

    with pytest.raises(RuntimeError):
        whisper._build_model(build, "small.en", "cuda", "int8_float16")
