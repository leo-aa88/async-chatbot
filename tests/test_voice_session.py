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
