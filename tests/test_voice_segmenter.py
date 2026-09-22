"""The pure VAD/segmentation core: AudioFrame, EnergyVad, and the turn state machine.

Frame-based and clock-free, so these assert exact boundary behavior deterministically — the same
determinism discipline the cognition core follows.
"""

from __future__ import annotations

import struct

from aca.voice.audio import AudioFrame
from aca.voice.segmenter import Segmenter
from aca.voice.vad import EnergyVad

SR = 16000
N = 480  # 30 ms at 16 kHz


def frame(amp: int, n: int = N, sr: int = SR) -> AudioFrame:
    return AudioFrame(pcm=struct.pack(f"<{n}h", *([amp] * n)), sample_rate=sr)


SIL = frame(0)
LOUD = frame(8000)  # rms ~0.24, well above a 0.02 gate


# --- AudioFrame ---------------------------------------------------------------------------
def test_frame_rms_and_duration():
    assert SIL.rms() == 0.0
    assert LOUD.rms() > 0.2
    assert abs(LOUD.duration_seconds - N / SR) < 1e-9
    assert LOUD.sample_count == N


def test_empty_frame_is_silent():
    assert AudioFrame(pcm=b"", sample_rate=SR).rms() == 0.0
    assert AudioFrame(pcm=b"", sample_rate=SR).duration_seconds == 0.0


# --- EnergyVad ----------------------------------------------------------------------------
def test_energy_vad_thresholds():
    vad = EnergyVad(0.02)
    assert vad.is_speech(LOUD) is True
    assert vad.is_speech(SIL) is False
    assert EnergyVad(0.5).is_speech(LOUD) is False  # raise the gate above the signal


# --- Segmenter ----------------------------------------------------------------------------
def _seg(**kw) -> Segmenter:
    defaults = dict(start_frames=2, silence_frames=3, min_speech_frames=2, max_frames=100, pre_roll_frames=2)
    defaults.update(kw)
    return Segmenter(**defaults)


def _drive(seg: Segmenter, pattern: str):
    """Feed a pattern string of 'x' (speech) / '.' (silence); collect emitted utterances."""
    out = []
    for ch in pattern:
        utt = seg.push(LOUD if ch == "x" else SIL, ch == "x")
        if utt is not None:
            out.append(utt)
    return out


def test_single_clean_utterance():
    seg = _seg()
    utterances = _drive(seg, "...xxxxx...")  # onset after 2 speech, close after 3 silence
    assert len(utterances) == 1
    assert utterances[0].sample_rate == SR
    assert utterances[0].duration_seconds > 0

def test_onset_debounce_ignores_single_blip():
    seg = _seg(start_frames=3)
    # Two isolated speech frames never reach the 3-frame onset, so no turn ever begins.
    assert _drive(seg, ".x.x.x.....") == []


def test_short_utterance_dropped_as_blip():
    seg = _seg(start_frames=1, min_speech_frames=5)
    # Onset at the first speech frame, but only 3 speech frames < 5 required -> dropped.
    assert _drive(seg, "xxx....") == []


def test_two_turns_separated_by_silence():
    seg = _seg(start_frames=1, silence_frames=2, min_speech_frames=1)
    utterances = _drive(seg, "xx...xx...")
    assert len(utterances) == 2


def test_max_frames_force_cut():
    seg = _seg(start_frames=1, silence_frames=99, min_speech_frames=1, max_frames=4, pre_roll_frames=0)
    utterances = _drive(seg, "xxxxxxxx")  # never goes silent; must be force-cut at 4 frames
    assert len(utterances) == 2
    assert all(len(u.frames) <= 4 for u in utterances)


def test_flush_closes_in_progress_turn():
    seg = _seg(start_frames=1, min_speech_frames=1)
    assert _drive(seg, "xxxx") == []  # no trailing silence yet
    tail = seg.flush()
    assert tail is not None and tail.duration_seconds > 0
    assert seg.flush() is None  # idempotent once drained


def test_pre_roll_prepended_so_onset_is_not_clipped():
    seg = _seg(start_frames=2, silence_frames=2, min_speech_frames=1, pre_roll_frames=2)
    utterances = _drive(seg, "..xxx..")
    assert len(utterances) == 1
    # pre-roll (2 silent) + onset speech captured -> more frames than just the speech run
    assert len(utterances[0].frames) >= 4


# --- Silero windowing core (pure, no torch) -----------------------------------------------
def test_silero_windowing_accumulates_subwindow_frames():
    from aca.voice.silero import _WindowedDetector

    # A 480-sample capture frame is smaller than the 512-sample window; frames must accumulate,
    # not be classified as permanent silence (the bug that swallowed the mic at default sizes).
    scored: list[int] = []

    def score(window: bytes) -> float:
        scored.append(len(window) // 2)  # samples in the window
        return 1.0  # always "speech"

    det = _WindowedDetector(window_samples=512, threshold=0.5, score=score)
    frame480 = b"\x01\x00" * 480
    assert det.push(frame480) is False   # 480 < 512: no window yet, holds initial verdict
    assert det.push(frame480) is True    # 960 total: one 512 window scored -> speech
    assert scored == [512]
    assert det.push(b"") is True          # verdict held between windows


def test_silero_windowing_reset_clears_buffer_and_verdict():
    from aca.voice.silero import _WindowedDetector

    det = _WindowedDetector(512, 0.5, lambda w: 1.0)
    det.push(b"\x01\x00" * 512)
    assert det.push(b"") is True
    det.reset()
    assert det.push(b"") is False  # buffer and last verdict both cleared


def test_silero_rejects_unsupported_sample_rate():
    import pytest

    from aca.errors import ConfigError
    from aca.voice.silero import SileroVad

    # Rejected at construction, before the torch import, so this holds without the extra installed.
    with pytest.raises(ConfigError):
        SileroVad(sample_rate=44100)
