"""faster-whisper transcriber (CTranslate2), imported lazily behind the ``voice`` extra.

Wraps a single loaded ``WhisperModel`` and transcribes accumulated PCM. On a 4 GB GPU ``small.en``
at ``int8_float16`` needs ~2 GB VRAM (English-only); the smaller, faster ``distil-small.en`` trades
a little accuracy for latency. Both load through this same class — only ``config.voice.model``
changes. With no CUDA present the model still runs on CPU (``int8``), just slower.

``device`` and ``compute_type`` both default to ``auto``: the device resolves to CUDA when present
else CPU, and the compute type is then chosen to match (``int8_float16`` on CUDA, ``int8`` on CPU).
A CUDA-only compute type explicitly configured against a CPU device is rejected as a ``ConfigError``
*before* the weights download, rather than surfacing CTranslate2's raw ``ValueError`` mid-load.

The model is loaded once at construction (so the first utterance isn't slow) and each transcription
runs in a worker thread via ``asyncio.to_thread`` — a Whisper forward pass must never block the
client's event loop.
"""

from __future__ import annotations

import asyncio

from ..errors import ConfigError
from .transcriber import Transcript

_INT16_FULL_SCALE = 32768.0
# CTranslate2 compute types that require a CUDA device; picking one on CPU raises at model load.
_CUDA_ONLY_COMPUTE = frozenset({"int8_float16", "float16", "int8_bfloat16", "bfloat16"})
_CUDA_DEFAULT_COMPUTE = "int8_float16"
_CPU_DEFAULT_COMPUTE = "int8"


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` to ``cuda`` when a CUDA device is present, else ``cpu`` (explicit passes through)."""
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # pragma: no cover - defensive: treat any probe failure as CPU
        return "cpu"


def _build_model(build, model: str, requested_device: str, compute_type: str):
    """Construct the model, falling back ``auto``→CPU when a *present but unusable* GPU fails to load.

    A GPU can be detected (``get_cuda_device_count() > 0``) yet be unable to run CUDA — an old driver
    or a missing ``libcublas`` makes CTranslate2 raise partway through the CUDA model load. When the
    device was auto-selected we retry on CPU rather than crash ``--voice``; an *explicit*
    ``device="cuda"`` is left to surface the error, since the user asked for CUDA specifically.
    Returns ``(model, resolved_device)``. ``build(model, device, compute)`` does the real load
    (injected so the fallback is testable without the ``faster-whisper`` extra).
    """
    device = _resolve_device(requested_device)
    compute = _resolve_compute_type(compute_type, device)
    try:
        return build(model, device, compute), device
    except Exception:
        if requested_device != "auto" or device != "cuda":
            raise
        device = "cpu"
        compute = _resolve_compute_type(compute_type, device)
        return build(model, device, compute), device


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick a compute type the resolved device can actually run.

    ``auto`` maps to the device's efficient default. An explicit CUDA-only type on a CPU device is a
    configuration error we surface cleanly (before any download) instead of letting CTranslate2 raise
    a raw ``ValueError`` partway through loading.
    """
    if compute_type == "auto":
        return _CUDA_DEFAULT_COMPUTE if device == "cuda" else _CPU_DEFAULT_COMPUTE
    if device != "cuda" and compute_type in _CUDA_ONLY_COMPUTE:
        raise ConfigError(
            f"voice.compute_type {compute_type!r} requires a CUDA device, but the resolved device "
            f"is {device!r}; set voice.compute_type='auto' (or 'int8') for CPU."
        )
    return compute_type


class FasterWhisperTranscriber:
    """A ``Transcriber`` backed by ``faster_whisper.WhisperModel``."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ConfigError(
                "the 'faster-whisper' package is required for voice.provider 'faster-whisper': "
                "pip install 'aca[voice]'"
            ) from exc
        try:
            import numpy  # noqa: F401  (required to marshal PCM into the model)
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "the 'numpy' package is required for voice transcription: pip install 'aca[voice]'"
            ) from exc

        self._language = language or None
        self._beam_size = beam_size
        # Resolve device + compute type (rejecting a CPU/CUDA-only mismatch before the download), and
        # fall back auto→CPU if a present-but-unusable GPU fails to load (old driver / missing libs).
        self._model, _ = _build_model(
            lambda m, d, c: WhisperModel(m, device=d, compute_type=c),
            model, device, compute_type,
        )

    async def transcribe(self, pcm: bytes, sample_rate: int) -> Transcript:
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> Transcript:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE
        segments, info = self._model.transcribe(
            audio, language=self._language, beam_size=self._beam_size
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration = len(audio) / sample_rate if sample_rate else 0.0
        language = getattr(info, "language", None) or (self._language or "")
        return Transcript(text=text, language=language, duration_seconds=duration)
