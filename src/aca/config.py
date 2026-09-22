"""Configuration model (DESIGN 18, 25).

The v0 parameter space is deliberately small; each parameter has one primary mechanism
(DESIGN 18). Config is immutable once loaded. Duration-like fields accept human-readable
strings (``45m``) and are converted to explicit model units (hours/seconds) at load time so
no implicit unit conversion happens later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .durations import parse_hours, parse_seconds
from .errors import ConfigError


def _fraction(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number in [0, 1]") from exc
    if not 0.0 <= number <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], got {number}")
    return number


@dataclass(frozen=True, slots=True)
class Temperament:
    """Priors that shape *expression*, not thought opportunity (DESIGN 11, 17, 18)."""

    initiative: float = 0.50
    inhibition: float = 0.50
    persistence: float = 0.50

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Temperament:
        return Temperament(
            initiative=_fraction("temperament.initiative", data.get("initiative", 0.50)),
            inhibition=_fraction("temperament.inhibition", data.get("inhibition", 0.50)),
            persistence=_fraction("temperament.persistence", data.get("persistence", 0.50)),
        )


@dataclass(frozen=True, slots=True)
class QuietHours:
    """A hard gate on unsolicited outward speech during a local-time interval (DESIGN 11.6)."""

    enabled: bool = False
    start_local: str = "01:00"
    end_local: str = "08:00"

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> QuietHours:
        return QuietHours(
            enabled=bool(data.get("enabled", False)),
            start_local=str(data.get("start_local", "01:00")),
            end_local=str(data.get("end_local", "08:00")),
        )


@dataclass(frozen=True, slots=True)
class Timing:
    """Expression-timing constants. Values are stored in explicit units."""

    refractory_tau_hours: float = 2.0
    proactive_cooldown_seconds: float = parse_seconds("45m")
    proactive_burst_window_seconds: float = parse_seconds("5m")
    proactive_ttl_seconds: float = parse_seconds("6h")
    quiet_hours: QuietHours = field(default_factory=QuietHours)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Timing:
        return Timing(
            refractory_tau_hours=parse_hours(data.get("refractory_tau_hours", 2.0)),
            proactive_cooldown_seconds=parse_seconds(data.get("proactive_cooldown", "45m")),
            proactive_burst_window_seconds=parse_seconds(
                data.get("proactive_burst_window", "5m")
            ),
            proactive_ttl_seconds=parse_seconds(data.get("proactive_ttl", "6h")),
            quiet_hours=QuietHours.from_mapping(data.get("quiet_hours", {})),
        )


@dataclass(frozen=True, slots=True)
class Cognition:
    """Internal-activation (thought-opportunity) parameters (DESIGN 10, 12)."""

    spontaneous_activation_rate_per_hour: float = 0.25
    selection_temperature: float = 0.7
    null_candidate_score: float = 0.5
    semantic_worthiness_floor: float = 0.35

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Cognition:
        rate = float(data.get("spontaneous_activation_rate_per_hour", 0.25))
        if rate < 0.0:
            raise ConfigError("cognition.spontaneous_activation_rate_per_hour must be >= 0")
        temperature = float(data.get("selection_temperature", 0.7))
        if temperature <= 0.0:
            raise ConfigError("cognition.selection_temperature must be > 0")
        return Cognition(
            spontaneous_activation_rate_per_hour=rate,
            selection_temperature=temperature,
            null_candidate_score=float(data.get("null_candidate_score", 0.5)),
            semantic_worthiness_floor=_fraction(
                "cognition.semantic_worthiness_floor",
                data.get("semantic_worthiness_floor", 0.35),
            ),
        )


@dataclass(frozen=True, slots=True)
class Conversation:
    """Cadence thresholds for inferring conversation mode (DESIGN 7.4, open tuning 31.1).

    A human turn within ``active_within`` keeps the conversation ACTIVE (proactive initiative
    suppressed); within ``idle_within`` it is IDLE; beyond that, DORMANT. Configurable so
    proactive behavior can be exercised without waiting the production defaults.
    """

    active_within_seconds: float = parse_seconds("3m")
    idle_within_seconds: float = parse_seconds("30m")

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Conversation:
        active = parse_seconds(data.get("active_within", "3m"))
        idle = parse_seconds(data.get("idle_within", "30m"))
        if idle < active:
            raise ConfigError("conversation.idle_within must be >= active_within")
        return Conversation(active_within_seconds=active, idle_within_seconds=idle)


@dataclass(frozen=True, slots=True)
class Memory:
    """Memory-availability parameters (DESIGN 12)."""

    default_decay_half_life_hours: float = 24.0
    persistence_decay_coefficient: float = 0.6  # alpha in k_eff = k_base(1 - alpha*P)
    candidate_activation_floor: float = 0.05
    # A candidate the agent just spoke about proactively is excluded from re-selection for this
    # long, so it doesn't nag (DESIGN 6, 11.3, 12.7 repetition regulation). Human-scale by default.
    repeat_suppression_seconds: float = parse_seconds("6h")
    # Quality gate: only memories at least this salient are enriched into topics (DESIGN 12.3).
    # Below it, a memory stays RAW/retrievable but never becomes a topic — keeps junk out.
    enrichment_salience_floor: float = 0.6
    # Topic de-duplication (lexical path): when a new enrichment's summary is at least this similar
    # (0..1) to an existing topic, reinforce that topic instead of creating a near-duplicate (DESIGN
    # 12.3). 0 disables the LEXICAL path only; the semantic path (topic_dedup_cosine) is separate, so
    # to disable merging entirely set BOTH to 0. Conservative default — near-identical wording only.
    topic_merge_similarity: float = 0.8
    # Semantic-neighborhood threshold (cosine, 0..1) for grouping distinct utterances that mean
    # roughly the same thing — used by the observational topic-dominance metric (and, later, semantic
    # repeated-topic). Deliberately looser than topic_merge_similarity: "same subject", not "duplicate".
    semantic_neighbor_threshold: float = 0.78
    # Semantic de-dup threshold (cosine, 0..1): a new enrichment whose embedding is at least this
    # close to an existing topic's is treated as a near-duplicate and merged, catching paraphrases
    # the lexical check misses. Deliberately VERY high (> semantic_neighbor_threshold) so related-
    # but-distinct topics ("robot embodiment" vs "robot safety") are not merged. 0 disables the
    # semantic path (set both this and topic_merge_similarity to 0 to disable all merging). τ_dedup.
    topic_dedup_cosine: float = 0.94
    # Discourse-focus gate (DESIGN §34, v0.7): affinity (cosine, 0..1) of a proactive candidate to
    # the current conversation focus. >= continue is CONTINUE, >= bridge is BRIDGE (both may speak);
    # below bridge is ORPHAN (suppressed while conversation mode is IDLE). These are their OWN keys
    # and must not alias the observational cuts above (topic_dedup_cosine / semantic_neighbor_
    # threshold) — sharing a cut would make the gate mechanically drive the advance-rate metric
    # (Goodhart, DESIGN 34.5). ``discourse_bridge_cosine <= 0`` disables the gate.
    discourse_continue_cosine: float = 0.75
    discourse_bridge_cosine: float = 0.55

    def __post_init__(self) -> None:
        # The ORPHAN boundary is discourse_bridge_cosine and must sit at or below the CONTINUE
        # boundary, or the bands are inverted and the advertised gate boundary is meaningless.
        # Enforced here so a *direct* Memory(...) construction can't bypass it either (DESIGN 34.5).
        if self.discourse_bridge_cosine > self.discourse_continue_cosine:
            raise ConfigError(
                "memory.discourse_bridge_cosine must be <= memory.discourse_continue_cosine"
            )

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Memory:
        half_life = float(data.get("default_decay_half_life_hours", 24.0))
        if half_life <= 0.0:
            raise ConfigError("memory.default_decay_half_life_hours must be > 0")
        return Memory(
            default_decay_half_life_hours=half_life,
            persistence_decay_coefficient=_fraction(
                "memory.persistence_decay_coefficient",
                data.get("persistence_decay_coefficient", 0.6),
            ),
            candidate_activation_floor=_fraction(
                "memory.candidate_activation_floor",
                data.get("candidate_activation_floor", 0.05),
            ),
            repeat_suppression_seconds=parse_seconds(data.get("repeat_suppression", "6h")),
            enrichment_salience_floor=_fraction(
                "memory.enrichment_salience_floor",
                data.get("enrichment_salience_floor", 0.6),
            ),
            topic_merge_similarity=_fraction(
                "memory.topic_merge_similarity",
                data.get("topic_merge_similarity", 0.8),
            ),
            semantic_neighbor_threshold=_fraction(
                "memory.semantic_neighbor_threshold",
                data.get("semantic_neighbor_threshold", 0.78),
            ),
            topic_dedup_cosine=_fraction(
                "memory.topic_dedup_cosine",
                data.get("topic_dedup_cosine", 0.94),
            ),
            discourse_continue_cosine=_fraction(
                "memory.discourse_continue_cosine",
                data.get("discourse_continue_cosine", 0.75),
            ),
            discourse_bridge_cosine=_fraction(
                "memory.discourse_bridge_cosine",
                data.get("discourse_bridge_cosine", 0.55),
            ),
        )


@dataclass(frozen=True, slots=True)
class Budgets:
    """Hard proactive-cognition limits that never bank across downtime (DESIGN 25)."""

    proactive_llm_calls_per_day: int = 12
    proactive_llm_calls_per_hour: int = 2
    proactive_messages_per_hour: int = 2
    max_enrichment_attempts: int = 3

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Budgets:
        def positive_int(name: str, value: Any) -> int:
            number = int(value)
            if number < 0:
                raise ConfigError(f"{name} must be >= 0")
            return number

        return Budgets(
            proactive_llm_calls_per_day=positive_int(
                "budgets.proactive_llm_calls_per_day",
                data.get("proactive_llm_calls_per_day", 12),
            ),
            proactive_llm_calls_per_hour=positive_int(
                "budgets.proactive_llm_calls_per_hour",
                data.get("proactive_llm_calls_per_hour", 2),
            ),
            proactive_messages_per_hour=positive_int(
                "budgets.proactive_messages_per_hour",
                data.get("proactive_messages_per_hour", 2),
            ),
            max_enrichment_attempts=positive_int(
                "budgets.max_enrichment_attempts", data.get("max_enrichment_attempts", 3)
            ),
        )


@dataclass(frozen=True, slots=True)
class LLM:
    """Which generative provider backs the slow loop (DESIGN 22, 28.5).

    Provider-agnostic: ``provider`` selects an adapter (``fake`` is the deterministic offline
    default; ``openai``/``grok``/``gemini`` share the OpenAI-style chat API; ``anthropic`` uses
    the Messages API). Credentials come from the environment (``api_key_env``); ``base_url`` may
    be overridden if an endpoint moves. The model is never hard-coded — set it here.
    """

    provider: str = "fake"
    model: str = ""
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    base_url: str | None = None
    api_key_env: str | None = None

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> LLM:
        max_tokens = int(data.get("max_tokens", 1024))
        if max_tokens <= 0:
            raise ConfigError("llm.max_tokens must be > 0")
        return LLM(
            provider=str(data.get("provider", "fake")).strip().lower(),
            model=str(data.get("model", "")),
            max_tokens=max_tokens,
            timeout_seconds=parse_seconds(data.get("timeout", "60s")),
            base_url=(str(data["base_url"]) if data.get("base_url") else None),
            api_key_env=(str(data["api_key_env"]) if data.get("api_key_env") else None),
        )


@dataclass(frozen=True, slots=True)
class Embedding:
    """Which worker produces provisional-memory embeddings (DESIGN 12.2, 28.4).

    ``fake`` (the default) is the deterministic offline hashing worker — good for tests but not
    semantic. ``openai`` (and other OpenAI-compatible endpoints via ``base_url``) produce real
    semantic vectors. Model must be set for a real provider (e.g. ``text-embedding-3-small``).

    IMPORTANT — this choice differs from ``llm.provider``. DESIGN 28.4 makes embeddings *local from
    v0* precisely because DESIGN 12.2 embeds *nearly every substantive human message*, unclassified.
    So opting into a real provider sends most of what the user types to a third party, at a
    frequency well above the (rate/budget-gated) generative calls, with real per-call cost and a new
    timeout/429/5xx failure surface on what was the cheapest part of the pipeline. It's a reasonable
    opt-in — the operator owns the data and the account, and real semantics unlock features the fake
    worker can't (semantic dedup, repeated-topic, dominance) — but it is not free or private the way
    the default is. Left at ``fake`` unless deliberately configured.
    """

    provider: str = "fake"
    model: str = ""
    timeout_seconds: float = 30.0
    base_url: str | None = None
    api_key_env: str | None = None

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Embedding:
        return Embedding(
            provider=str(data.get("provider", "fake")).strip().lower(),
            model=str(data.get("model", "")),
            timeout_seconds=parse_seconds(data.get("timeout", "30s")),
            base_url=(str(data["base_url"]) if data.get("base_url") else None),
            api_key_env=(str(data["api_key_env"]) if data.get("api_key_env") else None),
        )


@dataclass(frozen=True, slots=True)
class Voice:
    """Client-side speech-to-text for the ``aca chat --voice`` perception path (DESIGN 13, 28.2).

    Voice is a *perception adapter*, not cognition: the client captures a microphone, a lightweight
    VAD carves it into conversational turns, a Whisper model transcribes each utterance, and the
    text is injected through the *same* durable ingress as typed input (a ``HumanMessage`` tagged
    ``input_mode="voice"``). Nothing here runs in the daemon core or touches the reducer, so it
    cannot affect determinism or replay — the transcript is just another human utterance.

    ``provider`` selects the transcriber: ``fake`` (deterministic, offline, for tests) or
    ``faster-whisper`` (CTranslate2; English-only ``small.en`` at ``int8_float16`` fits a 4 GB GPU).
    ``vad`` selects the segmenter's speech detector: ``energy`` (dependency-free RMS gate, the
    default) or ``silero`` (optional, more robust). All timings are seconds; frame counts are
    derived from ``frame_seconds`` at build time.
    """

    provider: str = "fake"
    model: str = "small.en"
    device: str = "auto"
    # ``auto`` picks a compute type the resolved device can run (``int8_float16`` on CUDA, ``int8``
    # on CPU); an explicit CUDA-only type on a CPU device is rejected before the model downloads.
    compute_type: str = "auto"
    language: str = "en"
    beam_size: int = 5
    sample_rate: int = 16000
    frame_seconds: float = 0.03
    # Segmentation (VAD boundaries). Onset debounce, end-of-utterance silence (hangover), the
    # blip/runaway bounds, and pre-roll kept so speech onset isn't clipped.
    vad: str = "energy"
    vad_threshold: float = 0.02  # RMS in [0, 1]; frames at/above this are speech (energy VAD only)
    start_seconds: float = 0.15
    silence_seconds: float = 0.6
    min_utterance_seconds: float = 0.3
    max_utterance_seconds: float = 30.0
    pre_roll_seconds: float = 0.2

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Voice:
        sample_rate = int(data.get("sample_rate", 16000))
        if sample_rate <= 0:
            raise ConfigError("voice.sample_rate must be > 0")
        beam_size = int(data.get("beam_size", 5))
        if beam_size <= 0:
            raise ConfigError("voice.beam_size must be > 0")
        frame_seconds = parse_seconds(data.get("frame_seconds", 0.03))
        if frame_seconds <= 0.0:
            raise ConfigError("voice.frame_seconds must be > 0")
        min_utterance = parse_seconds(data.get("min_utterance", 0.3))
        max_utterance = parse_seconds(data.get("max_utterance", 30.0))
        if max_utterance < min_utterance:
            raise ConfigError("voice.max_utterance must be >= min_utterance")
        return Voice(
            provider=str(data.get("provider", "fake")).strip().lower(),
            model=str(data.get("model", "small.en")),
            device=str(data.get("device", "auto")).strip().lower(),
            compute_type=str(data.get("compute_type", "auto")).strip().lower(),
            language=str(data.get("language", "en")),
            beam_size=beam_size,
            sample_rate=sample_rate,
            frame_seconds=frame_seconds,
            vad=str(data.get("vad", "energy")).strip().lower(),
            vad_threshold=_fraction("voice.vad_threshold", data.get("vad_threshold", 0.02)),
            start_seconds=parse_seconds(data.get("start", 0.15)),
            silence_seconds=parse_seconds(data.get("silence", 0.6)),
            min_utterance_seconds=min_utterance,
            max_utterance_seconds=max_utterance,
            pre_roll_seconds=parse_seconds(data.get("pre_roll", 0.2)),
        )


@dataclass(frozen=True, slots=True)
class Identity:
    """Durable self-identity that outlives memory and DB resets (DESIGN 5, 27).

    ``name`` is the agent's own name. Unlike a fact it happens to remember, it's part of the fixed
    identity injected into every prompt, so it survives runtime sessions, memory decay, and a data
    reset. Empty by default (the agent has no name unless one is configured).
    """

    name: str = ""

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Identity:
        return Identity(name=str(data.get("name", "")).strip())


@dataclass(frozen=True, slots=True)
class Tts:
    """Optional speech output for the chat client (DESIGN 2.2 secondary goal; 6.1; 28.1).

    Speech output is a *client-side rendering* concern, not cognition. DESIGN 6.1 anticipates it
    exactly — "Future embodiment replaces or augments delivery adapters (for example, TTS/speaker
    output) without changing cognition semantics" — and 28.1 lists TTS among the reasons the
    reference stack is Python. The ``aca chat`` client speaks a *spoken normalization* of each
    delivered agent message (markup stripped so Kokoro doesn't read punctuation aloud); the text is
    still printed verbatim, and nothing here touches the reducer or durable state (invariant 1). It
    is a second rendering of an utterance the reducer already authorized and delivered, so it needs
    no injected ``Clock``/``Rng``.

    ``provider`` is ``none`` (the silent default: no audio, no extra dependencies) or ``kokoro``
    (offline Kokoro-82M neural TTS, ``pip install 'aca[tts]'``). Kokoro runs locally; after a
    one-time model download from Hugging Face on first use, nothing leaves the host. ``voice``
    selects a Kokoro voice (default ``am_onyx``, an American male voice); ``lang_code`` is Kokoro's
    pipeline language, auto-derived from the voice's first letter when left blank. ``speed`` scales
    the speaking rate; ``device`` optionally names a non-default audio output device. (There is no
    sample-rate knob: Kokoro's decoder output is fixed at 24 kHz, so the player clock is that
    constant, not configuration.)
    """

    provider: str = "none"
    voice: str = "am_onyx"
    lang_code: str = ""
    speed: float = 1.0
    device: str | None = None

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Tts:
        speed = float(data.get("speed", 1.0))
        if speed <= 0.0:
            raise ConfigError("tts.speed must be > 0")
        return Tts(
            provider=str(data.get("provider", "none")).strip().lower(),
            voice=str(data.get("voice", "am_onyx")).strip(),
            lang_code=str(data.get("lang_code", "")).strip(),
            speed=speed,
            device=(str(data["device"]) if data.get("device") else None),
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level immutable configuration."""

    local_timezone: str = "UTC"
    rng_seed: int | None = None
    identity: Identity = field(default_factory=Identity)
    cognition: Cognition = field(default_factory=Cognition)
    temperament: Temperament = field(default_factory=Temperament)
    timing: Timing = field(default_factory=Timing)
    memory: Memory = field(default_factory=Memory)
    budgets: Budgets = field(default_factory=Budgets)
    conversation: Conversation = field(default_factory=Conversation)
    llm: LLM = field(default_factory=LLM)
    embedding: Embedding = field(default_factory=Embedding)
    voice: Voice = field(default_factory=Voice)
    tts: Tts = field(default_factory=Tts)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> Config:
        seed = data.get("rng_seed")
        return Config(
            local_timezone=str(data.get("local_timezone", "UTC")),
            rng_seed=None if seed is None else int(seed),
            identity=Identity.from_mapping(data.get("identity", {})),
            cognition=Cognition.from_mapping(data.get("cognition", {})),
            temperament=Temperament.from_mapping(data.get("temperament", {})),
            timing=Timing.from_mapping(data.get("timing", {})),
            memory=Memory.from_mapping(data.get("memory", {})),
            budgets=Budgets.from_mapping(data.get("budgets", {})),
            conversation=Conversation.from_mapping(data.get("conversation", {})),
            llm=LLM.from_mapping(data.get("llm", {})),
            embedding=Embedding.from_mapping(data.get("embedding", {})),
            voice=Voice.from_mapping(data.get("voice", {})),
            tts=Tts.from_mapping(data.get("tts", {})),
        )

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a copy with top-level fields replaced (useful for tests)."""
        return replace(self, **overrides)
