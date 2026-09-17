"""Unit tests for the provider-agnostic LLM worker, adapters, and factory (no network, no keys)."""

from __future__ import annotations

import json

import httpx
import pytest

from aca.cognition.snapshot import Snapshot
from aca.config import LLM
from aca.domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE
from aca.errors import ConfigError, WorkerError
from aca.workers.fake_llm import FakeLLMWorker
from aca.workers.llm import build_llm_worker, key_env_for
from aca.workers.llm.adapters import AnthropicAdapter, ChatResult, OpenAICompatibleAdapter
from aca.workers.llm.prompt import build_prompt
from aca.workers.llm.worker import ProviderLLMWorker, _coerce


def _snapshot(cycle_type: str, **source) -> Snapshot:
    return Snapshot(
        cycle_id="cog_1", work_id="w1", basis_revision=3, template_version="v0.6",
        context={"source": {"cycle_type": cycle_type, **source}, "recent_conversation": []},
    )


class StubAdapter:
    def __init__(self, text: str):
        self._text = text
        self.seen: tuple[str, str] | None = None

    async def complete(self, system: str, user: str) -> ChatResult:
        self.seen = (system, user)
        return ChatResult(self._text, tokens_in=11, tokens_out=7)


# --- prompt ------------------------------------------------------------------------------
def test_prompt_is_deterministic_and_carries_contract():
    snap = _snapshot(CYCLE_MANDATORY, text="Explain this", response_required=True)
    system_a, user_a = build_prompt(snap)
    system_b, user_b = build_prompt(snap)
    assert (system_a, user_a) == (system_b, user_b)  # deterministic
    assert '"action"' in system_a and "mandatory" in system_a
    assert "cog_1" in user_a and "Explain this" in user_a


def test_proactive_and_reactive_are_distinct_speech_acts():
    # A self-initiated thought and a reply must be instructed differently (same model, same
    # memories, different speech-act context). Guard the proactive guidance that stops the model
    # from thanking-for-sharing / offering generic help on an unprompted turn.
    system, _ = build_prompt(_snapshot(CYCLE_PROACTIVE))
    assert "initiating this turn yourself" in system
    assert "Do NOT thank" in system
    assert "express the specific thought that made that topic worth" in system
    assert "reactive" in system and "mandatory" in system  # all three acts still described


def test_prompt_carries_plain_voice_guidance():
    # Guard the anti-"4o tic" voice guidance so the agent doesn't out itself as a chat assistant.
    system, _ = build_prompt(_snapshot(CYCLE_PROACTIVE))
    assert "Voice" in system
    assert "feel free to" in system  # the exact assistant tic it must avoid
    assert "filler openers" in system
    assert "keep the conversation going" in system  # no filler continuation-questions
    assert "not an always-available assistant" in system  # runtime owns timing/delivery, not it


def test_prompt_injects_configured_name():
    # A configured identity.name is surfaced in agent_state and prepended so the agent knows its
    # name every cycle (durable across restarts/decay). Absent -> no name line.
    named = Snapshot(cycle_id="c", work_id="w", basis_revision=1, template_version="v0.6",
                     context={"source": {"cycle_type": "proactive"}, "agent_state": {"name": "Wolfy"}})
    system, _ = build_prompt(named)
    assert system.startswith("Your name is Wolfy.")
    anon, _ = build_prompt(_snapshot(CYCLE_PROACTIVE))
    assert "Your name is" not in anon  # no name configured -> no name line


def test_prompt_instructs_enrichment():
    # With a real model, topics only form if the prompt actually asks for enrichment proposals
    # (the fake worker always emitted them, masking this). Guard the instruction.
    system, _ = build_prompt(_snapshot(CYCLE_PROACTIVE))
    assert "Enrichment —" in system
    assert "reactive or proactive alike" in system  # not scoped to proactive cycles
    assert "ENRICH_PROVISIONAL_MEMORY proposal" in system
    assert "output_eligible is present and false" in system  # degrades gracefully when absent


def test_prompt_carries_disposition():
    # Guard the persona: non-sycophantic, skeptical-but-open, self-respecting under abuse.
    system, _ = build_prompt(_snapshot(CYCLE_PROACTIVE))
    assert "Disposition" in system
    assert "sycophantic" in system
    assert "Skeptical but open-minded" in system
    assert "Self-respecting" in system


# --- response coercion -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_parses_json_action():
    adapter = StubAdapter('{"action": "speak", "message": "hi there"}')
    out = await ProviderLLMWorker(adapter).run(_snapshot(CYCLE_MANDATORY))
    assert out.result == {"action": "speak", "message": "hi there"}
    assert (out.tokens_in, out.tokens_out) == (11, 7)


@pytest.mark.asyncio
async def test_worker_strips_code_fences():
    adapter = StubAdapter('```json\n{"action":"silence"}\n```')
    out = await ProviderLLMWorker(adapter).run(_snapshot(CYCLE_PROACTIVE))
    assert out.result == {"action": "silence"}


def test_prose_fallback_speaks_on_mandatory_but_silences_on_proactive():
    assert _coerce("just some prose, no json", speech_fallback=True) == {
        "action": "speak", "message": "just some prose, no json",
    }
    assert _coerce("just some prose, no json", speech_fallback=False) == {"action": "silence"}


def test_malformed_json_is_never_spoken_as_prose():
    # The review's repro: a truncated/partial JSON object must NOT be spoken verbatim (scaffolding
    # leak). It's treated as a parse failure the reducer handles, not prose.
    partial = '{"action": "speak", "message": "Sure, here is a detailed explanation of the'
    assert _coerce(partial, speech_fallback=True) == {}
    assert _coerce("```json\n{\"action\":\"speak\"", speech_fallback=True) == {}


@pytest.mark.asyncio
async def test_worker_raises_on_truncated_response():
    class Truncating:
        async def complete(self, system, user):
            return ChatResult('{"action":"speak","message":"cut off mid', 10, 5, truncated=True)

    with pytest.raises(WorkerError):
        await ProviderLLMWorker(Truncating()).run(_snapshot(CYCLE_MANDATORY))


@pytest.mark.asyncio
async def test_openai_adapter_flags_truncation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 256},
        })

    _patch_httpx(monkeypatch, handler)
    result = await OpenAICompatibleAdapter("https://x/v1", "k", "m", 256, 30.0).complete("s", "u")
    assert result.truncated is True


@pytest.mark.asyncio
async def test_anthropic_adapter_flags_truncation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 9, "output_tokens": 256},
        })

    _patch_httpx(monkeypatch, handler)
    result = await AnthropicAdapter("https://api.anthropic.com", "k", "m", 256, 30.0).complete("s", "u")
    assert result.truncated is True


# --- adapters (mocked transport, no network) ---------------------------------------------
@pytest.mark.asyncio
async def test_openai_adapter_shapes_request_and_parses_usage(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":"speak","message":"ok"}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    _patch_httpx(monkeypatch, handler)
    adapter = OpenAICompatibleAdapter("https://api.example.com/v1", "sk-test", "some-model", 256, 30.0)
    result = await adapter.complete("sys", "usr")
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "some-model"
    assert captured["body"]["max_tokens"] == 256  # legacy models keep max_tokens
    assert "max_completion_tokens" not in captured["body"]
    assert result.text == '{"action":"speak","message":"ok"}'
    assert (result.tokens_in, result.tokens_out) == (5, 2)


@pytest.mark.asyncio
async def test_openai_adapter_uses_max_completion_tokens_for_gpt5(monkeypatch):
    # GPT-5 family / o-series reject max_tokens on /chat/completions (the 400 seen in the field).
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"action":"silence"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    _patch_httpx(monkeypatch, handler)
    await OpenAICompatibleAdapter("https://api.openai.com/v1", "k", "gpt-5.6-luna", 512, 30.0).complete("s", "u")
    assert captured["body"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["body"]


@pytest.mark.asyncio
async def test_openai_adapter_surfaces_error_body(monkeypatch):
    # A 4xx must raise with the provider's actual message, not a bare status (observability).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'"}})

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(WorkerError) as exc:
        await OpenAICompatibleAdapter("https://api.openai.com/v1", "k", "gpt-5.6-luna", 512, 30.0).complete("s", "u")
    assert "400" in str(exc.value) and "Unsupported parameter" in str(exc.value)


def test_token_limit_param_selection():
    from aca.workers.llm.adapters import _token_limit_param
    assert _token_limit_param("gpt-5.6-luna") == "max_completion_tokens"
    assert _token_limit_param("gpt-5-mini") == "max_completion_tokens"
    assert _token_limit_param("o3-mini") == "max_completion_tokens"
    assert _token_limit_param("gpt-4o") == "max_tokens"
    assert _token_limit_param("gpt-4o-mini") == "max_tokens"
    assert _token_limit_param("grok-2-latest") == "max_tokens"


@pytest.mark.asyncio
async def test_anthropic_adapter_shapes_request_and_parses_usage(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.headers.get("x-api-key")
        captured["ver"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": '{"action":"silence"}'}],
            "usage": {"input_tokens": 9, "output_tokens": 1},
        })

    _patch_httpx(monkeypatch, handler)
    adapter = AnthropicAdapter("https://api.anthropic.com", "sk-ant", "claude-x", 256, 30.0)
    result = await adapter.complete("sys", "usr")
    assert captured["url"].endswith("/v1/messages")
    assert captured["key"] == "sk-ant" and captured["ver"]
    assert result.text == '{"action":"silence"}' and result.tokens_in == 9


# --- factory -----------------------------------------------------------------------------
def test_key_env_for_maps_provider_to_single_credential():
    assert key_env_for(LLM(provider="fake")) is None
    assert key_env_for(LLM(provider="openai", model="m")) == "OPENAI_API_KEY"
    assert key_env_for(LLM(provider="claude", model="m")) == "ANTHROPIC_API_KEY"
    assert key_env_for(LLM(provider="xai", model="m")) == "XAI_API_KEY"
    assert key_env_for(LLM(provider="google", model="m")) == "GEMINI_API_KEY"
    assert key_env_for(LLM(provider="openai", model="m", api_key_env="MY_KEY")) == "MY_KEY"


def test_factory_defaults_to_fake():
    assert isinstance(build_llm_worker(LLM()), FakeLLMWorker)
    assert isinstance(build_llm_worker(LLM(provider="mock")), FakeLLMWorker)


def test_factory_builds_real_provider_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    worker = build_llm_worker(LLM(provider="openai", model="gpt-x"))
    assert isinstance(worker, ProviderLLMWorker)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ConfigError):
        build_llm_worker(LLM(provider="llama", model="x"))


def test_factory_requires_model_and_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        build_llm_worker(LLM(provider="grok"))  # no model
    with pytest.raises(ConfigError):
        build_llm_worker(LLM(provider="grok", model="grok-x"))  # no key in env


def _patch_httpx(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
