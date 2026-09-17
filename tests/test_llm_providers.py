"""Unit tests for the provider-agnostic LLM worker, adapters, and factory (no network, no keys)."""

from __future__ import annotations

import json

import httpx
import pytest

from aca.cognition.snapshot import Snapshot
from aca.config import LLM
from aca.domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE
from aca.errors import ConfigError
from aca.workers.fake_llm import FakeLLMWorker
from aca.workers.llm import build_llm_worker
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
    assert result.text == '{"action":"speak","message":"ok"}'
    assert (result.tokens_in, result.tokens_out) == (5, 2)


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
