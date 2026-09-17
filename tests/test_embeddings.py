"""Real embedding adapter + factory (no network, no keys)."""

from __future__ import annotations

import json

import httpx
import pytest

from aca.config import Embedding
from aca.errors import ConfigError, WorkerError
from aca.workers.embedding_factory import build_embedding_worker, embedding_key_env_for
from aca.workers.embeddings import OpenAIEmbeddingAdapter
from aca.workers.fake_embedding import FakeEmbeddingWorker


def _patch_httpx(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    real_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


@pytest.mark.asyncio
async def test_adapter_shapes_request_and_parses_vector(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "data": [{"embedding": [0.1, 0.2, 0.3]}], "model": "text-embedding-3-small",
        })

    _patch_httpx(monkeypatch, handler)
    out = await OpenAIEmbeddingAdapter("https://api.openai.com/v1", "sk", "text-embedding-3-small", 30.0).embed("hi")
    assert captured["url"].endswith("/embeddings")
    assert captured["auth"] == "Bearer sk"
    assert captured["body"] == {"model": "text-embedding-3-small", "input": "hi"}
    assert out.vector == [0.1, 0.2, 0.3]
    assert out.model_version == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_adapter_surfaces_error_body(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(WorkerError) as exc:
        await OpenAIEmbeddingAdapter("https://api.openai.com/v1", "bad", "m", 30.0).embed("hi")
    assert "401" in str(exc.value) and "invalid api key" in str(exc.value)


def test_factory_defaults_to_fake():
    assert isinstance(build_embedding_worker(Embedding()), FakeEmbeddingWorker)
    assert embedding_key_env_for(Embedding()) is None


def test_factory_builds_openai_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = Embedding(provider="openai", model="text-embedding-3-small")
    assert embedding_key_env_for(cfg) == "OPENAI_API_KEY"
    worker = build_embedding_worker(cfg)
    assert worker.model_version == "text-embedding-3-small"


def test_factory_requires_model_and_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        build_embedding_worker(Embedding(provider="openai", model=""))
    with pytest.raises(ConfigError):
        build_embedding_worker(Embedding(provider="openai", model="m"))  # key unset


def test_factory_rejects_unknown_provider():
    with pytest.raises(ConfigError):
        build_embedding_worker(Embedding(provider="nope", model="m"))
