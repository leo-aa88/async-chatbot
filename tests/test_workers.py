"""Unit tests for the deterministic fake workers.

Locks in the cycle-appropriate voice fix: a reactive reply to a just-sent message must not use
the proactive "I've been thinking about what you said" opener (which only makes sense when
resurfacing an older thought).
"""

from __future__ import annotations

import asyncio

from aca.cognition.snapshot import Snapshot
from aca.domain.cycles import CYCLE_MANDATORY, CYCLE_PROACTIVE, CYCLE_REACTIVE_OPTIONAL
from aca.workers.fake_embedding import FakeEmbeddingWorker, embed_text
from aca.workers.fake_llm import FakeLLMWorker


def _snapshot(source: dict) -> Snapshot:
    return Snapshot(cycle_id="c", work_id="w", basis_revision=0, template_version="v0.6",
                    context={"source": source})


def _run(source: dict) -> dict:
    return asyncio.run(FakeLLMWorker().run(_snapshot(source))).result


def test_reactive_reply_is_present_tense_not_resurfacing():
    result = _run({
        "cycle_type": CYCLE_REACTIVE_OPTIONAL, "response_required": False,
        "text": "another test", "candidate": {"kind": "PROVISIONAL_MEMORY", "id": "m", "text": "another test"},
    })
    assert result["action"] == "speak"
    assert "thinking about what you said" not in result["message"].lower()
    assert "another test" in result["message"]


def test_proactive_reply_uses_resurfacing_opener():
    result = _run({
        "cycle_type": CYCLE_PROACTIVE, "response_required": False,
        "candidate": {"kind": "PROVISIONAL_MEMORY", "id": "m", "text": "machine telos", "salience": 0.9},
    })
    assert result["action"] == "speak"
    assert "thinking about what you said" in result["message"].lower()


def test_mandatory_reply_answers_the_task():
    result = _run({"cycle_type": CYCLE_MANDATORY, "response_required": True,
                   "text": "Explain this stack trace."})
    assert result["action"] == "speak"
    assert "thinking about what you said" not in result["message"].lower()


def test_low_salience_proactive_stays_silent_but_enriches():
    result = _run({
        "cycle_type": CYCLE_PROACTIVE, "response_required": False,
        "candidate": {"kind": "PROVISIONAL_MEMORY", "id": "m", "text": "small thing", "salience": 0.1},
    })
    assert result["action"] == "silence"
    assert result.get("useful_enrichment") is True


def test_fake_embedding_is_deterministic_and_normalized():
    v1 = embed_text("hello world")
    v2 = asyncio.run(FakeEmbeddingWorker().embed("hello world")).vector
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6
