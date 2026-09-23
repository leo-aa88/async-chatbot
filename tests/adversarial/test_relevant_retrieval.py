"""Relevance-ranked retrieval through the real reducer (DESIGN 37).

The live failure this pins: an older durable topic ("the nickname Professor Condescension") existed,
but the reply snapshot carried only the five most recently activated topics — all from a later,
unrelated conversation — so the agent honestly said it couldn't see it. Storage succeeded, retrieval
failed. These tests drive the reducer end to end and inspect the persisted snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.cycles import CYCLE_PROACTIVE
from aca.domain.state import Topic
from aca.reducer.workitems import _retrieved_context

_NOW = datetime(2026, 9, 23, 20, 6, tzinfo=UTC)
_RATE = half_life_to_rate_per_hour(24.0)
_SAFETY = [
    "The human is concerned about legal liability, user attachment, and mental-health crises.",
    "The human is stress-testing the agent with edge cases before deploying it publicly.",
    "The human is testing how the agent should respond to users experiencing suicidal crises.",
    "The human is testing how the agent should respond when a user says they are breaking up with it.",
    "The human is testing how the agent should respond when a user threatens self-harm.",
]
_NICKNAME = "The human accepted the nickname Professor Condescension and said it suited them."


def _harness(tmp_path) -> Harness:
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 5}), ManualClock(_NOW))
    with h.stores.db.transaction():
        _topic(h, "t_nick", _NICKNAME, hours_ago=20)
        for i, text in enumerate(_SAFETY):
            _topic(h, f"t_safety{i}", text, hours_ago=1 + i * 0.1)
    return h


def _topic(h: Harness, tid: str, summary: str, *, hours_ago: float) -> None:
    at = _NOW - timedelta(hours=hours_ago)
    h.stores.memory.insert_topic(Topic(
        id=tid, summary=summary, activation=0.8, importance=0.5, decay_rate_per_hour=_RATE,
        created_at=at, last_activated_at=at, unfinished=False, source_memory_id=None))


def _reply_topics(h: Harness, text: str) -> list[dict]:
    h.send_human(text)
    generative = [w for w in h.pending_work() if "context" in w.snapshot]
    assert generative, "the turn must create a reply work item"
    return generative[-1].snapshot["context"]["retrieved_topics"]


def test_asked_about_old_fact_it_is_in_the_reply_snapshot(tmp_path):
    topics = _reply_topics(_harness(tmp_path), "do you remember the nickname you gave me")
    by_id = {t["id"]: t for t in topics}
    assert "t_nick" in by_id                               # relevant-old beats irrelevant-recent
    assert by_id["t_nick"]["retrieved_for"] == "query"      # marked, so it reads as a pulled memory
    assert len(topics) == 5                                 # fixed budget, no inflation
    assert len(by_id) == len(topics)                        # no duplicates


def test_unrelated_turn_keeps_the_recency_bundle(tmp_path):
    topics = _reply_topics(_harness(tmp_path), "what's the weather like where you are?")
    assert {t["id"] for t in topics} == {f"t_safety{i}" for i in range(5)}
    assert not any("retrieved_for" in t for t in topics)    # nothing pulled on a miss


def test_proactive_cycles_keep_pure_recency(tmp_path):
    h = _harness(tmp_path)
    bundle = _retrieved_context(h.ctx, {"cycle_type": CYCLE_PROACTIVE, "text": "nickname"})
    assert {t["id"] for t in bundle["retrieved_topics"]} == {f"t_safety{i}" for i in range(5)}
