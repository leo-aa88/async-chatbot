"""Unit tests for the persistence layer's durability/idempotency contracts."""

from __future__ import annotations

from datetime import UTC, datetime

from aca.domain.events import HumanMessage
from aca.domain.state import ConversationState, SelfModel
from aca.persistence.stores import Stores

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _stores(tmp_path) -> Stores:
    return Stores.open(tmp_path / "agent.db")


def test_event_accept_is_idempotent(tmp_path):
    s = _stores(tmp_path)
    ev = HumanMessage(event_id="evt_1", timestamp=T0, text="hi")
    assert s.events.accept(ev, T0) is True
    assert s.events.accept(ev, T0) is False  # duplicate ingress
    assert len(s.events.unreduced()) == 1
    s.close()


def test_unreduced_recovered_in_order(tmp_path):
    s = _stores(tmp_path)
    for i in range(3):
        s.events.accept(HumanMessage(event_id=f"evt_{i}", timestamp=T0, text=str(i)), T0)
    s.events.mark_reduced("evt_1", T0)
    remaining = [e.event_id for e in s.events.unreduced()]
    assert remaining == ["evt_0", "evt_2"]
    s.close()


def test_state_revision_monotonic(tmp_path):
    s = _stores(tmp_path)
    s.state.initialize(SelfModel(0.5, 0.5, 0.5), ConversationState())
    assert s.state.state_revision() == 0
    assert s.state.advance_revision() == 1
    assert s.state.advance_revision() == 2
    s.close()


def test_budget_usage_is_window_keyed(tmp_path):
    s = _stores(tmp_path)
    s.state.increment_budget("llm_day", "2026-06-01")
    s.state.increment_budget("llm_day", "2026-06-01")
    assert s.state.budget_count("llm_day", "2026-06-01") == 2
    # A different window has no banked credit.
    assert s.state.budget_count("llm_day", "2026-06-02") == 0
    s.close()


def test_self_model_roundtrip(tmp_path):
    s = _stores(tmp_path)
    s.state.initialize(SelfModel(0.4, 0.6, 0.5), ConversationState())
    model = s.state.load_self_model().with_delivered_proactive(T0)
    s.state.save_self_model(model)
    reloaded = s.state.load_self_model()
    assert reloaded.recent_proactive_messages == 1
    assert reloaded.last_delivered_proactive_at == T0
    s.close()
