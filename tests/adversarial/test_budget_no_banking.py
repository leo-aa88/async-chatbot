"""Adversarial: proactive budgets gate speech and never bank across windows (invariant 27)."""

from __future__ import annotations

from conftest import Harness

from aca import ids
from aca.cognition import budgets as budget
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.state import ProvisionalMemory


def _config():
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 1, "proactive_llm_calls_per_hour": 5,
                    "proactive_llm_calls_per_day": 50},
    })


def _seed(h: Harness):
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=ids.new_id(ids.PROVISIONAL_MEMORY), event_id="seed", text="fluid mechanics return",
            activation=0.95, salience=0.95, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))


def test_hourly_message_budget_blocks_second_proactive(tmp_path, clock):
    h = Harness(tmp_path, _config(), clock)
    _seed(h)
    # First proactive message consumes the single hourly slot.
    h.wake(); h.run_all_pending()
    hour = budget.hour_window_id(clock.now_utc())
    assert h.stores.state.budget_count("msg_hour", hour) == 1

    # Second attempt in the same hour is blocked at pre-outbox (no new proactive message).
    h.wake(); h.run_all_pending()
    assert h.stores.state.budget_count("msg_hour", hour) == 1  # still one; budget held the line
    h.close()


def test_next_window_starts_fresh_without_banking(tmp_path, clock):
    h = Harness(tmp_path, _config(), clock)
    _seed(h)
    h.wake(); h.run_all_pending()
    first_hour = budget.hour_window_id(clock.now_utc())

    # Advance one hour: a brand-new window. It grants exactly its capacity, never banked extra.
    clock.advance(3600)
    next_hour = budget.hour_window_id(clock.now_utc())
    assert next_hour != first_hour
    assert h.stores.state.budget_count("msg_hour", next_hour) == 0  # no accrued/banked credit
    h.wake(); h.run_all_pending()
    assert h.stores.state.budget_count("msg_hour", next_hour) == 1
    h.close()
