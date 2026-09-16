"""Adversarial: quiet hours are a hard gate on *unsolicited* speech only (invariant 26).

During quiet hours the agent must not emit proactive messages, but internal cognition (activation
/ selection / enrichment) still runs and explicit reactive tasks are still answered.
"""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import Harness

from aca import ids
from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import ProvisionalMemory


def _quiet_config():
    return Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -5.0,
                      "semantic_worthiness_floor": 0.3},
        "timing": {"quiet_hours": {"enabled": True, "start_local": "01:00", "end_local": "08:00"}},
    })


def _seed_raw_memory(h: Harness):
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=ids.new_id(ids.PROVISIONAL_MEMORY), event_id="seed", text="machine telos question",
            activation=0.9, salience=0.9, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))


def test_no_proactive_message_during_quiet_hours(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 2, 0, tzinfo=UTC))  # 02:00 local, inside quiet
    h = Harness(tmp_path, _quiet_config(), clock)
    _seed_raw_memory(h)
    h.wake()
    h.run_all_pending()
    proactive = [m for m in h.stores.outbox.deliverable() if m.kind is OutboundKind.PROACTIVE]
    assert proactive == []  # unsolicited speech blocked
    # But internal cognition was not suppressed: enrichment created a topic (DESIGN 11.6).
    assert h.stores.memory.all_topics(), "enrichment/internal cognition must still run in quiet hours"
    h.close()


def test_reactive_task_still_answered_during_quiet_hours(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 2, 0, tzinfo=UTC))
    h = Harness(tmp_path, _quiet_config(), clock)
    h.send_human("Explain this stack trace.")
    h.run_all_pending()
    msgs = h.stores.outbox.deliverable()
    assert len(msgs) == 1 and msgs[0].kind is OutboundKind.MANDATORY  # tasks unaffected by quiet
    h.close()
