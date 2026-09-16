"""Adversarial: crash recovery preserves identity, replays work, and never eats events.

Covers invariants 24, 25, 29 and DESIGN 29.1: a durable ingress survives restart, RUNNING leases
are reclaimed, stale proactive items expire, and the logical agent_id is never re-minted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aca import ids
from aca.clock import ManualClock
from aca.config import Config
from aca.domain.enums import OutboundKind, OutboundStatus, WorkKind, WorkStatus
from aca.domain.events import HumanMessage
from aca.domain.runtime import OutboundMessage, WorkItem
from aca.persistence.stores import Stores
from aca.service.recovery import recover

UTC = timezone.utc


def _config():
    return Config.from_mapping({"rng_seed": 1})


def test_identity_survives_restart(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, tzinfo=UTC))
    s1 = Stores.open(tmp_path / "agent.db")
    plan1 = recover(s1, clock, _config())
    agent_id = s1.identity.load_identity().agent_id
    assert plan1.is_new_agent is True
    s1.close()

    # "Restart": reopen the same data dir.
    s2 = Stores.open(tmp_path / "agent.db")
    plan2 = recover(s2, clock, _config())
    assert plan2.is_new_agent is False
    assert s2.identity.load_identity().agent_id == agent_id  # never re-minted (invariant 29)
    assert plan2.scheduler_generation > plan1.scheduler_generation  # fresh generation on resume
    s2.close()


def test_accepted_but_unreduced_event_is_recovered(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, tzinfo=UTC))
    s1 = Stores.open(tmp_path / "agent.db")
    recover(s1, clock, _config())
    # Simulate: ACKed (accepted) but the process crashed before reduction.
    ev = HumanMessage(event_id="evt_crash", timestamp=clock.now_utc(), text="Explain the crash.")
    with s1.db.transaction():
        s1.events.accept(ev, clock.now_utc())
    s1.close()

    s2 = Stores.open(tmp_path / "agent.db")
    plan = recover(s2, clock, _config())
    assert [e.event_id for e in plan.replay_events] == ["evt_crash"]  # not silently eaten (inv 14)
    s2.close()


def test_expired_running_lease_is_reclaimed(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    s = Stores.open(tmp_path / "agent.db")
    recover(s, clock, _config())
    with s.db.transaction():
        s.work.insert_work(WorkItem(
            work_id="work_stuck", kind=WorkKind.LLM_COGNITION, cycle_id="c", basis_revision=0,
            source_event_id="e", status=WorkStatus.RUNNING, created_at=clock.now_utc(),
            lease_until=clock.now_utc() - timedelta(minutes=5),
        ))
    plan = recover(s, clock, _config())
    assert "work_stuck" in plan.dispatch_work_ids  # reclaimed + requeued (DESIGN 23.5)
    assert s.work.get_work("work_stuck").status is WorkStatus.PENDING
    s.close()


def test_stale_proactive_item_expires_on_recovery(tmp_path):
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    s = Stores.open(tmp_path / "agent.db")
    recover(s, clock, _config())
    with s.db.transaction():
        s.outbox.insert_message(OutboundMessage(
            message_id="msg_stale", delivery_key=ids.new_delivery_key(), action_id="a",
            kind=OutboundKind.PROACTIVE, channel="cli", payload="old thought",
            status=OutboundStatus.PENDING_DELIVERY, created_at=clock.now_utc() - timedelta(hours=10),
            expires_at=clock.now_utc() - timedelta(hours=1),
        ))
    recover(s, clock, _config())
    # A stale proactive backlog is expired, not dumped on reconnect (DESIGN 23.7).
    assert s.outbox.get_message("msg_stale").status is OutboundStatus.EXPIRED
    s.close()
