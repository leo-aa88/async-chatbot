"""Adversarial: agent downtime is not user absence; missed wakes are not replayed.

Covers invariants 30-32: suspension interrupts cognition (not identity), downtime never counts
as observed user silence, and resume does not replay missed stochastic cognition.
"""

from __future__ import annotations

from dataclasses import replace

from conftest import Harness

from aca import ids
from aca.domain.events import AgentResumed, RuntimeInterruptionDetected


def test_resume_does_not_add_downtime_to_observed_silence(harness: Harness):
    conversation = harness.stores.state.load_conversation()
    with harness.stores.db.transaction():
        harness.stores.state.save_conversation(
            replace(conversation, active_observed_silence_seconds=120.0)
        )
    # Simulate four days of host-off downtime, then resume.
    harness.clock.advance_wall_only(4 * 24 * 3600)
    harness.reduce(AgentResumed(ids.new_id(ids.EVENT), harness.clock.now_utc(), "service",
                                "sess_1", downtime_seconds=4 * 24 * 3600))
    after = harness.stores.state.load_conversation().active_observed_silence_seconds
    assert after == 120.0  # downtime added nothing to observed silence (invariant 31)


def test_resume_replays_no_stochastic_wakes(harness: Harness):
    result = harness.reduce(
        AgentResumed(ids.new_id(ids.EVENT), harness.clock.now_utc(), "service", "sess_1", 3600)
    )
    # Resume commits temporal state but schedules no work and enqueues no missed wakes (inv 32).
    assert result.dispatch_work_ids == []
    assert result.reschedule is False


def test_interruption_records_downtime_without_new_identity(harness: Harness):
    original_agent = harness.stores.identity.load_identity().agent_id
    harness.reduce(
        RuntimeInterruptionDetected(ids.new_id(ids.EVENT), harness.clock.now_utc(), "service",
                                    "sess_1", downtime_seconds=99.0)
    )
    identity = harness.stores.identity.load_identity()
    assert identity.agent_id == original_agent  # never re-minted (invariant 29)
    assert identity.last_runtime_exit_kind is not None
