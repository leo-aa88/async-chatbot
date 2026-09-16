"""Adversarial: stale wake events are harmless (invariant 34).

An overdue timer callback that fires after a resume, or one carrying a superseded scheduler
generation / runtime session, must be dropped without any cognition and without advancing state.
"""

from __future__ import annotations

from dataclasses import replace

from conftest import Harness

from aca.domain.enums import LifecycleState


def test_wake_with_wrong_generation_is_dropped(harness: Harness):
    before = harness.stores.state.state_revision()
    result = harness.wake(generation=999)  # current generation is 1
    assert result.note == "stale_wake"
    assert not result.reschedule
    assert harness.stores.state.state_revision() == before  # no commit


def test_wake_with_wrong_session_is_dropped(harness: Harness):
    before = harness.stores.state.state_revision()
    result = harness.wake(session_id="sess_OLD")
    assert result.note == "stale_wake"
    assert harness.stores.state.state_revision() == before


def test_wake_while_not_running_is_dropped(harness: Harness):
    identity = harness.stores.identity.load_identity()
    with harness.stores.db.transaction():
        harness.stores.identity.update_identity(
            replace(identity, lifecycle_state=LifecycleState.SUSPENDED)
        )
    result = harness.wake()  # correct session/generation, but not RUNNING
    assert result.note == "stale_wake"


def test_valid_wake_is_processed(harness: Harness):
    result = harness.wake()  # matches current session + generation, RUNNING
    assert result.note != "stale_wake"
    assert result.reschedule  # a valid wake always resamples the next wake
