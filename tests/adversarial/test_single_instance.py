"""Adversarial: exactly one service may own a data directory (invariant 3, DESIGN 29.1)."""

from __future__ import annotations

import pytest

from aca.errors import ServiceAlreadyRunningError
from aca.service.lock import SingleInstanceLock


def test_second_instance_is_rejected(tmp_path):
    first = SingleInstanceLock(tmp_path)
    first.acquire()
    second = SingleInstanceLock(tmp_path)
    with pytest.raises(ServiceAlreadyRunningError):
        second.acquire()
    first.release()


def test_lock_is_reusable_after_release(tmp_path):
    lock = SingleInstanceLock(tmp_path)
    lock.acquire()
    lock.release()
    again = SingleInstanceLock(tmp_path)
    again.acquire()  # should not raise now that the first was released
    again.release()
