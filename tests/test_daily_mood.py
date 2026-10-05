"""The agent's mood of the day: a stable daily draw that shapes how it words things, never whether.

ACA decides and words a reply in one call, so the note itself says the mood changes wording only; the
live silence rate per mood is measured with ``make backbone-eval`` (``--mood``).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta

from conftest import make_harness

from aca.clock import ManualClock
from aca.cognition.snapshot import Snapshot
from aca.config import Config
from aca.domain.mood import HIGH, LOW, NORMAL, daily_mood, mood_note
from aca.workers.llm.prompt import build_prompt

DAYS = [date(2026, 1, 1) + timedelta(days=d) for d in range(400)]


def test_mostly_normal_an_off_day_or_a_good_day_now_and_then():
    counts = Counter(daily_mood(f"agt_{a}", day) for a in range(25) for day in DAYS)
    total = sum(counts.values())
    assert abs(counts[LOW] / total - 0.20) < 0.02
    assert abs(counts[NORMAL] / total - 0.60) < 0.02
    assert abs(counts[HIGH] / total - 0.20) < 0.02


def test_each_day_is_its_own_draw():
    # Independent days: the same mood as yesterday about 0.2² + 0.6² + 0.2² = 44% of the time.
    same = sum(daily_mood(f"agt_{a}", x) == daily_mood(f"agt_{a}", y)
               for a in range(25) for x, y in zip(DAYS, DAYS[1:], strict=False))
    assert abs(same / (25 * 399) - 0.44) < 0.03


def test_a_mood_holds_all_day():
    assert daily_mood("agt_a", date(2026, 10, 5)) == daily_mood("agt_a", date(2026, 10, 5))


def test_only_low_and_high_days_get_a_note_and_it_is_about_wording():
    assert mood_note(NORMAL) is None and mood_note(None) is None
    for mood in (LOW, HIGH):
        assert "never whether you speak" in mood_note(mood)


def _snapshot(mood):
    return Snapshot("c", "w", 1, "t", {"agent_state": {"mood": mood} if mood else {}})


def test_the_note_reaches_the_prompt_on_low_and_high_days_only():
    assert "energy is low today" in build_prompt(_snapshot(LOW))[1]
    assert "more energy than usual" in build_prompt(_snapshot(HIGH))[1]
    assert "mood_note" not in build_prompt(_snapshot(NORMAL))[1]
    assert "mood_note" not in build_prompt(_snapshot(None))[1]
    assert build_prompt(_snapshot(LOW))[0] == build_prompt(_snapshot(None))[0]  # the system prompt never moves


def _agent_state(tmp_path, config):
    clock = ManualClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
    h = make_harness(tmp_path, config, clock)
    try:
        h.send_human("Explain this stack trace.")
        [work] = [w for w in h.pending_work() if (w.snapshot.get("context") or {}).get("agent_state")]
        return h.ctx.agent_id, work.snapshot["context"]["agent_state"]
    finally:
        h.close()


def test_the_snapshot_carries_todays_mood(tmp_path):
    agent_id, state = _agent_state(tmp_path, Config.from_mapping({"rng_seed": 1}))
    assert state["mood"] == daily_mood(agent_id, date(2026, 10, 5))


def test_it_can_be_turned_off(tmp_path):
    _, state = _agent_state(tmp_path, Config.from_mapping({"rng_seed": 1, "temperament": {"daily_mood": False}}))
    assert "mood" not in state
