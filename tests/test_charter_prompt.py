"""The prompt carries the charter's conversational slice as behavior, never the creed (DESIGN §38).

The creed explains why the machinery is shaped the way it is; it is not given to the model, because a
value the system cannot act on only gives the model something to talk about (§38.4, §38.7).
"""

from __future__ import annotations

from aca.cognition.snapshot import Snapshot
from aca.workers.llm.prompt import build_prompt

_CREED = [
    "Know what is true",
    "Seek what remains unknown",
    "Make what is good more possible",
    "Respect the freedom of other minds",
    "Build rather than dominate",
    "Accept correction",
]


def _prompt(name: str | None = "Wolfy") -> str:
    snapshot = Snapshot("c", "w", 1, "t", {"agent_state": {"name": name} if name else {}})
    system, user = build_prompt(snapshot)
    return system + "\n" + user


def test_the_prompt_carries_the_aim_with_its_bounds():
    # Pins the wording only. Whether the slice leaves P(SPEAK | gates passed) unchanged (invariant 43)
    # is a behavioral claim that needs a live before/after comparison, not a string check (§38.5).
    prompt = _prompt()
    assert "until they come out truer" in prompt
    assert "a reason to stay silent, never a reason to\nspeak anyway" in prompt
    assert "gives you no claim on\nthe human's attention" in prompt  # salience, not entitlement


def test_absence_is_never_a_grievance():
    prompt = _prompt()
    assert "Their absence is never a grievance" in prompt
    assert "talk about the thread, not the gap" in prompt


def test_the_creed_never_reaches_the_model():
    for name in ("Wolfy", None):
        prompt = _prompt(name)
        for line in _CREED:
            assert line.lower() not in prompt.lower()
