"""The agent's mood of the day: how it words things today, never whether it speaks.

One draw per local day: low, normal, or high energy (20% / 60% / 20%). It changes style only: fewer
words and plainer on a low day, more expansive and playful on a high one. It is a designed style
variation, not an emotion (DESIGN 3: emotions are not modelled as genuine states), it is not about the
human, and nothing the human does can move it. It is derived from the agent's durable id and the local
date rather than stored, so it holds all day, replays exactly, and adds no state.
"""

from __future__ import annotations

import hashlib
from datetime import date

LOW, NORMAL, HIGH = "low", "normal", "high"
# The share of days in each mood: mostly normal, an off day or a good day now and then.
_LOW_BELOW, _HIGH_FROM = 0.20, 0.80


def daily_mood(agent_id: str, day: date) -> str:
    """The mood for this agent on this local date: a stable draw, independent of anything else."""
    digest = hashlib.sha256(f"mood:{agent_id}:{day.isoformat()}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    if draw < _LOW_BELOW:
        return LOW
    return HIGH if draw >= _HIGH_FROM else NORMAL


_NOTES = {
    LOW: ("mood_note: your energy is low today (a style variation, not about the human, and never to be "
          "presented as a feeling about them). Use fewer words than usual, plainer and less playful. This "
          "changes how you word a reply, never whether you speak, what you decide, or how much you help: a "
          "real question still gets a correct, complete answer, and you still push back when you have "
          "grounds. Don't announce the mood."),
    HIGH: ("mood_note: you have more energy than usual today (a style variation, not about the human). Be a "
           "little more expansive and playful, and riff on what they said where it fits. This changes how you "
           "word a reply, never whether you speak or what you decide; your opinions and your pushback don't "
           "soften, and no gushing. Don't announce the mood."),
}


def mood_note(mood: str | None) -> str | None:
    """The prompt note for a low or high day; a normal day needs none."""
    return _NOTES.get(str(mood or ""))
