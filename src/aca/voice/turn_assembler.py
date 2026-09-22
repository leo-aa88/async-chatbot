"""Voice turn assembly: acoustic utterances → one conversational turn (DESIGN §36).

Segmentation answers *"where did continuous speech stop?"*; turn assembly answers *"has the human
yielded the floor?"* — two different questions. A VAD boundary is not a turn boundary: we pause to
think, breathe, and use fillers, so committing every utterance as its own ``HumanMessage`` makes the
agent answer before the human has finished (observed live: "So do you think that perhaps" / "uh, an
algorithm like yours is suitable." / "for AGI." answered as three turns).

This assembler joins consecutive utterances and commits the joined text as one turn only after the
human yields the floor, decided from **trailing silence in the captured frame stream** (fed by the
session as ``on_silence`` seconds) — never wall-clock after ASR, which would count Whisper latency
as human silence. Two thresholds: an ordinary turn commits after ``turn_gap``; a turn whose
transcript looks syntactically unfinished waits the longer ``continuation_gap`` (don't answer a
hesitation, but don't stall a finished sentence). ``max_turn`` is a safety cap that forces a commit
at the next silence boundary. Deterministic and pure (seconds in, one callback out) — no clock, no
I/O — so it is exhaustively unit-testable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

OnTurn = Callable[[str], Awaitable[None]]

# Tokens that mark a turn as *syntactically unfinished* when they are the last word — a trailing
# conjunction, article, preposition, or filler. Versioned in code (not user config): this is
# linguistic structure, not a tuning knob. Conservative; extend with evidence.
_UNFINISHED_TAIL = frozenset({
    "and", "or", "but", "so", "because", "that", "which", "who", "the", "a", "an", "to", "of",
    "for", "with", "in", "on", "at", "as", "if", "when", "while", "then", "than", "like", "about",
    "into", "uh", "um", "er", "hmm", "perhaps", "maybe", "i", "my", "your", "is", "are", "was",
})


def _ends_unfinished(text: str) -> bool:
    """Whether ``text`` ends on a word that syntactically expects more (a weak completeness signal)."""
    words = text.rstrip(".!?,;: ").lower().split()
    return bool(words) and words[-1] in _UNFINISHED_TAIL


class VoiceTurnAssembler:
    """Accumulate transcribed utterances into one turn; commit on floor-yield (DESIGN §36).

    ``on_turn`` receives the joined turn text exactly once per committed turn. All thresholds are in
    seconds, matching the ``on_silence`` the session feeds from the frame stream.
    """

    def __init__(
        self,
        on_turn: OnTurn,
        *,
        turn_gap_seconds: float,
        continuation_gap_seconds: float,
        max_turn_seconds: float,
    ) -> None:
        self._on_turn = on_turn
        self._turn_gap = turn_gap_seconds
        self._continuation_gap = continuation_gap_seconds
        self._max_turn = max_turn_seconds
        self._parts: list[str] = []
        self._turn_seconds = 0.0  # accumulated speech duration in the current turn (for max_turn)

    async def add(self, text: str, duration_seconds: float) -> None:
        """A transcribed acoustic utterance joins the in-progress turn (does not commit)."""
        text = text.strip()
        if text:
            self._parts.append(text)
            self._turn_seconds += max(0.0, duration_seconds)

    async def on_silence(self, silence_seconds: float) -> None:
        """Ongoing trailing silence (since the human last spoke). Commit when the floor is yielded."""
        if not self._parts:
            return
        gap = self._continuation_gap if _ends_unfinished(" ".join(self._parts)) else self._turn_gap
        if silence_seconds >= gap or self._turn_seconds >= self._max_turn:
            await self._commit()

    async def flush(self) -> None:
        """Commit any buffered turn (end of the audio stream)."""
        if self._parts:
            await self._commit()

    def clear(self) -> None:
        """Drop the in-progress turn WITHOUT committing it (no ``on_turn`` call).

        Used by the half-duplex gate when the agent takes the audio floor: any partially-buffered
        utterances are discarded rather than committed, so the agent's own captured speech can never
        be joined onto — or flushed as — a human turn. Distinct from :meth:`flush`, which commits."""
        self._parts = []
        self._turn_seconds = 0.0

    async def _commit(self) -> None:
        text = " ".join(self._parts).strip()
        self._parts = []
        self._turn_seconds = 0.0
        if text:
            await self._on_turn(text)
