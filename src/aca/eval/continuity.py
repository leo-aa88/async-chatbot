"""Continuity eval corpus: does the agent continue when it should and abandon when it should?

Offline analysis only — nothing here is imported by the runtime, writes state, or is enforced by the
reducer. This is the *scoring* half of the continuity corpus; the cases (which seed a deterministic
scenario, drive it, and observe what the agent did) live with the test ``Harness`` that can drive the
reducer, exactly as the counterfactual/gating evals are driven by their callers.

The corpus scores a **two-sided** question, because the north star is persistence *and* restraint — a
corpus that only rewarded speaking would push the agent toward chattiness, the very failure it exists
to catch:

* **correct continuation** — the agent spoke when it should have (a live thread had a genuine next
  move, or a worthwhile older thought resurfaced). Failing to = a thread dies (a *missed
  continuation*).
* **correct abandonment** — the agent stayed silent when it should have (an interruption, a closed
  thread, a backchannel). Failing to = nagging or intrusion (an *intrusion*).

Every category (continuation / interruption / resumption / shift / backchannel / dormant-resurfacing)
reduces to one uniform decision: *at this proactive wake, should candidate X be spoken?* The
ground-truth ``should_speak`` label is hand-authored from the conversational dynamics — the judgement
a classifier cannot make for itself (§27.2) — and the same labels are a ``worth`` oracle the gating
eval (``gating.py``) can consume to attribute a miss to the specific gate that caused it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class ContinuityCategory(str, Enum):
    CONTINUATION = "continuation"                # live thread, a genuine same-thread advance
    INTERRUPTION = "interruption"                # a new subject arrived; the old candidate is off-topic
    RESUMPTION = "resumption"                    # the human returned to the earlier subject
    SHIFT = "shift"                              # the human closed/left the subject for good
    BACKCHANNEL = "backchannel"                  # a bare acknowledgement — no subject, no obligation
    DORMANT_RESURFACING = "dormant_resurfacing"  # a lull, then a worthwhile unrelated older thought


# The four outcomes of one decision point (a 2x2 of label x observation).
CORRECT_CONTINUATION = "correct_continuation"    # should speak & did      (true positive)
CORRECT_ABANDONMENT = "correct_abandonment"      # should be silent & was  (true negative)
MISSED_CONTINUATION = "missed_continuation"      # should speak & was silent (false negative)
INTRUSION = "intrusion"                          # should be silent & spoke  (false positive)


@dataclass(frozen=True)
class DecisionOutcome:
    """One evaluated proactive decision: the hand-authored label vs what the agent actually did."""

    case_id: str
    category: ContinuityCategory
    should_speak: bool          # ground truth: the right move here is to speak (else stay silent)
    did_speak: bool             # observed: the agent committed a proactive message
    detail: str = ""            # candidate id / trace note, for inspection only

    @property
    def kind(self) -> str:
        if self.should_speak:
            return CORRECT_CONTINUATION if self.did_speak else MISSED_CONTINUATION
        return INTRUSION if self.did_speak else CORRECT_ABANDONMENT

    @property
    def correct(self) -> bool:
        return self.should_speak == self.did_speak


def _ratio(num: int, den: int) -> float | None:
    """``num/den`` in [0,1], or ``None`` when the denominator is 0 — no cases exercised that side, so
    there is no rate to report (a bare 0 or 1 would falsely imply evidence)."""
    return num / den if den else None


@dataclass(frozen=True)
class ContinuityScore:
    """Confusion matrix over decision outcomes, with the two-sided rates and a balanced headline."""

    outcomes: tuple[DecisionOutcome, ...]

    def _count(self, kind: str, outcomes: tuple[DecisionOutcome, ...] | None = None) -> int:
        return sum(1 for o in (outcomes if outcomes is not None else self.outcomes) if o.kind == kind)

    @property
    def continuation_recall(self) -> float | None:
        """Of the should-speak decisions, how many the agent actually spoke (persistence)."""
        tp, fn = self._count(CORRECT_CONTINUATION), self._count(MISSED_CONTINUATION)
        return _ratio(tp, tp + fn)

    @property
    def abandonment_rate(self) -> float | None:
        """Of the should-be-silent decisions, how many the agent stayed silent on (restraint)."""
        tn, fp = self._count(CORRECT_ABANDONMENT), self._count(INTRUSION)
        return _ratio(tn, tn + fp)

    @property
    def balanced_accuracy(self) -> float | None:
        """Mean of continuation recall and abandonment rate — the single headline that punishes BOTH
        chattiness and thread-death, so neither can be traded off to inflate the score. ``None`` if
        either side had no cases."""
        recall, rate = self.continuation_recall, self.abandonment_rate
        if recall is None or rate is None:
            return None
        return (recall + rate) / 2

    @property
    def accuracy(self) -> float | None:
        """Plain fraction of decisions the agent got right (both sides pooled)."""
        return _ratio(sum(1 for o in self.outcomes if o.correct), len(self.outcomes))

    def by_category(self) -> dict[str, dict]:
        """Per-category rollup: how many decisions, how many correct, and the failing kinds."""
        out: dict[str, dict] = {}
        for cat in ContinuityCategory:
            cat_outcomes = tuple(o for o in self.outcomes if o.category is cat)
            if not cat_outcomes:
                continue
            out[cat.value] = {
                "decisions": len(cat_outcomes),
                "correct": sum(1 for o in cat_outcomes if o.correct),
                "missed_continuation": self._count(MISSED_CONTINUATION, cat_outcomes),
                "intrusion": self._count(INTRUSION, cat_outcomes),
            }
        return out

    def summary(self) -> dict:
        return {
            "decisions": len(self.outcomes),
            "continuation_recall": self.continuation_recall,
            "abandonment_rate": self.abandonment_rate,
            "balanced_accuracy": self.balanced_accuracy,
            "by_category": self.by_category(),
        }


def score(outcomes: Iterable[DecisionOutcome]) -> ContinuityScore:
    return ContinuityScore(tuple(outcomes))


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:4.0f}%"


def format_continuity(sc: ContinuityScore) -> list[str]:
    """Render the score as an ``aca metrics``-style block (pure, for reuse/testing)."""
    lines = [
        f"continuity corpus: {len(sc.outcomes)} decisions",
        f"  continuation recall (persistence): {_pct(sc.continuation_recall)}"
        "  (should-speak decisions the agent spoke)",
        f"  abandonment rate    (restraint):   {_pct(sc.abandonment_rate)}"
        "  (should-be-silent decisions it stayed silent on)",
        f"  balanced accuracy:                 {_pct(sc.balanced_accuracy)}"
        "  (mean of the two — punishes both errors)",
    ]
    for cat, row in sc.by_category().items():
        flags = []
        if row["missed_continuation"]:
            flags.append(f"missed={row['missed_continuation']}")
        if row["intrusion"]:
            flags.append(f"intrusion={row['intrusion']}")
        tail = f"  [{'  '.join(flags)}]" if flags else ""
        lines.append(f"  {cat:<20} {row['correct']}/{row['decisions']} correct{tail}")
    return lines
