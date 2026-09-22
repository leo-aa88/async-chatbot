"""Unit tests for the counterfactual gating eval (DESIGN §27): pure functions over decision streams.

No reducer here — synthetic ``ProactiveDecision`` streams isolate the classification logic (which
positions a gate suppressed, and how the injected oracle splits them into right vs false-negative).
The end-to-end wiring against a real gate ablation lives in
``tests/adversarial/test_counterfactual_gating.py``.
"""

from __future__ import annotations

from aca.eval.counterfactual import ProactiveDecision
from aca.eval.gating import evaluate, evaluate_gate, gate_suppressions, over_speech


def _d(cycle: str, candidate: str | None, action: str, effect: str | None = None) -> ProactiveDecision:
    return ProactiveDecision(cycle, candidate, action, effect=effect)


def test_gate_suppressions_are_silent_baseline_spoken_ablated():
    baseline = (_d("b0", "c1", "silence"), _d("b1", "c2", "speak", "kept"))
    ablated = (_d("a0", "c1", "speak", "withheld"), _d("a1", "c2", "speak", "kept"))
    sup = gate_suppressions(baseline, ablated)
    assert len(sup) == 1
    assert sup[0].candidate_id == "c1"
    assert sup[0].would_say == "withheld"   # the payload the gate withheld (from the ablated run)
    assert sup[0].cycle_id == "a0"          # the ablated run's id, for inspection


def test_no_suppression_when_ablation_changes_nothing():
    # Same outcome with the gate on or off -> the gate suppressed nothing here.
    stream = (_d("x", "c1", "silence"), _d("y", "c2", "speak", "s"))
    assert gate_suppressions(stream, stream) == ()


def test_evaluate_gate_splits_false_negatives_by_oracle():
    baseline = (_d("b", "orphan", "silence"),)
    ablated = (_d("a", "orphan", "speak", "the withheld thought"),)
    # Oracle says the withheld utterance WAS worth saying -> the gate silenced value (false negative).
    fn = evaluate_gate("discourse", baseline, ablated, worth=lambda c: c == "orphan")
    assert fn.suppressed == 1 and fn.false_negative_count == 1
    assert fn.correct_suppressions == 0
    assert fn.suppression_precision == 0.0
    # Oracle says it was NOT worth saying -> the gate did its job (a correct suppression).
    ok = evaluate_gate("discourse", baseline, ablated, worth=lambda c: False)
    assert ok.false_negative_count == 0 and ok.suppression_precision == 1.0


def test_suppression_precision_is_none_without_signal():
    # A gate that suppressed nothing has no evidence either way — precision must be None, not 1.0.
    ge = evaluate_gate("continuity", (_d("b", "c", "speak", "s"),), (_d("a", "c", "speak", "s"),),
                       worth=lambda c: True)
    assert ge.suppressed == 0
    assert ge.suppression_precision is None


def test_over_speech_is_spoken_but_not_worth():
    baseline = (_d("b0", "c1", "speak", "good"), _d("b1", "c2", "speak", "noise"),
                _d("b2", "c3", "silence"))
    fp = over_speech(baseline, worth=lambda c: c == "c1")   # only c1 is worth saying
    assert [d.candidate_id for d in fp] == ["c2"]           # c2 spoke but wasn't worth it; c3 silent


def test_evaluate_aggregates_gates_and_over_speech():
    baseline = (_d("b0", "c1", "silence"), _d("b1", "c2", "speak", "noise"))
    ablated_by_gate = {
        "discourse": (_d("a0", "c1", "speak", "withheld"), _d("a1", "c2", "speak", "noise")),
        "continuity": (_d("a0", "c1", "silence"), _d("a1", "c2", "speak", "noise")),  # suppressed nothing
    }
    report = evaluate(baseline, ablated_by_gate, worth=lambda c: c == "c1")
    s = report.summary()
    assert s["gates"]["discourse"] == {"suppressed": 1, "false_negatives": 1, "suppression_precision": 0.0}
    assert s["gates"]["continuity"] == {"suppressed": 0, "false_negatives": 0, "suppression_precision": None}
    assert s["over_speech"] == 1   # c2 spoke in baseline and isn't worth it
