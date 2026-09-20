"""Pure unit tests for the counterfactual diff (no reducer, no I/O).

These pin the instrument's contract directly on synthetic traces: what counts as a proactive
decision, what is compared (the observable outcome), and what is deliberately ignored (trigger,
cycle_id, candidate identity).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aca.domain.runtime import CognitionTrace
from aca.eval.counterfactual import compare, proactive_decisions

_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _trace(cycle_id, *, trigger="StochasticWake", candidate_id=None, action=None):
    return CognitionTrace(
        cycle_id=cycle_id, created_at=_NOW, trigger=trigger,
        candidate_id=candidate_id, action=action,
    )


def test_only_stochastic_wake_cycles_are_proactive_decisions():
    traces = [
        _trace("c1", trigger="HumanMessage", candidate_id="m1", action="speak"),
        _trace("c2", trigger="StochasticWake", candidate_id="m2", action="speak"),
        _trace("c3", trigger="Obligation", candidate_id="m3", action="speak"),
    ]
    decisions = proactive_decisions(traces)
    assert [d.cycle_id for d in decisions] == ["c2"]  # reactive/mandatory cycles filtered out


def test_identical_streams_are_unchanged():
    a = [_trace("c1", candidate_id="m1", action="speak")]
    b = [_trace("c1", candidate_id="m1", action="speak")]
    result = compare(a, b)
    assert not result.changed
    assert result.changed_cycles == 0


def test_cycle_id_and_candidate_are_excluded_from_the_diff():
    # Different cycle ids (fresh ids.new_id per run) and different internal picks, identical observable
    # outcome (same action, same effect) -> NOT changed. This is the theater case at the unit level.
    base = [_trace("c_aaa", candidate_id="m1", action="speak")]
    abl = [_trace("c_zzz", candidate_id="m2", action="speak")]
    fx = {"m1": "hello", "m2": "hello"}  # both say the same thing
    result = compare(base, abl, baseline_effects={"m1": fx["m1"]}, ablated_effects={"m2": fx["m2"]})
    assert not result.changed
    assert result.changed_cycles == 0


def test_effect_flip_counts_even_with_same_candidate_and_action():
    base = [_trace("c1", candidate_id="m1", action="speak")]
    abl = [_trace("c1", candidate_id="m1", action="speak")]
    result = compare(base, abl, baseline_effects={"m1": "one"}, ablated_effects={"m1": "two"})
    assert result.changed  # same pick, same action, different words -> observable change
    assert result.changed_cycles == 1


def test_action_only_flip_counts():
    base = [_trace("c1", candidate_id="m1", action="speak")]
    abl = [_trace("c1", candidate_id="m1", action="silence")]
    result = compare(base, abl)
    assert result.changed
    assert result.changed_cycles == 1


def test_length_mismatch_contributes_to_changed_cycles():
    base = [_trace("c1", candidate_id="m1", action="speak"),
            _trace("c2", candidate_id="m2", action="speak")]
    abl = [_trace("c1", candidate_id="m1", action="speak")]
    result = compare(base, abl)
    assert result.changed
    assert result.changed_cycles == 1  # one extra positional decision on the baseline side
