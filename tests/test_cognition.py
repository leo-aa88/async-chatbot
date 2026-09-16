"""Unit tests for pure cognition: activation, classifier, selection, scheduler, gating, budgets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aca.cognition import activation as act
from aca.cognition import budgets as budget
from aca.cognition import scheduler
from aca.cognition.classifier import ClassificationContext, classify, is_trivial, jaccard_similarity
from aca.cognition.gating import (
    OptionalResponseFactors,
    ProactiveExpressionFactors,
    ProactiveHardGateInput,
    evaluate_proactive_hard_gates,
    optional_response_probability,
    proactive_speak_probability,
    quiet_hours_active,
    refractory_factor,
)
from aca.cognition.selection import Candidate, select
from aca.config import Budgets, QuietHours
from aca.domain.enums import CandidateKind, ConversationMode, MessageClass
from aca.rng import Rng

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# --- activation ---------------------------------------------------------------------------
def test_decay_halves_after_one_half_life():
    rate = act.half_life_to_rate_per_hour(24.0)
    later = T0 + timedelta(hours=24)
    assert act.decayed_activation(1.0, T0, later, rate) == pytest.approx(0.5, abs=1e-6)


def test_persistence_slows_decay():
    rate = 0.1
    later = T0 + timedelta(hours=10)
    fast = act.decayed_activation(1.0, T0, later, rate, persistence=0.0, alpha=0.6)
    slow = act.decayed_activation(1.0, T0, later, rate, persistence=1.0, alpha=0.6)
    assert slow > fast


def test_out_of_order_timestamp_never_increases_activation():
    earlier = T0 - timedelta(hours=5)
    assert act.decayed_activation(0.5, T0, earlier, 0.1) == pytest.approx(0.5)


# --- classifier ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["Explain this stack trace.", "Fix the bug in this code.",
                                  "what does this error mean?", "How do I refactor this?"])
def test_tasks_are_response_required(text):
    assert classify(text).response_required is True


@pytest.mark.parametrize("text", ["lol", "ok", "nice", "yeah"])
def test_acknowledgements_are_optional(text):
    result = classify(text)
    assert result.response_required is False
    assert result.message_class in (MessageClass.ACKNOWLEDGEMENT, MessageClass.LOW_INFORMATION)


def test_ambiguous_question_biases_to_task():
    # A non-social question routes to the response-required path (DESIGN 13.2, 20.1).
    assert classify("is the deployment done?").response_required is True


def test_reprompt_after_silence_forces_response():
    ctx = ClassificationContext(previous_human_text="did the build pass",
                                previous_turn_was_silent=True)
    result = classify("?", ctx)
    assert result.is_reprompt and result.response_required and result.possible_prior_miss


def test_trivial_detection():
    assert is_trivial("") and is_trivial("ok") and is_trivial("👍")
    assert not is_trivial("I finally got the prototype running")


def test_jaccard_similarity_bounds():
    assert jaccard_similarity("a b c", "a b c") == 1.0
    assert jaccard_similarity("a b", "c d") == 0.0


# --- selection ----------------------------------------------------------------------------
def test_selection_can_choose_nothing():
    # With a dominant NOTHING score and low candidate, NOTHING should sometimes win.
    rng = Rng(1)
    picks = [
        select([Candidate(CandidateKind.TOPIC, "t", 0.1)], null_score=3.0, temperature=0.5, rng=rng)
        for _ in range(50)
    ]
    assert any(p.is_nothing for p in picks)


def test_selection_is_deterministic_with_seed():
    c = [Candidate(CandidateKind.TOPIC, "t", 0.8), Candidate(CandidateKind.PROVISIONAL_MEMORY, "m", 0.4)]
    a = select(c, 0.5, 0.7, Rng(9)).candidate.id
    b = select(c, 0.5, 0.7, Rng(9)).candidate.id
    assert a == b


# --- scheduler ----------------------------------------------------------------------------
def test_lambda_excludes_expression_variables_and_scales_with_signals():
    from aca.config import Cognition

    cog = Cognition(spontaneous_activation_rate_per_hour=0.25)
    base = scheduler.compute_lambda(cog, scheduler.WakeSignals())
    louder = scheduler.compute_lambda(
        cog, scheduler.WakeSignals(active_observed_silence_hours=10, max_candidate_salience=1.0,
                                   unfinished_count=4)
    )
    assert base == pytest.approx(0.25)
    assert louder > base


# --- gating -------------------------------------------------------------------------------
def test_quiet_hours_wraps_midnight():
    quiet = QuietHours(enabled=True, start_local="22:00", end_local="06:00")
    assert quiet_hours_active(datetime(2026, 6, 1, 23, 0, tzinfo=UTC), quiet)
    assert quiet_hours_active(datetime(2026, 6, 1, 3, 0, tzinfo=UTC), quiet)
    assert not quiet_hours_active(datetime(2026, 6, 1, 12, 0, tzinfo=UTC), quiet)


def test_quiet_hours_disabled_never_active():
    assert not quiet_hours_active(T0, QuietHours(enabled=False))


def test_refractory_recovers_over_time():
    assert refractory_factor(None, 2.0) == 1.0
    assert refractory_factor(0.0, 2.0) == pytest.approx(0.0)
    assert refractory_factor(10.0, 2.0) > refractory_factor(1.0, 2.0)


def test_hard_gate_short_circuits_on_budget():
    gate = ProactiveHardGateInput(True, False, True, ConversationMode.DORMANT, False, False)
    assert evaluate_proactive_hard_gates(gate).reason == "budget_exhausted"


def test_active_mode_raises_optional_response_probability():
    common = dict(desire=0.5, relevance=0.6, initiative=0.5, inhibition=0.5)
    active = optional_response_probability(OptionalResponseFactors(active_mode=True, **common))
    idle = optional_response_probability(OptionalResponseFactors(active_mode=False, **common))
    assert active > idle


def test_inhibition_lowers_speak_probability():
    high = ProactiveExpressionFactors(0.8, 0.8, 0.5, inhibition=0.9, refractory_penalty=0.0)
    low = ProactiveExpressionFactors(0.8, 0.8, 0.5, inhibition=0.1, refractory_penalty=0.0)
    assert proactive_speak_probability(low) > proactive_speak_probability(high)


# --- budgets ------------------------------------------------------------------------------
def test_budget_window_ids_change_by_hour_and_day():
    assert budget.hour_window_id(T0) != budget.hour_window_id(T0 + timedelta(hours=1))
    assert budget.day_window_id(T0) != budget.day_window_id(T0 + timedelta(days=1))


def test_budget_blocks_when_exhausted():
    budgets = Budgets(proactive_llm_calls_per_day=2, proactive_llm_calls_per_hour=2)
    usage = budget.ProactiveUsage(llm_calls_today=2)
    assert not budget.proactive_llm_call_allowed(usage, budgets).allowed
