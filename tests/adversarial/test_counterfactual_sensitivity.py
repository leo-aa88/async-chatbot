"""Counterfactual sensitivity: memory must causally change *observable behavior*, else it's theater.

ACA's determinism (seeded Rng + ManualClock) lets us replay the same wake with a memory present vs
ablated and diff what the agent actually says. The falsifiable pair:

* a content-sensitive agent, with the used memory ablated, says something different (continuity);
* a content-insensitive agent that always emits the same line does NOT count as continuity even
  though its internal candidate pick moved (theater).

The full cycle is driven to completion (``run_all_pending``) so ``action`` is finalized and a
committed outbound payload exists; the diff keys on that observable payload, not the candidate id.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import ProvisionalMemory
from aca.eval.counterfactual import compare
from aca.workers.base import LLMOutput


class CannedLLMWorker:
    """A content-insensitive worker: always speaks the identical line, whatever the candidate.

    Stands in for a *theatrical* agent — it still stores memories and rides the same selection, but
    its output carries no trace of which memory won. Ablation must NOT register as continuity.
    """

    async def run(self, snapshot) -> LLMOutput:
        candidate = (snapshot.context.get("source") or {}).get("candidate") or {}
        mid = candidate.get("provisional_memory_id") or candidate.get("id")
        proposals = (
            [{"type": "ENRICH_PROVISIONAL_MEMORY", "provisional_memory_id": mid,
              "topic_summary": "canned"}]
            if mid else []
        )
        return LLMOutput(
            result={"action": "speak", "message": "Same thing, every time.", "proposals": proposals},
            tokens_in=100, tokens_out=8,
        )


def _config():
    # Argmax, not a seed lottery: near-zero temperature and a wide activation gap make the highest
    # logit win deterministically. NOTHING never wins; a salient memory is always eligible.
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.02, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
    })


def _seed(h: Harness, mid: str, activation: float, order: int) -> None:
    # Distinct last_activated_at per memory so pool order is deterministic (recent_memories has no
    # secondary sort); more recent == less decayed, which aligns with the intended winner.
    at = h.clock.now_utc() + timedelta(seconds=order)
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e", text=f"content about {mid}", activation=activation,
            salience=activation, decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=at, last_activated_at=at, enrichment_status=EnrichmentStatus.RAW,
        ))


def _effects(h: Harness) -> dict[str, str]:
    """candidate_id -> committed spoken payload for this run's proactive outbound (the observable)."""
    return {
        m.candidate_id: m.payload
        for m in h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)
        if m.candidate_id is not None
    }


def _run(tmp_path, present: list[tuple[str, float]], *, llm=None):
    """Seed memories, fire one wake, drain the cycle. Return (traces oldest-first, effects map)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    h = Harness(tmp_path, _config(), clock, llm=llm)
    for order, (mid, act) in enumerate(present):
        _seed(h, mid, act, order)
    h.wake()
    h.run_all_pending()  # finalize the cycle so action + outbound payload exist
    traces = list(reversed(h.stores.work.recent_traces(limit=10)))
    effects = _effects(h)
    h.close()
    return traces, effects


# Wide gap: winner logit 1.5*0.95=1.425 vs runner 1.5*0.65=0.975; at T=0.02 the winner is argmax.
# Runner salience 0.65 >= 0.6 speak threshold, so the ablated run still speaks (a different line).
_WINNER, _RUNNER = ("mem_winner", 0.95), ("mem_runnerup", 0.65)


def test_ablating_the_used_memory_changes_observable_behavior(tmp_path):
    (winner_id, _), (runner_id, _) = _WINNER, _RUNNER
    b_traces, b_fx = _run(tmp_path / "b", [_WINNER, _RUNNER])
    a_traces, a_fx = _run(tmp_path / "a", [_RUNNER])  # the winning memory removed
    result = compare(b_traces, a_traces, baseline_effects=b_fx, ablated_effects=a_fx)

    assert any(d.candidate_id == winner_id and d.action == "speak" for d in result.baseline)
    assert all(d.candidate_id != winner_id for d in result.ablated)
    assert any(d.candidate_id == runner_id for d in result.ablated)  # runner-up is the fallback pick
    assert result.changed  # the agent SAYS something different — a real causal consequence
    assert result.changed_cycles == 1


def test_theatrical_agent_is_not_counted_as_continuity(tmp_path):
    # Same ablation, but a content-insensitive worker: the candidate pick moves, the output does not.
    winner_id = _WINNER[0]
    b_traces, b_fx = _run(tmp_path / "b", [_WINNER, _RUNNER], llm=CannedLLMWorker())
    a_traces, a_fx = _run(tmp_path / "a", [_RUNNER], llm=CannedLLMWorker())
    result = compare(b_traces, a_traces, baseline_effects=b_fx, ablated_effects=a_fx)

    # The internal pick genuinely changed...
    assert any(d.candidate_id == winner_id for d in result.baseline)
    assert all(d.candidate_id != winner_id for d in result.ablated)
    # ...but the observable behavior did not, so this is theater, not continuity.
    assert not result.changed
    assert result.changed_cycles == 0


def test_ablating_an_unused_memory_does_not_change_behavior(tmp_path):
    loser = ("mem_loser", 0.30)  # below the speak threshold; never voiced
    b_traces, b_fx = _run(tmp_path / "b", [_WINNER, loser])
    a_traces, a_fx = _run(tmp_path / "a", [_WINNER])  # remove the memory it wasn't going to use
    result = compare(b_traces, a_traces, baseline_effects=b_fx, ablated_effects=a_fx)
    assert not result.changed
    assert result.changed_cycles == 0


def test_harness_agrees_with_itself_on_an_identical_pool(tmp_path):
    # Self-consistency: replaying the exact same scenario is never "changed" — determinism, and
    # proof the instrument doesn't drift on cycle_id/ids.new_id noise.
    b_traces, b_fx = _run(tmp_path / "b", [_WINNER, _RUNNER])
    a_traces, a_fx = _run(tmp_path / "a", [_WINNER, _RUNNER])
    result = compare(b_traces, a_traces, baseline_effects=b_fx, ablated_effects=a_fx)
    assert not result.changed
    assert result.changed_cycles == 0
