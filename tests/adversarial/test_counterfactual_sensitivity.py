"""Counterfactual sensitivity: memory must causally change a later choice (else it's theater).

ACA's determinism (seeded Rng + ManualClock) lets us replay the same wake with a memory present vs
ablated and diff the proactive selection. The falsifiable pair: ablating the memory the agent would
*use* changes its choice; ablating an unrelated one does not.
"""

from __future__ import annotations

from datetime import UTC, datetime

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus
from aca.domain.state import ProvisionalMemory
from aca.eval.counterfactual import compare


def _config():
    # Near-deterministic selection so the counterfactual is crisp: low temperature, NOTHING never
    # wins, low worthiness floor so a salient memory is chosen.
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
    })


def _seed(h: Harness, mid: str, activation: float) -> None:
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        h.stores.memory.insert_memory(ProvisionalMemory(
            id=mid, event_id="e", text=mid, activation=activation, salience=activation,
            decay_rate_per_hour=half_life_to_rate_per_hour(24.0),
            created_at=now, last_activated_at=now, enrichment_status=EnrichmentStatus.RAW,
        ))


def _run(tmp_path, present: list[tuple[str, float]]) -> list:
    """Seed the given memories, fire one wake, return traces oldest-first."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    clock = ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC))
    h = Harness(tmp_path, _config(), clock)
    for mid, act in present:
        _seed(h, mid, act)
    h.wake()
    traces = list(reversed(h.stores.work.recent_traces(limit=10)))
    h.close()
    return traces


def test_ablating_the_used_memory_changes_the_choice(tmp_path):
    winner, runner_up = "mem_winner", "mem_runnerup"
    baseline = _run(tmp_path / "b", [(winner, 0.95), (runner_up, 0.90)])
    ablated = _run(tmp_path / "a", [(runner_up, 0.90)])  # the winning memory removed
    result = compare(baseline, ablated)
    # Baseline chose the winner; with it ablated, a different candidate is chosen.
    assert any(d.candidate_id == winner for d in result.baseline)
    assert result.changed  # memory had a causal consequence — not theater
    assert all(d.candidate_id != winner for d in result.ablated)


def test_ablating_an_unused_memory_does_not_change_the_choice(tmp_path):
    winner, loser = "mem_winner", "mem_loser"
    baseline = _run(tmp_path / "b", [(winner, 0.95), (loser, 0.30)])
    ablated = _run(tmp_path / "a", [(winner, 0.95)])  # remove the low-activation loser
    result = compare(baseline, ablated)
    assert not result.changed  # ablating a memory it wasn't going to use leaves the choice intact
    assert result.changed_cycles == 0
