"""End-to-end counterfactual gating eval (DESIGN §27): ablate a real gate, classify what it withheld.

Drives the deterministic reducer twice on the *same* proactive scenario — once with the §34 discourse
gate active (baseline) and once with it ablated (``memory.discourse_bridge_cosine = 0`` disables it) —
and feeds both observable decision streams to ``aca.eval.gating``. An off-focus (ORPHAN) candidate is
muted in the baseline and spoken when the gate is off, so the eval sees exactly one suppression; the
injected oracle then decides whether that suppression was right (a low-value orphan the gate correctly
withheld) or wrong (a valuable resurfacing the gate silenced — a false negative).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.enums import EnrichmentStatus, OutboundKind
from aca.domain.state import ProvisionalMemory, Topic
from aca.eval.counterfactual import proactive_decisions
from aca.eval.gating import evaluate_gate
from aca.workers.base import LLMOutput

_RATE = half_life_to_rate_per_hour(24.0)
_FOCUS_VEC = [1.0, 0.0, 0.0]
_ORPHAN_VEC = [0.0, 1.0, 0.0]  # cosine 0 to the focus < bridge (0.55) -> ORPHAN while IDLE


class _SpeakLLM:
    """A worker that always speaks a fixed line — so whether a proactive message appears is decided
    purely by the gates, not by model restraint."""

    async def run(self, snapshot) -> LLMOutput:
        return LLMOutput(result={"action": "speak", "message": "About that earlier thread —"},
                         tokens_in=8, tokens_out=6)


def _config(bridge_cosine: float) -> Config:
    return Config.from_mapping({
        "rng_seed": 4,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        "conversation": {"active_within": "10s", "idle_within": "1h"},  # 60s-old turn -> IDLE
        "memory": {"discourse_bridge_cosine": bridge_cosine},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })


def _effects(h: Harness) -> dict[str, str]:
    return {m.candidate_id: m.payload
            for m in h.stores.outbox.pending_by_kind(OutboundKind.PROACTIVE)
            if m.candidate_id is not None}


def _run(tmp_path, *, bridge_cosine: float):
    """One IDLE wake with a focus subject and an off-focus candidate; return (traces, effects)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    h = Harness(tmp_path, _config(bridge_cosine), ManualClock(datetime(2026, 6, 1, 12, 0, tzinfo=UTC)),
                llm=_SpeakLLM())
    now = h.clock.now_utc()
    with h.stores.db.transaction():
        # A focus subject (enriched, so it stays out of the candidate pool) with an embedding, and an
        # IDLE cadence so the discourse gate is active.
        h.stores.memory.insert_memory(ProvisionalMemory(
            id="focus_mem", event_id="e", text="the current subject", activation=0.8, salience=0.8,
            decay_rate_per_hour=_RATE, created_at=now, last_activated_at=now,
            enrichment_status=EnrichmentStatus.ENRICHED,
        ))
        h.stores.memory.insert_embedding("focus_emb", "focus_mem", "m", _FOCUS_VEC, now)
        conv = h.stores.state.load_conversation()
        h.stores.state.save_conversation(conv.__class__(
            last_human_message_at=now - timedelta(seconds=60), focus_memory_id="focus_mem"))
        # The off-focus candidate the wake will select.
        h.stores.memory.insert_topic(Topic(
            id="t_orphan", summary="t_orphan", activation=0.9, importance=0.9, decay_rate_per_hour=_RATE,
            created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None))
        h.stores.memory.insert_topic_embedding("t_orphan", "m", _ORPHAN_VEC, now)
    h.wake()
    h.run_all_pending()
    traces = list(reversed(h.stores.work.recent_traces(limit=10)))
    effects = _effects(h)
    h.close()
    return traces, effects


def _streams(tmp_path):
    b_traces, b_fx = _run(tmp_path / "baseline", bridge_cosine=0.55)   # §34 gate active
    a_traces, a_fx = _run(tmp_path / "ablated", bridge_cosine=0.0)     # §34 gate disabled
    baseline = proactive_decisions(b_traces, effects=b_fx)
    ablated = proactive_decisions(a_traces, effects=a_fx)
    return baseline, ablated


def test_ablation_exposes_one_discourse_suppression(tmp_path):
    baseline, ablated = _streams(tmp_path)
    # Baseline muted the orphan; the ablated run (gate off) spoke it — exactly one suppression.
    assert [(d.candidate_id, d.action) for d in baseline] == [("t_orphan", "silence")]
    assert [(d.candidate_id, d.action) for d in ablated] == [("t_orphan", "speak")]

    ge = evaluate_gate("discourse", baseline, ablated, worth=lambda c: False)  # orphan wasn't worth it
    assert ge.suppressed == 1
    assert ge.suppressions[0].candidate_id == "t_orphan"
    assert ge.suppressions[0].would_say == "About that earlier thread —"  # what the gate withheld
    assert ge.false_negative_count == 0
    assert ge.suppression_precision == 1.0  # the gate correctly withheld low-value speech


def test_oracle_marks_a_valuable_suppression_as_a_false_negative(tmp_path):
    # Same real ablation, opposite verdict: the withheld thought WAS worth saying, so the discourse
    # gate's suppression is a false negative — the measurable "valuable candidate silenced" signal.
    baseline, ablated = _streams(tmp_path)
    ge = evaluate_gate("discourse", baseline, ablated, worth=lambda c: c == "t_orphan")
    assert ge.suppressed == 1
    assert ge.false_negative_count == 1
    assert ge.suppression_precision == 0.0
