"""Proactive block-reason decomposition (DESIGN 26): why did proactive cycles NOT speak?

``WorkStore.proactive_block_reasons`` buckets non-speaking ``StochasticWake`` cycles by the trace
``notes`` the reducer already records — read-only observability, the foundation for the offline
counterfactual gating eval. These tests pin the note-string → bucket mapping against the *actual*
literals the reducer emits (wake ``blocked:{reason}``, pre-outbox ``pre_outbox:{reason}``, the
advancement tags from §35.4, the model's own silence), so a new suppression path can't silently fall
into ``other`` unnoticed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from conftest import Harness

from aca.cli.main import _format_block_reasons
from aca.clock import ManualClock
from aca.cognition.activation import half_life_to_rate_per_hour
from aca.config import Config
from aca.domain.runtime import CognitionTrace
from aca.domain.state import Topic
from aca.persistence.stores import Stores
from aca.workers.base import LLMOutput

_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

# Every proactive non-speak note the reducer can write, paired with the bucket it must land in.
# (Kept exhaustive on purpose — it is the regression guard for the classifier.)
_NOTE_TO_BUCKET = [
    ("nothing", "nothing"),
    ("low_worth", "low_worth"),
    ("llm_budget", "budget"),
    ("blocked:budget_exhausted", "budget"),
    ("pre_outbox:budget_exhausted", "budget"),
    ("blocked:cooldown_active", "cooldown"),
    ("pre_outbox:cooldown_active", "cooldown"),
    ("blocked:quiet_hours", "quiet_hours"),
    ("blocked:mode_suppresses_initiative", "mode_suppressed"),
    ("pre_outbox:mode_suppresses_initiative", "mode_suppressed"),
    ("blocked:capability_denied", "mode_suppressed"),
    ("enrichment_gated:low_salience", "enrichment_gated"),
    ("discourse_orphan", "discourse_orphan_wake"),
    ("continuity_repeat", "continuity_repeat_wake"),
    ("pre_outbox:discourse_orphan", "discourse_orphan_recheck"),
    ("pre_outbox:continuity_repeat", "continuity_repeat_recheck"),
    ("pre_outbox:advancement_orphan", "advancement"),
    ("pre_outbox:advancement_repeat", "advancement"),
    ("pre_outbox:newer_human_event", "superseded"),
    ("pre_outbox:candidate_resolved_or_gone", "superseded"),
    # A model silence leaves the dispatch tag in place (finalize writes note=None -> COALESCE keeps
    # 'proactive_dispatch'), flipping only the action to silence — so the persisted note is the
    # dispatch tag, NOT the transient 'proactive_silence' HandlerOutcome note.
    ("proactive_dispatch", "model_silence"),
    ("enrichment_only", "enrichment_only"),
    ("a_brand_new_untagged_reason", "other"),
]


def _stores(tmp_path) -> Stores:
    return Stores.open(tmp_path / "agent.db")


def _add_proactive(stores, i: int, notes: str | None, *, action: str = "silence") -> None:
    dispatched = bool(notes and notes.startswith(("pre_outbox", "proactive", "enrichment_only")))
    stores.work.insert_trace(CognitionTrace(
        cycle_id=f"cyc_{i}", created_at=_T0 + timedelta(seconds=i), trigger="StochasticWake",
        cycle_type="proactive", llm_called=dispatched, action=action, notes=notes,
    ))


def test_every_reducer_note_maps_to_its_bucket(tmp_path):
    stores = _stores(tmp_path)
    for i, (notes, _bucket) in enumerate(_NOTE_TO_BUCKET):
        _add_proactive(stores, i, notes)

    reasons = stores.work.proactive_block_reasons()
    expected: dict[str, int] = {}
    for _notes, bucket in _NOTE_TO_BUCKET:
        expected[bucket] = expected.get(bucket, 0) + 1

    for bucket, count in expected.items():
        assert reasons[bucket] == count, f"{bucket}: got {reasons[bucket]}, want {count}"
    # Partition property: the buckets sum to exactly the non-speaking proactive cycles.
    assert sum(reasons.values()) == len(_NOTE_TO_BUCKET)
    # Nothing unexpected leaked into `other`.
    assert reasons["other"] == 1


def test_stable_ordered_shape_with_zeros(tmp_path):
    # Every bucket is present (zeros included) and in pipeline order, so downstream rendering/diffing
    # sees a stable shape regardless of which reasons actually occurred.
    reasons = _stores(tmp_path).work.proactive_block_reasons()
    assert list(reasons) == list(_stores(tmp_path).work._BLOCK_REASON_ORDER)
    assert set(reasons.values()) == {0}


def test_speaks_and_non_proactive_cycles_are_excluded(tmp_path):
    stores = _stores(tmp_path)
    _add_proactive(stores, 0, "nothing")                       # counted
    _add_proactive(stores, 1, None, action="speak")            # a proactive SPEAK — excluded
    # A reactive human-turn silence must not count as a proactive block.
    stores.work.insert_trace(CognitionTrace(
        cycle_id="r0", created_at=_T0, trigger="HumanMessage", cycle_type="reactive",
        action="silence", notes="reactive_silence",
    ))
    reasons = stores.work.proactive_block_reasons()
    assert sum(reasons.values()) == 1
    assert reasons["nothing"] == 1


def test_since_windows_the_decomposition(tmp_path):
    stores = _stores(tmp_path)
    stores.work.insert_trace(CognitionTrace(
        cycle_id="old", created_at=_T0, trigger="StochasticWake", cycle_type="proactive",
        action="silence", notes="nothing",
    ))
    stores.work.insert_trace(CognitionTrace(
        cycle_id="new", created_at=_T0 + timedelta(hours=2), trigger="StochasticWake",
        cycle_type="proactive", action="silence", notes="low_worth",
    ))
    recent = stores.work.proactive_block_reasons(since=_T0 + timedelta(hours=1))
    assert recent["low_worth"] == 1 and recent["nothing"] == 0
    assert sum(recent.values()) == 1


class _SilentLLM:
    """A proactive worker that always chooses silence — the model's own restraint."""

    async def run(self, snapshot) -> LLMOutput:
        return LLMOutput(result={"action": "silence"}, tokens_in=5, tokens_out=1)


def test_end_to_end_model_silence_is_bucketed(tmp_path):
    # Guards the real reducer literal against classifier drift: a dispatched proactive cycle the
    # model silences must land in `model_silence`, not `other`. (DORMANT so initiative is allowed and
    # the wake reaches the model; the deterministic gates fail open with no focus.)
    config = Config.from_mapping({
        "rng_seed": 3,
        "cognition": {"selection_temperature": 0.2, "null_candidate_score": -10.0,
                      "semantic_worthiness_floor": 0.2},
        "timing": {"proactive_cooldown": "0s"},
        "budgets": {"proactive_messages_per_hour": 100, "proactive_llm_calls_per_hour": 100,
                    "proactive_llm_calls_per_day": 1000},
    })
    h = Harness(tmp_path, config, ManualClock(_T0), llm=_SilentLLM())
    now = h.clock.now_utc()
    rate = half_life_to_rate_per_hour(24.0)
    with h.stores.db.transaction():
        h.stores.memory.insert_topic(Topic(
            id="t1", summary="t1", activation=0.9, importance=0.9, decay_rate_per_hour=rate,
            created_at=now, last_activated_at=now, unfinished=False, source_memory_id=None,
        ))
    h.wake()
    h.run_all_pending()
    reasons = h.stores.work.proactive_block_reasons()
    assert reasons["model_silence"] == 1
    assert sum(reasons.values()) == 1  # exactly one non-speaking proactive cycle, correctly placed
    h.close()


def test_format_block_reasons_renders_nonzero_in_order():
    rendered = _format_block_reasons({"nothing": 0, "budget": 3, "model_silence": 5, "other": 0})
    assert rendered == "budget=3  model_silence=5"       # non-zero only, dict order preserved
    assert _format_block_reasons({"budget": 0, "nothing": 0}) == "(none)"
    assert _format_block_reasons({}) == "(none)"
