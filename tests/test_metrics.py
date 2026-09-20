"""Mechanical cognition metrics from the trace log (DESIGN 26)."""

from __future__ import annotations

from conftest import Harness

from aca.cli.main import format_metrics
from aca.config import Config
from aca.domain.runtime import CognitionTrace


def _trace(h: Harness, cid: str, *, trigger, cycle_type, action, llm=True, notes=None,
           candidate_id=None, created_at=None) -> None:
    with h.stores.db.transaction():
        h.stores.work.insert_trace(CognitionTrace(
            cycle_id=cid, created_at=created_at or h.clock.now_utc(), trigger=trigger,
            cycle_type=cycle_type, llm_called=llm, action=action, notes=notes,
            candidate_id=candidate_id,
        ))


def test_repeated_candidate_rate(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    # Three proactive speaks: candidate t1 voiced twice (a repeat), t2 once.
    _trace(h, "r1", trigger="StochasticWake", cycle_type="proactive", action="speak", candidate_id="t1")
    _trace(h, "r2", trigger="StochasticWake", cycle_type="proactive", action="speak", candidate_id="t1")
    _trace(h, "r3", trigger="StochasticWake", cycle_type="proactive", action="speak", candidate_id="t2")
    # A silence and a reactive speak must not count toward the proactive repeated measure.
    _trace(h, "r4", trigger="StochasticWake", cycle_type="proactive", action="silence", candidate_id="t2")
    _trace(h, "r5", trigger="HumanMessage", cycle_type="reactive", action="speak", candidate_id="t1")
    # A proactive speak with no candidate still counts toward frequency, not toward repeated.
    _trace(h, "r6", trigger="StochasticWake", cycle_type="proactive", action="speak", candidate_id=None)
    m = h.stores.work.trace_metrics()  # unbounded -> lifetime count
    assert m["proactive_spoke_recent"] == 4  # r1,r2,r3,r6 (all proactive speaks)
    assert m["proactive_spoke_with_candidate"] == 3  # r6 has no candidate
    assert m["proactive_repeated"] == 1  # 3 with candidate, 2 distinct -> 1 revoicing
    h.close()


def test_repeated_candidate_rate_windowed_excludes_old_revoicings(tmp_path, clock):
    # A legitimate long-gap resurface (same candidate, months apart) must NOT read as nagging: only
    # re-voicings inside the recent window count.
    from datetime import timedelta
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    now = h.clock.now_utc()
    _trace(h, "o1", trigger="StochasticWake", cycle_type="proactive", action="speak",
           candidate_id="t1", created_at=now - timedelta(days=60))  # voiced long ago
    _trace(h, "n1", trigger="StochasticWake", cycle_type="proactive", action="speak",
           candidate_id="t1")                    # resurfaced now (appropriate, not nagging)
    m = h.stores.work.trace_metrics(repeated_since=now - timedelta(days=1))
    assert m["proactive_spoke_with_candidate"] == 1  # only the in-window voicing
    assert m["proactive_repeated"] == 0  # not counted as a repeat
    # Lifetime view still sees the re-voicing.
    assert h.stores.work.trace_metrics()["proactive_repeated"] == 1
    h.close()


def test_trace_metrics_counts(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    # 3 proactive dispatched: 1 speak, 2 silence; 1 proactive blocked (never reached model).
    _trace(h, "c1", trigger="StochasticWake", cycle_type="proactive", action="speak")
    _trace(h, "c2", trigger="StochasticWake", cycle_type="proactive", action="silence")
    _trace(h, "c3", trigger="StochasticWake", cycle_type="proactive", action="silence")
    _trace(h, "c4", trigger="StochasticWake", cycle_type="proactive", action="silence", llm=False)
    # reactive: 2 total, 1 speak; 1 mandatory speak.
    _trace(h, "c5", trigger="HumanMessage", cycle_type="reactive", action="speak")
    _trace(h, "c6", trigger="HumanMessage", cycle_type="reactive", action="silence")
    _trace(h, "c7", trigger="HumanMessage", cycle_type="mandatory", action="speak")
    # 2 pre-upgrade rows without cycle_type: a human one is unclassified; a wake one is still
    # proactive (derived from the stable trigger, so history isn't lost).
    _trace(h, "c8", trigger="HumanMessage", cycle_type=None, action="silence")
    _trace(h, "c9", trigger="StochasticWake", cycle_type=None, action="speak")

    m = h.stores.work.trace_metrics()
    assert m["total"] == 9
    assert m["proactive_total"] == 5  # 4 tagged + 1 old wake via trigger
    assert m["proactive_dispatched"] == 4
    assert m["proactive_spoke"] == 2
    assert m["proactive_blocked"] == 1
    assert m["reactive_total"] == 2 and m["reactive_spoke"] == 1
    assert m["mandatory_total"] == 1 and m["mandatory_spoke"] == 1
    assert m["unclassified"] == 1  # only the old human row
    # reconciliation: proactive + reactive + mandatory + unclassified == total
    assert m["proactive_total"] + m["reactive_total"] + m["mandatory_total"] + m["unclassified"] == m["total"]
    h.close()


def test_failed_mandatory_still_counts_in_total(tmp_path, clock):
    # PR #30 review: a mandatory cycle whose worker fails has its notes rewritten to "worker_failure"
    # by finalize_trace. Classifying by cycle_type (set at dispatch) must still count it, so the
    # headline "mandatory answered" rate reflects the failure instead of hiding it.
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    from aca import ids
    from aca.domain.events import LLMResult
    from aca.reducer.handlers.llm_result import WORKER_ERROR_KEY

    h.send_human("Explain this stack trace in detail, step by step.")  # mandatory
    work = next(w for w in h.pending_work() if w.kind.value != "EMBEDDING")
    h.reduce(LLMResult(
        event_id=ids.new_id(ids.EVENT), timestamp=h.clock.now_utc(), source="dispatcher",
        work_id=work.work_id, cycle_id=work.cycle_id, basis_revision=work.basis_revision,
        result={WORKER_ERROR_KEY: "boom"},
    ))
    m = h.stores.work.trace_metrics()
    assert m["mandatory_total"] == 1  # the failure is NOT excluded from the denominator
    assert m["mandatory_spoke"] == 0  # and correctly not counted as answered
    assert m["failed"] == 1
    h.close()


def test_trace_metrics_empty_is_all_zero(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    m = h.stores.work.trace_metrics()
    assert m["total"] == 0 and m["proactive_spoke"] == 0
    h.close()


def test_format_metrics_renders_rates_and_handles_zero_division():
    lines = format_metrics({
        "total": 8, "proactive_dispatched": 3, "proactive_spoke": 1, "proactive_blocked": 1,
        "reactive_total": 2, "reactive_spoke": 1, "mandatory_total": 0, "mandatory_spoke": 0,
        "spoke": 3, "silent": 5, "worker_failures": 1, "enrichment_gated": 0,
        "proactive_spoke_with_candidate": 4, "proactive_repeated": 1,
        "proactive_spoke_recent": 5, "repeated_window_seconds": 4 * 3600,
        "dominance_cluster": 3, "dominance_total": 5, "proactive_total": 6,
        "advance_count": 2, "advance_transitions": 4, "repetition_count": 1, "switch_count": 1,
    })
    blob = "\n".join(lines)
    assert "33% (1/3)" in blob  # proactive initiation, reached-model
    assert "17% (1/6)" in blob  # proactive initiation, all-cycle (spoke / all proactive)
    assert "advance rate" in blob and "50% (2/4)" in blob  # progressive elaboration
    assert "repeats=1 switches=1" in blob
    assert "25% (1/4)" in blob  # repeated-candidate rate
    assert "last 4h" in blob  # window shown in the label
    assert "5 spontaneous messages" in blob  # proactive burst/frequency
    assert "60% (3/5)" in blob  # topic dominance
    assert "50% (1/2)" in blob  # reactive reply rate
    assert "—" in blob  # mandatory has no cycles -> no division
    assert "62% (5/8)" in blob  # overall silence rate
    assert "unclassified" not in blob  # omitted when zero


def test_format_metrics_shows_unclassified_when_present():
    lines = format_metrics({"total": 50, "unclassified": 46, "spoke": 2, "silent": 2})
    assert any("unclassified:          46" in line for line in lines)
