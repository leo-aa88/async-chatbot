"""Mechanical cognition metrics from the trace log (DESIGN 26)."""

from __future__ import annotations

from conftest import Harness

from aca.cli.main import format_metrics
from aca.config import Config
from aca.domain.runtime import CognitionTrace


def _trace(h: Harness, cid: str, *, trigger, cycle_type, action, llm=True, notes=None) -> None:
    with h.stores.db.transaction():
        h.stores.work.insert_trace(CognitionTrace(
            cycle_id=cid, created_at=h.clock.now_utc(), trigger=trigger, cycle_type=cycle_type,
            llm_called=llm, action=action, notes=notes,
        ))


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
    })
    blob = "\n".join(lines)
    assert "33% (1/3)" in blob  # proactive initiation
    assert "50% (1/2)" in blob  # reactive reply rate
    assert "—" in blob  # mandatory has no cycles -> no division
    assert "62% (5/8)" in blob  # overall silence rate
    assert "unclassified" not in blob  # omitted when zero


def test_format_metrics_shows_unclassified_when_present():
    lines = format_metrics({"total": 50, "unclassified": 46, "spoke": 2, "silent": 2})
    assert any("unclassified:          46" in line for line in lines)
