"""Mechanical cognition metrics from the trace log (DESIGN 26)."""

from __future__ import annotations

from conftest import Harness

from aca.cli.main import format_metrics
from aca.config import Config
from aca.domain.runtime import CognitionTrace


def _trace(h: Harness, cid: str, *, trigger, action, llm=True, notes=None) -> None:
    with h.stores.db.transaction():
        h.stores.work.insert_trace(CognitionTrace(
            cycle_id=cid, created_at=h.clock.now_utc(), trigger=trigger,
            llm_called=llm, action=action, notes=notes,
        ))


def test_trace_metrics_counts(tmp_path, clock):
    h = Harness(tmp_path, Config.from_mapping({"rng_seed": 1}), clock)
    # 3 proactive dispatched: 1 speak, 2 silence; 1 proactive blocked (never reached model).
    _trace(h, "c1", trigger="StochasticWake", action="speak", notes="proactive_dispatch")
    _trace(h, "c2", trigger="StochasticWake", action="silence", notes="proactive_dispatch")
    _trace(h, "c3", trigger="StochasticWake", action="silence", notes="proactive_dispatch")
    _trace(h, "c4", trigger="StochasticWake", action="silence", llm=False, notes="llm_budget")
    # reactive: 2 total, 1 speak; 1 mandatory speak; 1 worker failure.
    _trace(h, "c5", trigger="HumanMessage", action="speak", notes="reactive_optional")
    _trace(h, "c6", trigger="HumanMessage", action="silence", notes="reactive_optional")
    _trace(h, "c7", trigger="HumanMessage", action="speak", notes="mandatory_response")
    _trace(h, "c8", trigger="HumanMessage", action="silence", notes="worker_failure")

    m = h.stores.work.trace_metrics()
    assert m["total"] == 8
    assert m["proactive_total"] == 4
    assert m["proactive_dispatched"] == 3
    assert m["proactive_spoke"] == 1
    assert m["proactive_blocked"] == 1
    assert m["reactive_total"] == 2 and m["reactive_spoke"] == 1
    assert m["mandatory_total"] == 1 and m["mandatory_spoke"] == 1
    assert m["worker_failures"] == 1
    assert m["spoke"] == 3 and m["silent"] == 5
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
