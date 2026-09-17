"""Unit tests for foundations: durations, clock, rng, config."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from aca.clock import ManualClock, from_rfc3339, to_rfc3339
from aca.config import Config
from aca.durations import parse_hours, parse_seconds
from aca.errors import ConfigError, DurationParseError
from aca.rng import Rng


def test_parse_duration_units():
    assert parse_seconds("45m") == 45 * 60
    assert parse_seconds("2h") == 7200
    assert parse_seconds("1h30m") == 5400
    assert parse_seconds(90) == 90.0
    assert parse_hours("90m") == pytest.approx(1.5)


def test_parse_duration_rejects_garbage():
    with pytest.raises(DurationParseError):
        parse_seconds("soon")
    with pytest.raises(DurationParseError):
        parse_seconds("5x")


def test_rfc3339_roundtrip():
    moment = datetime(2026, 6, 1, 12, 30, tzinfo=UTC)
    assert from_rfc3339(to_rfc3339(moment)) == moment


def test_manual_clock_separates_axes():
    clk = ManualClock(datetime(2026, 1, 1, tzinfo=UTC))
    m0, w0 = clk.monotonic(), clk.now_utc()
    clk.advance_wall_only(3600)  # host suspend: wall moves, monotonic frozen
    assert clk.monotonic() == m0
    assert (clk.now_utc() - w0).total_seconds() == 3600


def test_rng_is_deterministic_and_seed_reproduces():
    a = Rng(42)
    b = Rng(42)
    assert [a.uniform() for _ in range(5)] == [b.uniform() for _ in range(5)]


def test_rng_exponential_zero_rate_is_infinite():
    assert Rng(1).exponential(0.0) == math.inf


def test_config_rejects_out_of_range_fraction():
    with pytest.raises(ConfigError):
        Config.from_mapping({"temperament": {"initiative": 2.0}})


def test_config_defaults_match_design():
    cfg = Config()
    assert cfg.cognition.spontaneous_activation_rate_per_hour == 0.25
    assert cfg.budgets.proactive_llm_calls_per_day == 12
    assert cfg.conversation.active_within_seconds == 180
    assert cfg.conversation.idle_within_seconds == 1800
    assert cfg.memory.repeat_suppression_seconds == 6 * 3600
    assert cfg.memory.enrichment_salience_floor == 0.6


def test_repeat_suppression_is_configurable():
    cfg = Config.from_mapping({"memory": {"repeat_suppression": "5s"}})
    assert cfg.memory.repeat_suppression_seconds == 5


def test_enrichment_salience_floor_is_configurable_and_bounded():
    cfg = Config.from_mapping({"memory": {"enrichment_salience_floor": 0.8}})
    assert cfg.memory.enrichment_salience_floor == 0.8
    import pytest

    from aca.errors import ConfigError

    with pytest.raises(ConfigError):
        Config.from_mapping({"memory": {"enrichment_salience_floor": 1.5}})


def test_conversation_windows_are_configurable():
    cfg = Config.from_mapping({"conversation": {"active_within": "10s", "idle_within": "5m"}})
    assert cfg.conversation.active_within_seconds == 10
    assert cfg.conversation.idle_within_seconds == 300


def test_conversation_idle_must_not_precede_active():
    with pytest.raises(ConfigError):
        Config.from_mapping({"conversation": {"active_within": "5m", "idle_within": "1m"}})
