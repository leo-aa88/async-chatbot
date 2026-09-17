"""Unit tests for human-readable timestamp rendering (DESIGN 28.2)."""

from __future__ import annotations

from datetime import UTC, datetime

from aca.timefmt import clock_time, human_time


def test_human_time_renders_utc_iso_in_local_zone():
    iso = "2026-09-17T12:30:05+00:00"
    # America/Sao_Paulo is UTC-3 → 09:30:05 local.
    assert human_time(iso, "America/Sao_Paulo").startswith("2026-09-17 09:30:05")
    assert human_time(iso, "UTC") == "2026-09-17 12:30:05 UTC"


def test_human_time_accepts_datetime_and_naive_is_treated_as_utc():
    aware = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)
    assert human_time(aware, "UTC") == "2026-09-17 12:00:00 UTC"
    naive = datetime(2026, 9, 17, 12, 0, 0)
    assert human_time(naive, "UTC") == "2026-09-17 12:00:00 UTC"


def test_clock_time_is_compact_local_time():
    iso = "2026-09-17T12:30:05+00:00"
    assert clock_time(iso, "UTC") == "12:30:05"
    assert clock_time(iso, "America/Sao_Paulo") == "09:30:05"


def test_missing_or_unparseable_values_render_as_placeholder():
    assert human_time(None) == "—"
    assert human_time("") == "—"
    assert human_time("not-a-date") == "—"
    assert clock_time(None) == "—"


def test_unknown_timezone_falls_back_to_utc():
    iso = "2026-09-17T12:30:05+00:00"
    assert human_time(iso, "Not/AZone") == "2026-09-17 12:30:05 UTC"
