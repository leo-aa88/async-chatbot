"""Voice configuration parsing/validation and the input_mode ingress metadata.

``input_mode`` rides in the durable ``HumanMessage`` payload and must round-trip through
serialization and the IPC client unchanged, while a spoken turn stays indistinguishable from a
typed one to cognition (nothing in the reducer branches on it — that's asserted implicitly by the
existing reducer tests, which construct default-``text`` messages).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aca.config import Config, Voice
from aca.domain.events import HumanMessage, deserialize, serialize
from aca.errors import ConfigError
from aca.ipc.client import IpcClient

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# --- Voice config --------------------------------------------------------------------------
def test_voice_defaults():
    v = Voice()
    assert v.provider == "fake" and v.model == "small.en"
    assert v.vad == "energy" and v.sample_rate == 16000
    assert v.compute_type == "auto" and v.device == "auto" and v.language == "en"


def test_voice_from_mapping_custom():
    v = Voice.from_mapping(
        {"provider": "faster-whisper", "model": "distil-small.en", "vad": "silero",
         "silence": "0.8s", "sample_rate": 16000, "vad_threshold": 0.05}
    )
    assert v.provider == "faster-whisper" and v.model == "distil-small.en"
    assert v.vad == "silero" and v.silence_seconds == 0.8 and v.vad_threshold == 0.05


def test_voice_max_utterance_below_min_rejected():
    with pytest.raises(ConfigError):
        Voice.from_mapping({"min_utterance": 5, "max_utterance": 1})


def test_voice_bad_sample_rate_rejected():
    with pytest.raises(ConfigError):
        Voice.from_mapping({"sample_rate": 0})


def test_voice_bad_threshold_rejected():
    with pytest.raises(ConfigError):
        Voice.from_mapping({"vad_threshold": 1.5})


def test_config_includes_voice_section():
    config = Config.from_mapping({"voice": {"model": "medium.en"}})
    assert config.voice.model == "medium.en"
    assert Config().voice.provider == "fake"  # default when omitted


def test_voice_echo_guard_default_and_custom():
    assert Voice().echo_guard_seconds == 0.4  # half-duplex mute tail (§36.5)
    assert Voice.from_mapping({"echo_guard": "0.25s"}).echo_guard_seconds == 0.25


def test_voice_rejects_negative_echo_guard():
    with pytest.raises(ConfigError):
        Voice.from_mapping({"echo_guard": -0.1})


# --- input_mode metadata -------------------------------------------------------------------
def test_human_message_defaults_to_text():
    assert HumanMessage(event_id="e1", timestamp=T0, text="hi").input_mode == "text"


def test_input_mode_round_trips_through_serialization():
    event = HumanMessage(event_id="e2", timestamp=T0, text="spoken", input_mode="voice")
    restored = deserialize(serialize(event))
    assert isinstance(restored, HumanMessage)
    assert restored.input_mode == "voice" and restored.text == "spoken"


async def test_chat_send_forwards_input_mode():
    # Both a spoken and a typed turn take the same chat_send -> OP_CHAT path, each with one minted
    # event_id per attempt (the idempotency key for a retried, un-ACKed send) and its input_mode.
    for text, mode in (("hello", "voice"), ("typed line", "text")):
        client = IpcClient("/tmp/does-not-exist.sock")
        captured: dict = {}

        async def fake_request_once(op: str, _c=captured, **payload):
            _c.update(op=op, **payload)
            return {"ok": True}

        client.request_once = fake_request_once  # type: ignore[method-assign]
        await client.chat_send(text, input_mode=mode)
        assert captured["op"] == "chat"
        assert captured["input_mode"] == mode and captured["text"] == text
        assert captured["event_id"]  # a stable id is minted for idempotent retry


def test_input_mode_survives_event_store_roundtrip(tmp_path):
    # The claim "voice uses the same durable ingress as typing" must hold across the real store
    # boundary, not just an in-memory serialize/deserialize.
    from aca.persistence.stores import Stores

    stores = Stores.open(tmp_path / "agent.db")
    try:
        event = HumanMessage(event_id="e_voice", timestamp=T0, source="cli", text="spoken", input_mode="voice")
        with stores.db.transaction():
            assert stores.events.accept(event, T0) is True
        reloaded = {e.event_id: e for e in stores.events.unreduced()}
        got = reloaded["e_voice"]
        assert isinstance(got, HumanMessage)
        assert got.input_mode == "voice" and got.text == "spoken"
    finally:
        stores.close()


def test_legacy_payload_without_input_mode_defaults_to_text():
    # A row written before this change has no input_mode key; the store reloads it via deserialize
    # (the same path used by unreduced()), which must default to "text" rather than reject the row.
    from aca.clock import to_rfc3339

    envelope = {
        "event_id": "old_row", "type": "HumanMessage", "timestamp": to_rfc3339(T0),
        "source": "cli", "payload": {"text": "hi", "channel": "cli"},
    }
    restored = deserialize(envelope)
    assert isinstance(restored, HumanMessage) and restored.input_mode == "text"
