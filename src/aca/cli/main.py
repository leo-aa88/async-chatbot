"""``aca`` CLI (DESIGN 28.2).

Subcommands: ``service start|stop``, ``status``, ``chat`` (``--voice`` adds microphone input),
``logs``, ``memories``, ``topics``.
The CLI is a client — closing it never stops the agent (invariant 36). The daemon runs via
``service start``; every other command talks to it over the Unix socket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from ..domain.enums import ExitKind
from ..dotenv import load_dotenv
from ..errors import AcaError
from ..ipc.client import IpcClient
from ..ipc.server import IpcServer
from ..service.service import AgentService
from ..timefmt import clock_time, human_time
from ..workers.embedding_factory import build_embedding_worker, embedding_key_env_for
from ..workers.llm import build_llm_worker, key_env_for
from .chat_ui import ChatUI


def _data_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir or os.environ.get("ACA_DATA_DIR") or Path.home() / ".aca")


def _socket_path(data_dir: Path) -> Path:
    return data_dir / "aca.sock"


def _load_config(data_dir: Path) -> Config:
    config_file = data_dir / "config.json"
    if config_file.exists():
        return Config.from_mapping(json.loads(config_file.read_text()))
    return Config()


# --- service start -------------------------------------------------------------------------
async def _run_service(data_dir: Path) -> None:
    config = _load_config(data_dir)
    # Load only the configured providers' keys from .env (data dir first, then cwd); the shell
    # environment wins, and no unrelated secrets from a cwd .env are absorbed into the daemon.
    needed_keys = {k for k in (key_env_for(config.llm), embedding_key_env_for(config.embedding)) if k}
    if needed_keys:
        load_dotenv(data_dir / ".env", Path.cwd() / ".env", allow=needed_keys)
    llm_worker = build_llm_worker(config.llm)  # fail fast on a misconfigured provider
    embedding_worker = build_embedding_worker(config.embedding)  # fail fast likewise
    service = AgentService(data_dir, config, llm_worker=llm_worker, embedding_worker=embedding_worker)
    server = IpcServer(service, _socket_path(data_dir))
    await service.start()
    await server.start()
    print(
        f"aca service running ("
        f"{'name: ' + config.identity.name + ', ' if config.identity.name else ''}"
        f"data dir: {data_dir}, llm: {config.llm.provider}"
        f"{'/' + config.llm.model if config.llm.model else ''}"
        f", embedding: {config.embedding.provider}"
        f"{'/' + config.embedding.model if config.embedding.model else ''})",
        flush=True,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with_suppress = getattr(loop, "add_signal_handler", None)
        if with_suppress is not None:
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:  # pragma: no cover
                pass

    shutdown_watch = asyncio.ensure_future(server.serve_until_shutdown())
    signal_watch = asyncio.ensure_future(stop_event.wait())
    done, pending = await asyncio.wait(
        {shutdown_watch, signal_watch}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()

    await server.close()
    await service.stop(ExitKind.CLEAN_SUSPEND)
    print("aca service stopped", flush=True)


def _cmd_service(args: argparse.Namespace) -> int:
    data_dir = _data_dir(args)
    if args.service_command == "start":
        asyncio.run(_run_service(data_dir))
        return 0
    if args.service_command == "stop":
        return _cmd_simple(args, "shutdown")
    print("usage: aca service {start|stop}", file=sys.stderr)
    return 2


# --- client commands -----------------------------------------------------------------------
def _cmd_simple(args: argparse.Namespace, op: str) -> int:
    client = IpcClient(_socket_path(_data_dir(args)))

    async def run() -> int:
        method = getattr(client, op)
        response = await method()
        print(json.dumps(response, indent=2))
        return 0 if response.get("ok", True) else 1

    return asyncio.run(run())


def _start_voice(config, client: IpcClient, ui: ChatUI) -> tuple:
    """Build and start a concurrent voice session feeding the same ingress as typed input.

    Imported lazily and only when ``--voice`` is passed, so plain text chat never depends on the
    optional voice extra. Transcribed turns are echoed in the shared UI and injected as
    ``HumanMessage``s tagged ``input_mode="voice"`` — indistinguishable to cognition from typing.
    Returns ``(session, task)`` so the caller can tear it down.
    """
    from ..voice.capture import MicrophoneSource
    from ..voice.factory import build_segmenter, build_speech_detector, build_transcriber
    from ..voice.session import VoiceSession

    voice = config.voice
    tz = config.local_timezone
    transcriber = build_transcriber(voice)  # fail fast on a missing model/extra
    detector = build_speech_detector(voice)
    segmenter = build_segmenter(voice)
    source = MicrophoneSource(sample_rate=voice.sample_rate, frame_seconds=voice.frame_seconds)

    async def on_transcript(text: str) -> None:
        ui.print_message(text, at=clock_time(datetime.now(UTC), tz), sender="you/voice")
        await client.chat_send(text, input_mode="voice")

    session = VoiceSession(source, detector, segmenter, transcriber, on_transcript)
    return session, asyncio.ensure_future(session.run())


async def _chat(data_dir: Path, *, voice: bool = False) -> None:
    client = IpcClient(_socket_path(data_dir))
    config = _load_config(data_dir)
    tz = config.local_timezone
    # Stamp the human's own input line too, so a copied transcript shows who spoke when — the same
    # local HH:MM:SS used for agent messages (display-only; the deterministic core is untouched).
    ui = ChatUI(stamp=lambda: clock_time(datetime.now(UTC), tz))

    async def on_message(frame: dict) -> None:
        ui.print_message(frame.get("text", ""), at=clock_time(frame.get("at"), tz))

    subscription = asyncio.ensure_future(client.subscribe(on_message))
    # Voice, when enabled, runs alongside typing — both keystrokes and spoken turns feed the same
    # ingress. Text chat works with no voice dependencies installed; --voice is a pure add-on.
    voice_session = voice_task = None
    if voice:
        voice_session, voice_task = _start_voice(config, client, ui)

    banner = "Connected. Type a message and press enter (Ctrl-D to quit)."
    if voice:
        banner += f"  [voice on: {config.voice.provider}/{config.voice.model} — just speak]"
    print(banner)
    ui.start()
    try:
        # Keystrokes and agent-message prints are both handled on this event loop — a single
        # terminal writer, so async messages can't clobber the input line (no thread, no mutex).
        async for line in ui.lines():
            text = line.strip()
            if text:
                await client.chat_send(text)
    finally:
        ui.stop()
        subscription.cancel()
        if voice_session is not None:
            voice_session.stop()
        if voice_task is not None:
            voice_task.cancel()


def _cmd_chat(args: argparse.Namespace) -> int:
    asyncio.run(_chat(_data_dir(args), voice=getattr(args, "voice", False)))
    return 0


# --- human-readable list views ------------------------------------------------------------
def _fmt_trace(row: dict, tz: str) -> str:
    parts = [human_time(row.get("created_at"), tz), row.get("trigger", "?"),
             row.get("action") or "-", row.get("candidate_kind") or "-"]
    if row.get("llm_called"):
        parts.append("llm")
    if row.get("notes"):
        parts.append(f"({row['notes']})")
    return "  ".join(parts)


def _fmt_memory(row: dict, tz: str) -> str:
    text = (row.get("text") or "").replace("\n", " ")
    return (f"{human_time(row.get('created_at'), tz)}  [{row.get('enrichment_status', '?')}] "
            f"act={row.get('activation', 0):.2f}  {text}")


def _fmt_topic(row: dict, tz: str) -> str:
    flag = "unfinished" if row.get("unfinished") else "done"
    summary = (row.get("summary") or "").replace("\n", " ")
    return (f"{human_time(row.get('created_at'), tz)}  [{flag}] "
            f"act={row.get('activation', 0):.2f}  {summary}")


def _pct(n: int, d: int) -> str:
    return "—" if d == 0 else f"{100 * n / d:.0f}% ({n}/{d})"


def _hours(seconds: int) -> str:
    """Compact human window like '4h' or '6d' for the metrics label."""
    if seconds <= 0:
        return "all time"
    hours = seconds / 3600
    return f"{hours / 24:.0f}d" if hours >= 48 else f"{hours:.0f}h"


def format_metrics(m: dict) -> list[str]:
    """Render mechanical cognition metrics from raw trace counts (pure, for reuse/testing)."""
    spoke, silent = m.get("spoke", 0), m.get("silent", 0)
    return [
        f"cognition cycles: {m.get('total', 0)}",
        f"proactive initiation:  {_pct(m.get('proactive_spoke', 0), m.get('proactive_dispatched', 0))} reached-model"
        f"  |  {_pct(m.get('proactive_spoke', 0), m.get('proactive_total', 0))} all-cycle"
        "  (spoke / reached-model | spoke / all proactive cycles)",
        f"proactive blocked:     {m.get('proactive_blocked', 0)}  (budget/mode/quiet — never reached model)",
        f"proactive frequency (last {_hours(m.get('repeated_window_seconds', 0))}): "
        f"{m.get('proactive_spoke_recent', 0)} spontaneous messages  (blunt count; says nothing about value or theme)",
        f"repeated-candidate rate (last {_hours(m.get('repeated_window_seconds', 0))}): "
        f"{_pct(m.get('proactive_repeated', 0), m.get('proactive_spoke_with_candidate', 0))}"
        "  (proactive re-voicings of the same candidate; a proxy for topic repetition)",
        f"topic dominance (last {_hours(m.get('repeated_window_seconds', 0))}): "
        f"{_pct(m.get('dominance_cluster', 0), m.get('dominance_total', 0))}"
        "  (largest semantic cluster of self-voiced messages; observational)",
        f"advance rate (last {_hours(m.get('repeated_window_seconds', 0))}): "
        f"{_pct(m.get('advance_count', 0), m.get('advance_transitions', 0))}"
        f"  (same-topic new implication; repeats={m.get('repetition_count', 0)} switches={m.get('switch_count', 0)})",
        f"reactive reply rate:   {_pct(m.get('reactive_spoke', 0), m.get('reactive_total', 0))}",
        f"mandatory answered:    {_pct(m.get('mandatory_spoke', 0), m.get('mandatory_total', 0))}",
        f"overall silence rate:  {_pct(silent, spoke + silent)}",
        f"worker failures: {m.get('worker_failures', 0)}   enrichment gated: {m.get('enrichment_gated', 0)}",
    ] + (
        [f"unclassified:          {m['unclassified']}  (pre-upgrade cycles, no cycle_type)"]
        if m.get("unclassified", 0) else []
    )


def _cmd_metrics(args: argparse.Namespace) -> int:
    client = IpcClient(_socket_path(_data_dir(args)))

    async def run() -> int:
        response = await client.metrics()
        if not response.get("ok", True):
            print(json.dumps(response, indent=2))
            return 1
        for line in format_metrics(response.get("metrics", {})):
            print(line)
        return 0

    return asyncio.run(run())


def _cmd_list(args: argparse.Namespace, op: str, key: str, fmt) -> int:
    data_dir = _data_dir(args)
    tz = _load_config(data_dir).local_timezone
    client = IpcClient(_socket_path(data_dir))

    async def run() -> int:
        response = await getattr(client, op)()
        if not response.get("ok", True):
            print(json.dumps(response, indent=2))
            return 1
        rows = response.get(key, [])
        if not rows:
            print(f"(no {key})")
            return 0
        for row in rows:
            print(fmt(row, tz))
        return 0

    return asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aca", description="Asynchronous Conversational Agent")
    parser.add_argument("--data-dir", default=None, help="agent data directory (default ~/.aca)")
    sub = parser.add_subparsers(dest="command", required=True)

    service = sub.add_parser("service", help="manage the agent service")
    service.add_argument("service_command", choices=["start", "stop"])

    sub.add_parser("status", help="show agent status")
    chat = sub.add_parser("chat", help="interactive chat")
    chat.add_argument(
        "--voice", action="store_true",
        help="also listen on the microphone and transcribe speech (needs the 'voice' extra)",
    )
    sub.add_parser("logs", help="recent cognition traces")
    sub.add_parser("memories", help="recent provisional memories")
    sub.add_parser("topics", help="enriched topics")
    sub.add_parser("metrics", help="mechanical cognition metrics from the trace log")
    sub.add_parser(
        "reconcile-embeddings", help="backfill summary embeddings for topics missing one"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "service":
            return _cmd_service(args)
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "logs":
            return _cmd_list(args, "logs", "traces", _fmt_trace)
        if args.command == "memories":
            return _cmd_list(args, "memories", "memories", _fmt_memory)
        if args.command == "topics":
            return _cmd_list(args, "topics", "topics", _fmt_topic)
        if args.command == "metrics":
            return _cmd_metrics(args)
        if args.command == "reconcile-embeddings":
            return _cmd_simple(args, "reconcile_embeddings")
        return _cmd_simple(args, args.command)
    except AcaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
