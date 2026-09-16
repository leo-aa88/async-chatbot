# async-chatbot — Asynchronous Conversational Agent (ACA)

A persistent, introspective conversational agent whose central contract is:

> **Human input is optional. Agent output is optional.**

A conversational turn is a *bounded cognition cycle* that may or may not have been triggered by
a human and may or may not produce outward language. The agent can stay silent, think and answer
later, initiate on its own, remember without replying, and survive process restarts with a
durable identity — without keeping a generative LLM running while idle.

The full behavioral specification is in [`docs/DESIGN.md`](docs/DESIGN.md) (v0.6). The
engineering guide and the non-negotiable invariants are in [`AGENTS.md`](AGENTS.md).

## Status

v0 reference implementation (Python 3.12+, `asyncio`, SQLite). The full cognition pipeline runs
with deterministic **fake** LLM/embedding workers behind clean interfaces — no network, fully
replayable — plus a resident daemon with Unix-socket IPC, a single-instance lock, and crash
recovery. A real Claude-backed worker and a local embedding model slot in behind the worker
protocols without touching cognition semantics.

## Architecture

```
events → reducer (single writer) → commit → work item → worker → result event → reducer
```

```
src/aca/
  clock, rng, durations, config, ids, errors   # injectable foundations (deterministic)
  domain/      enums, events, state, runtime, proposals   # pure data contracts
  cognition/   activation, classifier, selection, scheduler, gating, budgets, snapshot
  persistence/ schema, db, *_store             # SQLite; the only durable-state IO
  reducer/     reducer, handlers/*, support    # THE single writer
  workers/     base, fake_llm, fake_embedding  # side-effect-isolated; return events only
  service/     lock, timer, dispatcher, recovery, delivery, service   # the daemon
  ipc/         protocol, server, client        # unix-socket line-framed JSON
  cli/         main                            # the `aca` command
```

## Quickstart

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 83 tests, incl. tests/adversarial (invariant tests)
```

Run the daemon and talk to it:

```bash
aca service start           # resident daemon (single-instance flock per data dir)
aca status                  # lifecycle / session / state snapshot
aca chat                    # interactive chat over the socket
aca logs                    # recent cognition traces
aca memories                # recent provisional memories
aca topics                  # enriched topics
aca service stop
```

The data directory defaults to `~/.aca/` (override with `--data-dir` or `ACA_DATA_DIR`).

## Design principles preserved by the code

- A thought is not a message.
- Persistence and cheap semantic addressability are eager; expensive interpretation is lazy.
- The model interprets; code decides what the interpretation is allowed to do.
- Process uptime is not agent identity; agent downtime is not user absence.
- Deciding to speak is not the same as being heard.
- Time may pass without cognition.
