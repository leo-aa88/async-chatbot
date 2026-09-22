# ACA — Asynchronous Conversational Agent

[![CI](https://github.com/leo-aa88/async-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/leo-aa88/async-chatbot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A persistent, introspective conversational agent whose central contract is:

> **Human input is optional. Agent output is optional.**

A conversational turn is not "one human message → one AI reply." It is a *bounded cognition
cycle* that may or may not have been triggered by a human and may or may not produce outward
language. The agent can stay silent, think and answer later, initiate on its own, remember
without replying, and survive process restarts with a durable identity — **without keeping a
generative LLM running while idle**.

The full behavioral specification is in [`docs/DESIGN.md`](docs/DESIGN.md) (v0.6, design-frozen).
The engineering guide and the non-negotiable invariants are in [`AGENTS.md`](AGENTS.md).

---

## Table of contents

- [What makes it different](#what-makes-it-different)
- [Status](#status)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Using the agent](#using-the-agent)
- [Testing](#testing)
- [Design principles](#design-principles-preserved-by-the-code)
- [Contributing](#contributing)
- [License](#license)

## What makes it different

| Traditional chatbot | ACA |
|---|---|
| One request → one response | A cognition cycle that may produce speech, deferred speech, state-only change, or silence |
| Silence means failure | Silence is a first-class action, distinct from failure/timeout/crash |
| Speaks only when spoken to | Can initiate autonomously via a stochastic activation hazard |
| Forgets between calls | Durable identity, memory, and temporal continuity across restarts |
| LLM is the clock | Cheap local activation/decay/gating; the LLM is used only when justified |
| Model output mutates state | Model output is *data*: validated, clamped, and applied by a single reducer |

## Status

**v0 reference implementation** — Python 3.12+, `asyncio`, SQLite.

The full cognition pipeline runs with deterministic **fake** LLM/embedding workers behind clean
interfaces (no network, fully replayable), plus a resident daemon with Unix-socket IPC, a
single-instance lock, and crash recovery. A real Claude-backed worker and a local embedding model
slot in behind the worker protocols without touching cognition semantics.

## Architecture

Data flows one direction — the reducer is the single writer of durable state:

```
events → reducer → commit → work item → worker → result event → reducer
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
  tts/         base, factory, kokoro_engine, controller   # optional client-side speech output
  cli/         main                            # the `aca` command
```

Cognition modules are pure and take an injected `Clock` + `Rng`, so the stochastic core replays
deterministically. Every source file is kept under 600 LOC.

## Quickstart

Requires Python 3.12+.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # full suite, including tests/adversarial/ (invariant tests)
```

Or use the [`Makefile`](Makefile) (`make help` lists all targets):

```bash
make install                # create .venv and install with dev deps
make check                  # lint + tests (what CI gates on)
make run                    # start the agent service
make demo                   # throwaway agent that messages you on its own, then chat
```

### Using a real LLM (provider-agnostic)

By default the agent uses a deterministic **fake** worker (canned replies — it exercises the
architecture offline). To use a real model, install the LLM extra and pick a provider in
`<data-dir>/config.json`:

```bash
pip install -e ".[llm]"      # adds httpx
```

```json
"llm": { "provider": "openai", "model": "gpt-4o-mini", "max_tokens": 512 }
```

Supported providers and the env var each reads for its key:

| `provider` | API | Key env var |
|---|---|---|
| `openai` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `anthropic` (alias `claude`) | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `grok` (alias `xai`) | xAI (OpenAI-compatible) | `XAI_API_KEY` |
| `gemini` (alias `google`) | Google (OpenAI-compatible) | `GEMINI_API_KEY` |
| `fake` (alias `mock`) | — (deterministic, offline) | none |

Provide the key in a **`.env` file** (git-ignored) rather than exporting it — the service reads
`<data-dir>/.env` (next to `config.json`, e.g. `~/.aca/.env`) and a `.env` in the current
directory at startup; a real shell export still wins over the file. See
[`examples/.env.example`](examples/.env.example):

```bash
echo 'OPENAI_API_KEY=sk-...' > ~/.aca/.env
aca service start
```

Model output is still validated, clamped, and applied by the reducer — a provider is never
trusted as authority. `base_url` / `api_key_env` can be overridden in config if an endpoint or
credential name differs. See [`examples/config.llm.json`](examples/config.llm.json).

### Using real embeddings (optional)

Embeddings default to a **local, free, deterministic** worker (`fake`). A real provider gives
memories genuine *semantic* vectors (needed for semantic dedup / repeated-topic / dominance):

```json
"embedding": { "provider": "openai", "model": "text-embedding-3-small" }
```

⚠️ **This is a different trade-off from the LLM provider choice.** By design the agent embeds
**nearly every substantive message** (DESIGN 12.2), and embeddings are *local from v0* on purpose
(DESIGN 28.4). Turning on a real embedding provider therefore sends most of what you type to a
third party — at a much higher frequency than the rate/budget-gated generative calls, with real
per-call cost and a new network failure surface. It's a fine opt-in (your data, your account, and
it unlocks the semantic features), but it is **not** free or private like the default. Supported:
`openai` (`OPENAI_API_KEY`), `gemini` (`GEMINI_API_KEY`); `base_url` overridable. Left at `fake`
unless set.

### Speaking the agent's messages aloud (optional)

The `aca chat` client can speak each delivered agent message using [Kokoro-82M](https://github.com/hexgrad/kokoro),
a small open-weights neural TTS model that runs **locally**. It's off by default (the agent is
silent). Install the extra plus two system libraries and opt in:

```bash
# system deps: PortAudio for playback, espeak-ng for Kokoro's out-of-dictionary G2P fallback
sudo apt install libportaudio2 espeak-ng        # Debian/Ubuntu (use your platform's equivalent)
pip install -e ".[tts]"                          # adds kokoro, sounddevice, numpy
```

```json
"tts": { "provider": "kokoro", "voice": "am_onyx" }
```

The default voice is `am_onyx` (an American male voice), or the persona's voice when
`identity.persona` is set (`tsundere` → `af_bella`); `lang_code` is derived from the voice's
first letter. This is **client-side rendering only** — DESIGN §6.1 anticipates exactly this ("a TTS
adapter … without changing cognition semantics"). The client still **prints every message verbatim**
(you always read what the agent says); TTS additionally synthesizes a *spoken normalization* of that
text — markup unwrapped so Kokoro doesn't read `*` or backticks aloud. It sits entirely outside
cognition and the reducer and never touches durable state (invariant 1); a synthesis failure is
shown once as a `[client]` status line and latches speech off rather than disrupting chat.

Kokoro runs on-device, but the **first** use downloads the model weights from Hugging Face (~330 MB);
after that it is fully offline and no audio or text leaves the host. See
[`examples/config.tts.json`](examples/config.tts.json).

### Persona (optional, experimental)

`identity.persona` selects the agent's user-facing *voice*. `default` is the plain, non-sycophantic
voice. `tsundere` is prickly, defensive, quick to tease, and easily flustered when her care is
noticed — warmer and more attached than she admits, and unexpectedly direct when something is
actually wrong. The prompt anchors on a well-known personality *structure* (an original character,
not a roleplay). She's never possessive or guilt-trippy, and a real question still gets a real answer:

```json
"identity": { "persona": "tsundere" }
```

A persona is wording only. It adds one section to the shared prompt (reactive, mandatory, and
proactive cycles alike) and never touches a gate, budget, cooldown, discourse focus, or cadence, so
the agent speaks no more often in character than without it. Machine-facing output (topic
summaries, intents, relation/focus labels) is told to stay neutral so durable memory isn't written
in character. The persona also picks the default TTS voice: `tsundere` → Kokoro `af_bella`
(`default` → `am_onyx`), and an explicit `tts.voice` overrides it. It needs a real LLM provider;
the offline `fake` worker ignores personas. See [`examples/config.tsundere.json`](examples/config.tsundere.json).
To score a provider against the persona eval corpus (behavioral shape, not wording), run
`python scripts/persona_eval.py`.

### Seeing autonomous (proactive) messages

By default the agent will not interrupt an `ACTIVE` conversation, and its spontaneous wake rate is
low — so autonomous messages are rare (that's the point: they should feel earned, not spammy). To
watch the behavior quickly, run `make demo`: it launches a throwaway agent tuned with a short
`conversation.active_within` and a modest wake rate, then drops you into chat. Say something, stop
typing for ~15 seconds, and it will message you on its own — no prompt, no reconnect.

> Tip: a very high `spontaneous_activation_rate_per_hour` (e.g. `3600`) will quickly exhaust the
> daily proactive budget (`budgets.proactive_llm_calls_per_day`) and then go quiet until the UTC
> day rolls over. The demo profile uses a sane rate; prefer ~`300` for hand-testing.

## Using the agent

Run the daemon and talk to it (the client is not the agent — closing it never stops the agent):

```bash
aca service start           # resident daemon (single-instance flock per data dir)
aca status                  # lifecycle / session / state snapshot
aca chat                    # interactive chat over the socket
aca chat --voice            # also listen on the mic (needs `pip install -e ".[voice]"`)
aca logs                    # recent cognition traces (why it spoke or stayed silent)
aca memories                # recent provisional memories
aca topics                  # enriched topics
aca service stop            # graceful suspension (identity persists)
```

The data directory defaults to `~/.aca/` (override with `--data-dir` or `ACA_DATA_DIR`). Optional
configuration lives in `<data-dir>/config.json` (see `docs/DESIGN.md` §18 for the parameters).

Voice input (`aca chat --voice`) transcribes speech on the client and injects it through the same
ingress as typing — spoken and typed turns are indistinguishable to cognition. It is entirely
optional and lives alongside typing; see [`docs/VOICE.md`](docs/VOICE.md).

## Testing

- **83 tests**, including [`tests/adversarial/`](tests/adversarial) — one file per non-negotiable
  invariant (idempotent ingress, no recursive cognition, stale-wake rejection,
  LLM-output-is-not-authority, tasks-never-silenced, quiet-hours hard gate, budgets-don't-bank,
  downtime-is-not-user-absence, delivery dedup, crash recovery, single-instance lock).
- Deterministic and offline: a `ManualClock` and seeded `Rng` drive the synchronous reducer;
  workers are mocked/faked (per `docs/DESIGN.md` §26.1).

```bash
pytest                 # everything
pytest tests/adversarial -q
ruff check src tests   # lint
```

## Design principles preserved by the code

- A thought is not a message.
- Persistence and cheap semantic addressability are eager; expensive interpretation is lazy.
- The model interprets; code decides what the interpretation is allowed to do.
- Process uptime is not agent identity; agent downtime is not user absence.
- Deciding to speak is not the same as being heard.
- Time may pass without cognition.

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The architecture invariants in [`AGENTS.md`](AGENTS.md)
must be preserved, and structural changes to the frozen v0.6 design need justification.

## License

Licensed under the [Apache License 2.0](LICENSE).
