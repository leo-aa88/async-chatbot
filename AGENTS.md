# AGENTS.md — Working on the Asynchronous Conversational Agent (ACA)

This file is the canonical guide for any coding agent (human or AI) working in this
repository. `CLAUDE.md` defers to this document.

## What this project is

ACA is a **persistent, introspective conversational agent** whose central contract is:

> **Human input is optional. Agent output is optional.**

A conversational turn is a *bounded cognition cycle* that may or may not have been
triggered by a human and may or may not produce outward language. The full behavioral
specification lives in [`docs/DESIGN.md`](docs/DESIGN.md) (v0.6, design-frozen). **Read it
before making architectural changes.** This file summarizes what you need to build safely.

## The non-negotiable invariants

These are hard rules enforced *outside* the LLM. They are the reason the architecture
exists. Every change must preserve them; the adversarial tests in `tests/adversarial/`
exist to catch violations.

1. **Serialized mutation.** Only the reducer commits state and advances `state_revision`.
   Workers (LLM/embedding) consume immutable snapshots and return *result events* only.
   Delivery adapters perform only reducer-authorized transport and never mutate state.
2. **LLM output is data, never authority.** Worker results are typed proposals that are
   schema-parsed, whitelisted, identity-checked, revision-checked, clamped, and
   revalidated against current state before anything is committed.
3. **No unbounded recursive cognition.** One source cycle requests *at most one*
   generative LLM call. A result never re-triggers generative cognition inline.
4. **Stale/superseded proactive results are dropped or deferred, never regenerated inline.**
5. **Durable ingress.** A client `HumanMessage` is ACKed only after durable commit;
   retries reuse the same `event_id`; accepted-but-unreduced events survive restart and
   reduce effectively once.
6. **Deciding to speak is not delivery.** An undelivered outbound item is auditable
   intent, not a user-visible utterance. `delivered_at` anchors interruption cooldown.
7. **Outbound delivery is idempotency-aware and time-sensitive.** `delivery_key` suppresses
   duplicates; proactive items have TTL + coalescing and never dogpile on reconnect.
8. **Budgets never bank across downtime.** Resume resolves the *current* window only.
9. **Process uptime is not agent identity; agent downtime is not user absence.** A restart
   is a new runtime session, not a new logical agent. Downtime is cognition-free and never
   counts as observed user silence. Missed stochastic wakes are never replayed.
10. **Clock semantics are explicit.** Monotonic time drives in-process durations; UTC drives
    durable chronology/windows; configured local time drives quiet-hours/time-of-day.
11. **Stale wake events are harmless.** A wake must match current lifecycle state,
    `runtime_session_id`, and `scheduler_generation` before cognition starts.
12. **Tasks are never intentionally silenced.** The classifier is recall-biased for tasks;
    ambiguous task-like input routes to response-required. Proactive gates never silence an
    explicit user task.
13. **Quiet hours** are a hard gate on *unsolicited* outward speech only. They never suppress
    internal cognition or reactive responses.

## Architecture map (`src/aca/`)

```
errors, ids, clock, rng, durations, config   # foundations (pure, injectable)
domain/     enums, events, state, proposals   # data contracts, no behavior/IO
cognition/  activation, classifier, selection, scheduler, gating, budgets, snapshot
            # pure decision logic; deterministic given (state, clock, rng)
persistence/ schema, db, *_store             # SQLite; the only IO for durable state
reducer/    context, reducer, handlers/*      # THE single writer; dispatch by event type
workers/    base, fake_llm, fake_embedding    # side-effect-isolated; return events only
service/    lock, timer, dispatcher, recovery, delivery, service  # runtime orchestration
ipc/        protocol, server, client          # unix-socket line-framed JSON
cli/        main                              # `aca service|status|chat|logs|memories|topics`
```

Data flows one direction: `events → reducer → commit → (optional work item) → worker →
result event → reducer`. Cognition modules are pure and take an injected clock + rng so they
replay deterministically.

## Coding standards

- **Python 3.12+, asyncio.** The reducer is one coroutine consuming a queue. Heavy/CPU-bound
  work never runs inline on the event loop.
- **SOLID / DRY / KISS / clean code.** Prefer small, single-responsibility modules with
  explicit dependencies passed in (constructor/parameter injection). No global mutable state,
  no hidden wall-clock reads inside cognition, no service locator.
- **No source file exceeds 600 LOC.** Split by responsibility before you approach the limit.
- **Determinism.** Cognition takes an injected `Clock` and `Rng`. Never call
  `time.time()`/`random` directly in cognition or the reducer — use the injected instances.
- **Typed data contracts.** `domain/` holds frozen dataclasses with no IO and no behavior
  beyond validation. The reducer owns all persistence.
- Type-hint everything; keep functions focused; match the surrounding style.

## Commands

```bash
python3.12 -m venv .venv && . .venv/bin/activate   # once
pip install -e ".[dev]"                            # install with dev deps
pytest                                             # run all tests
pytest tests/adversarial -q                        # run invariant/adversarial tests
aca service start   # start the daemon (single-instance flock per data dir)
aca chat            # interactive client over the unix socket
aca status          # lifecycle/session/state snapshot
```

Data directory defaults to `~/.aca/` (override with `--data-dir` or `ACA_DATA_DIR`).

## When you change things

- Adding an event type → add to `domain/events.py`, register a handler in `reducer/handlers/`,
  and add reducer + adversarial tests.
- Adding an LLM proposal type → whitelist it in `domain/proposals.py` with bounds/clamps and
  cover it in `tests/test_reducer_llm.py`.
- Touching timing/decay/budgets → verify with a `ManualClock`; never introduce a real
  `sleep`/wall-clock read into the decision path.
