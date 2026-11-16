# AGENTS.md — Working on the Asynchronous Conversational Agent (ACA)

This file is the canonical guide for any coding agent (human or AI) working in this
repository. `CLAUDE.md` defers to this document.

## What this project is

ACA is a **persistent, introspective conversational agent** whose central contract is:

> **Human input is optional. Agent output is optional.**

A conversational turn is a *bounded cognition cycle* that may or may not have been
triggered by a human and may or may not produce outward language. The full behavioral
specification lives in [`docs/DESIGN.md`](docs/DESIGN.md) (v0.8: the frozen v0.6 core plus the
§34–§38 extensions). **Read it before making architectural changes.** This file summarizes what
you need to build safely.

## The charter, and the review question

What the agent is *for* is specified in [`docs/DESIGN.md` §38](docs/DESIGN.md). The human-readable
form is six lines:

```text
Know what is true.
Seek what remains unknown.
Make what is good more possible.
Respect the freedom of other minds.
Build rather than dominate.
Accept correction.
```

None of these lines is given to the model. They explain why the machinery is shaped the way it
is: ends act through activation and soft factors, side-constraints act as hard gates (never
weights), and continuity, memory, and uptime are instrumental with no weight of their own. Ask of
every meaningful change:

> **Does this make the creed more causally true of the system, or only more talked about?**

A change whose only effect is that the agent *says* more about truth, curiosity, or autonomy
fails this test. A change to a gate, decay rate, or salience rule that alters trajectories passes
it, and must be shown to with a counterfactual test (same history, mechanism off, different
outcome). Charter mechanisms are asymmetric: **discourse-relation labels are suppress-only.**
`CLOSE`/`ORPHAN`/`REPEAT` may reduce future expression or persistence; `ADVANCE`/`EVIDENCE`/`REVISE`
may never increase it. (This is narrower than "the model never promotes its own influence": the
whitelisted memory proposals can, by design. See DESIGN §38.6.) The charter's four invariants
(DESIGN §30, 43–46):

- The telos can justify silence, never force speech. (A design constraint: the prompt slice's
  effect on the speak rate is not yet verified, §38.5.)
- An unfinished thought keeps salience, not entitlement.
- The human's absence is never a grievance.
- Disagreement may affect communication, never authority.

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
14. **The discourse-focus gate is proactive-only, expression-only, and `IDLE`-scoped.** It can mute
    an off-subject proactive thought while the conversation is `IDLE`; it never removes a thought
    from candidacy, never blocks a reply, and is inactive when `DORMANT` so autonomous resurfacing
    survives (DESIGN §34, §30 41).
15. **The LLM discourse relation is policy-gated.** The advancement label is suppress-only
    (`ORPHAN`/`REPEAT` mute; it never forces a speak the deterministic gates block, and stays out of
    the observational metrics). The focus transition is a validated write: it can only point the
    focus at an existing provisional memory, or `KEEP`/`CLEAR` (DESIGN §35, §30 42).

DESIGN §30 lists all 46 invariants; this list is the subset you're most likely to break.

## Architecture map (`src/aca/`)

```
errors, ids, clock, rng, durations, config, dotenv, timefmt   # foundations (pure, injectable)
domain/      enums, events, state, proposals, cycles, mood, runtime  # data contracts, no IO
cognition/   activation, classifier, selection, scheduler, gating, budgets, snapshot,
             relevance, textsim, vectors
             # pure decision logic; deterministic given (state, clock, rng)
persistence/ schema, db, stores, *_store     # SQLite; the only IO for durable state
reducer/     context, reducer, handlers/*    # THE single writer; dispatch by event type
             gates, continuity, discourse, revalidation   # proactive gates (§11, §16, §34, §35)
             workitems, conversation_notes, apply_proposals, outbound, support
workers/     base, fake_llm, fake_embedding, embeddings, embedding_factory
             # side-effect-isolated; return events only
workers/llm/ adapters, factory, prompt, worker   # real providers + the system prompt (§21, §22)
eval/        backbone, continuity, counterfactual, gating
             # OFFLINE analysis only; never imported by the runtime, never writes state
voice/       audio, capture, vad, silero, segmenter, turn_assembler, transcriber, whisper,
             session, factory                # CLIENT-SIDE speech->text perception (§36)
service/     lock, timer, dispatcher, recovery, delivery, service  # runtime orchestration
ipc/         protocol, server, client        # unix-socket line-framed JSON
tts/         base, factory, kokoro_engine, controller, text  # optional client-side speech output
cli/         main, chat_ui   # `aca service start|stop`, `aca status|chat [--voice]|logs|memories|
                             #  topics|metrics|reconcile-embeddings`
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
  The single sanctioned exception is `ids.new_id()`, which mints opaque durable keys from the
  wall clock + `os.urandom`; IDs are never decision inputs, so replay of behavior is unaffected
  (tests that assert on durable identity inject their own IDs).
- **Typed data contracts.** `domain/` holds frozen dataclasses with no IO and no behavior
  beyond validation. The reducer owns all persistence.
- Type-hint everything; keep functions focused; match the surrounding style.

## Commands

```bash
make install          # create .venv and install with dev deps (or: python3.12 -m venv .venv,
                      # then .venv/bin/pip install -e ".[dev]"; extras: llm, voice, tts)
make check            # ruff + full pytest: exactly what CI gates on
make test-adversarial # invariant/adversarial tests only
make lint             # ruff check src tests
make format           # ruff auto-fix + format
aca service start     # start the daemon (single-instance flock per data dir)
aca chat              # interactive client over the unix socket (--voice adds the mic)
aca status            # lifecycle/session/state snapshot
```

`make help` lists the rest (`run`, `chat`, `demo`, `backbone-eval`, which costs API calls). Use the
project venv, not a system `pip`: an old system pip can't do the editable (PEP 660) install, and
extras installed outside the venv aren't seen by `aca`.

Data directory defaults to `~/.aca/` (override with `--data-dir` or `ACA_DATA_DIR`).

## When you change things

- Adding an event type → add to `domain/events.py`, register a handler in `reducer/handlers/`,
  and add reducer + adversarial tests.
- Adding an LLM proposal type → whitelist it in `domain/proposals.py` with bounds/clamps, cover it
  in `tests/test_proposals.py` and `tests/adversarial/test_llm_authority.py`, and update the pinned
  whitelist in `tests/adversarial/test_charter.py`. That pin is deliberate: a new proposal type is a
  charter change (invariant 46, DESIGN §38.6), so review it against the charter.
- Touching timing/decay/budgets → verify with a `ManualClock`; never introduce a real
  `sleep`/wall-clock read into the decision path.
