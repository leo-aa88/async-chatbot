# CLAUDE.md

The canonical engineering guide for this repository is **[AGENTS.md](AGENTS.md)**. Read it
first — it covers the architecture, the non-negotiable invariants, and the coding standards.
The behavioral specification is [`docs/DESIGN.md`](docs/DESIGN.md) (v0.6, design-frozen).

## Claude-specific notes

- **Prefer editing small, single-responsibility modules.** No source file may exceed 600 LOC;
  split by responsibility before approaching that limit.
- **The reducer is the only state writer.** If a change makes an LLM/embedding worker or a
  delivery adapter mutate durable state, it is wrong by construction — route it through a
  result event and the reducer instead.
- **Determinism is a feature.** Cognition and the reducer receive an injected `Clock` and
  `Rng`. Do not call `time.time()`, `datetime.now()`, or the `random` module directly in
  those layers — tests rely on `ManualClock` and a seeded `Rng`. **One sanctioned exception:**
  `ids.new_id()` reads the wall clock and `os.urandom` to mint opaque durable keys. IDs are
  never inputs to a cognition decision, so this does not affect replay of behavior; tests that
  assert on durable identity inject their own IDs. Do not add any other direct clock/RNG reads.
- **Every invariant in AGENTS.md has a test.** When you touch cognition, persistence, the
  reducer, delivery, or lifecycle/recovery, run `pytest` and, when relevant, extend
  `tests/adversarial/`.
- When in doubt about intended behavior, `docs/DESIGN.md` is authoritative; quote the section
  number in your reasoning rather than guessing.

## Quick commands

```bash
pip install -e ".[dev]"
pytest
pytest tests/adversarial -q
```
