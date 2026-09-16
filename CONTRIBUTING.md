# Contributing to ACA

Thanks for your interest in the Asynchronous Conversational Agent! This guide covers how to set
up, the standards we hold code to, and how to get a change merged.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- Read [`AGENTS.md`](AGENTS.md) — it summarizes the architecture and the **non-negotiable
  invariants**. Every change must preserve them.
- The behavioral specification is [`docs/DESIGN.md`](docs/DESIGN.md) (v0.6, **design-frozen**).
  Structural changes require evidence from the running prototype or a concrete embodiment
  requirement; when in doubt, open an issue to discuss first.

## Development setup

Requires Python 3.12+.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Run the checks CI runs:

```bash
pytest                 # full suite, including tests/adversarial/
ruff check src tests   # lint
python -m build        # (optional) sdist + wheel, matches the CI build job
```

## Coding standards

- **SOLID / DRY / KISS / clean code.** Small, single-responsibility modules with explicit
  dependencies passed in — no hidden globals, no service locator.
- **No source file exceeds 600 LOC.** Split by responsibility before you get close.
- **Determinism.** Cognition and the reducer take an injected `Clock` and `Rng`. Never call
  `time.time()`, `datetime.now()`, or the `random` module directly in those layers — tests rely
  on `ManualClock` and a seeded `Rng`.
- **The reducer is the only writer of durable state.** Workers return result events; delivery
  adapters only transport. If a change makes a worker or adapter mutate state, it's wrong by
  construction — route it through a result event and the reducer.
- **LLM output is data, never authority.** New model proposals must be whitelisted, bounded, and
  clamped in `domain/proposals.py`, and covered by tests.
- Type-hint everything and match the surrounding style. `ruff` enforces formatting-adjacent rules.

## Tests

- Add or update tests for every behavioral change. The harness in `tests/conftest.py` drives the
  synchronous reducer with a `ManualClock` and seeded `Rng`.
- If your change touches an invariant, add or extend a test in `tests/adversarial/` (one file
  per invariant).
- Keep the suite green and deterministic — no reliance on wall-clock timing or network.

## Making a change

1. Fork and create a feature branch (`feat/...`, `fix/...`, `docs/...`).
2. Make focused commits with clear messages (imperative mood: "Add ...", "Fix ...").
3. Ensure `pytest` and `ruff check src tests` pass.
4. Open a pull request (not a draft when it's ready for review) and fill in the PR template,
   including the invariant checklist.
5. A maintainer will review. Please be responsive to feedback; small, reviewable PRs merge faster.

## Reporting bugs and requesting features

Use the issue templates (Bug report / Feature request). Search existing issues first, and include
enough detail to reproduce.

## License

By contributing, you agree that your contributions are licensed under the repository's
[LICENSE](LICENSE).
