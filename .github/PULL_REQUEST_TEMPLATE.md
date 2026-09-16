<!--
Thanks for contributing to ACA! Please read CONTRIBUTING.md first.
Keep the change focused, and make sure the invariants in AGENTS.md still hold.
-->

## Summary

<!-- What does this PR do, and why? Link any related issue: "Closes #123". -->

## Changes

<!-- Bullet the notable changes. -->
-

## How was this tested?

<!-- Commands run, scenarios covered, new/updated tests. -->
- [ ] `pytest` passes locally
- [ ] `ruff check src tests` is clean
- [ ] Added/updated tests for the change (including `tests/adversarial/` if an invariant is affected)

## Invariant checklist

<!-- The non-negotiable invariants live in AGENTS.md. Confirm the ones your change touches. -->
- [ ] The reducer remains the only writer of durable state
- [ ] LLM/worker output is still validated and clamped before use (never trusted as authority)
- [ ] No new source file exceeds 600 LOC
- [ ] Cognition/reducer code reads time/randomness only through the injected `Clock`/`Rng`
- [ ] Not applicable / no invariants affected

## Notes for reviewers

<!-- Anything non-obvious, trade-offs, follow-ups, or areas you'd like extra eyes on. -->
