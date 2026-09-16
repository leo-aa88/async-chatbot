"""Pure cognition logic: activation, classification, selection, scheduling, gating, budgets.

Every function here is deterministic given its inputs plus an injected ``Clock`` and ``Rng``.
Nothing in this package performs IO or reads the wall clock directly — that is what makes the
stochastic core replayable (DESIGN 26.1) and unit-testable with a ``ManualClock``.
"""
