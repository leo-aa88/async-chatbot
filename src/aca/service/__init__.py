"""Runtime orchestration: single-instance lock, wake timer, worker dispatch, recovery, delivery.

This package turns the pure reducer into a resident daemon. It owns concurrency (asyncio),
timers (monotonic), the single-instance advisory lock, and the outbound delivery pump — but it
never writes durable state directly. All state changes still flow through the reducer as events
(invariant 2).
"""
