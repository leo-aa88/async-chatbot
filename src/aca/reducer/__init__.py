"""The serialized reducer: the single writer of agent state (DESIGN 8.1, invariant 2).

Only code in this package commits state transitions and advances ``state_revision``. Handlers
are dispatched by event type; each runs inside one atomic transaction and produces a
``ReduceResult`` describing side effects (work to dispatch, whether to resample the wake timer,
whether new outbound is deliverable) for the service to perform *outside* the write path.
"""
