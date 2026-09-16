"""Async semantic workers (LLM, embedding).

Workers are side-effect isolated (invariant 4): they consume an immutable snapshot and return
*data*, which the service wraps into a result event for the reducer. Workers never write
durable state, never contact the user, and never advance ``state_revision``.
"""
