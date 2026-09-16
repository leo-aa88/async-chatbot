"""Local IPC between clients and the agent service (DESIGN 4.10, 28.2).

Clients are not the agent: closing a client never stops the logical agent (invariant 36). The
transport is a newline-delimited JSON protocol over a Unix-domain socket — an adapter, not part
of cognition semantics.
"""
