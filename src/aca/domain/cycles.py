"""Cycle-type tags carried in a work snapshot's ``source.cycle_type`` (DESIGN 14).

These label how a generative cycle should be finalized and what register its language should be
in. They are part of the reducer<->worker snapshot contract, so they live in the neutral domain
layer: the reducer writes them, the LLM result handler branches on them, and a worker may read
them to choose an appropriate voice (a reactive reply to a just-sent message must not sound like
resurfacing an old thought).
"""

from __future__ import annotations

CYCLE_MANDATORY = "mandatory"          # obligated reply (DESIGN 14.1)
CYCLE_REACTIVE_OPTIONAL = "reactive"   # chosen optional reply to a human turn (DESIGN 14.2)
CYCLE_PROACTIVE = "proactive"          # unsolicited initiative from a wake (DESIGN 14.3)
