"""Composed proactive gate evaluation, shared by pre-outbox and delivery-time revalidation.

Every proactive SPEAK is revalidated against *current* committed state before an outbound item
is created, and every delayed proactive item is revalidated again before transport (DESIGN
22.2, 23.7, invariant 9). Centralizing the composition keeps both checkpoints identical (DRY).
"""

from __future__ import annotations

from datetime import datetime

from ..cognition import budgets as budget
from ..cognition.gating import (
    ProactiveHardGateInput,
    evaluate_proactive_hard_gates,
    quiet_hours_active,
)
from ..cognition.gating import HardGateResult
from .context import ReducerContext
from .support import infer_mode, read_proactive_usage


def _cooldown_clear(ctx: ReducerContext, now: datetime) -> bool:
    """Cooldown is anchored to the last *delivered* proactive message (DESIGN 6.1, 11.4)."""
    last = ctx.stores.state.load_self_model().last_delivered_proactive_at
    if last is None:
        return True
    elapsed = (now - last).total_seconds()
    return elapsed >= ctx.config.timing.proactive_cooldown_seconds


def evaluate_proactive(
    ctx: ReducerContext, now: datetime, *, superseded: bool = False
) -> HardGateResult:
    """Evaluate all proactive hard gates against current state (DESIGN 11.1)."""
    usage = read_proactive_usage(ctx, now)
    message_budget = budget.proactive_message_allowed(usage, ctx.config.budgets)
    conversation = ctx.stores.state.load_conversation()
    now_local = now.astimezone(ctx.clock.local_tz())
    gate = ProactiveHardGateInput(
        capability_allowed=True,  # v0: proactive speech capability always granted
        budget_available=message_budget.allowed,
        cooldown_clear=_cooldown_clear(ctx, now),
        mode=infer_mode(conversation, now),
        quiet_active=quiet_hours_active(now_local, ctx.config.timing.quiet_hours),
        superseded=superseded,
    )
    return evaluate_proactive_hard_gates(gate)
