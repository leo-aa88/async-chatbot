"""Proactive-cognition budget accounting (DESIGN 25, invariant 27).

Budgets are hard limits on *autonomous/proactive* cognition. They **never bank across
downtime**: usage is keyed by wall-clock window identity (UTC hour / UTC day), so a window
that elapsed while the service was offline simply has no usage row and grants exactly its
configured capacity — never more (DESIGN 25).

These are pure functions over *current-window usage counts* (which the store supplies) plus
config. A direct user task follows the ordinary reactive budget and is never blocked here
(DESIGN 25, invariant 11).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import Budgets


def day_window_id(now_utc: datetime) -> str:
    """Stable identity for the current UTC-calendar-day budget window."""
    return now_utc.astimezone(now_utc.tzinfo).strftime("%Y-%m-%d")


def hour_window_id(now_utc: datetime) -> str:
    """Stable identity for the current UTC-hour budget window."""
    return now_utc.strftime("%Y-%m-%dT%H")


@dataclass(frozen=True, slots=True)
class ProactiveUsage:
    """Current-window usage counts, as read from durable storage."""

    llm_calls_today: int = 0
    llm_calls_this_hour: int = 0
    messages_this_hour: int = 0


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str = "ok"


def proactive_llm_call_allowed(usage: ProactiveUsage, budgets: Budgets) -> BudgetDecision:
    """Whether another proactive generative-LLM call fits the current windows (DESIGN 25)."""
    if usage.llm_calls_today >= budgets.proactive_llm_calls_per_day:
        return BudgetDecision(False, "daily_llm_budget_exhausted")
    if usage.llm_calls_this_hour >= budgets.proactive_llm_calls_per_hour:
        return BudgetDecision(False, "hourly_llm_budget_exhausted")
    return BudgetDecision(True)


def proactive_message_allowed(usage: ProactiveUsage, budgets: Budgets) -> BudgetDecision:
    """Whether another proactive *message* fits the current hourly window (DESIGN 25)."""
    if usage.messages_this_hour >= budgets.proactive_messages_per_hour:
        return BudgetDecision(False, "hourly_message_budget_exhausted")
    return BudgetDecision(True)
