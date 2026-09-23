"""Query-relevant retrieval for the reply context bundle (DESIGN 21, 37).

The context bundle used to retrieve only the *most recently activated* topics and memories, so an
older durable fact the human asks about directly ("do you remember the nickname you gave me?") could
be stored yet absent from the snapshot — storage succeeded, retrieval failed. This module adds a small,
bounded, relevance-selected slice on reply cycles: a few items ranked by lexical overlap with the
human's turn, deduplicated against the recency slice, inside the same fixed budget.

It is deliberately lexical and deterministic: the current turn's embedding is computed asynchronously
*after* the reply work item is created, so it isn't available at snapshot time, and a pure stdlib
scorer keeps snapshots replay-stable. Scoring is query-coverage over distinctive content stems:

* the query and each item are reduced to content stems (lowercase words, stopwords dropped, a light
  suffix strip so "calling"/"called"/"call" meet);
* a stem shared by at least half of the pool (e.g. "human" in "The human ...") carries no signal and
  is ignored — distinctiveness is judged against the pool, not a fixed list;
* an item's score is the number of distinctive query stems it contains (ties: more recent first, then
  id), and an item needs at least one to qualify at all.

Known limit, on purpose: this catches shared *words*, not paraphrase ("what did you call me?" does
not reach "the nickname Professor Condescension"). Those misses are recorded as evidence for a later
semantic layer rather than papered over with synonym lists.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

_WORD = re.compile(r"[a-z0-9]+")

# Function words plus conversational filler that says nothing about *which* fact is wanted.
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing done down during each else ever few for from further get
got had has have having he her here hers herself him himself his how i if in into is it its itself just
let like me more most my myself no nor not now of off on once only or other our ours out over own really
remember same say said she should so some such tell than that the their theirs them then there these they
thing things this those through to too under until up very was we were what when where which while who
whom why will with would you your yours yourself yeah okay ok hey well um uh oh know think mean want
please thanks thank sorry again back kind sort lot bit maybe still even actually basically
told tells telling give gave given giving came come comes going goes went gone started start
""".split())

_SUFFIXES = ("ingly", "edly", "ing", "ed", "ly", "s")


def _stem(word: str) -> str:
    """A light, deterministic suffix strip: "calling"/"called"/"calls" -> "call"; "name"/"named"/"names"
    -> "nam" (a trailing "e" is dropped last so the verb and noun forms meet)."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def content_stems(text: str) -> frozenset[str]:
    """Distinct content stems of ``text`` (stopwords and 1-2 letter tokens dropped)."""
    return frozenset(_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS)


@dataclass(frozen=True, slots=True)
class Retrievable:
    """One topic or memory as the retriever sees it: an id, its text, and when it was last active."""

    id: str
    text: str
    last_activated_at: datetime


def rank_relevant(query: str, pool: Sequence[Retrievable], *, limit: int,
                  exclude: frozenset[str] = frozenset()) -> list[Retrievable]:
    """Up to ``limit`` items of ``pool`` most relevant to ``query`` (none if nothing qualifies).

    Deterministic: the same query and pool always yield the same list in the same order.
    """
    pool = [item for item in pool if item.id not in exclude]  # e.g. the turn's own memory: never evidence
    if limit <= 0 or not pool:
        return []
    query_stems = content_stems(query)
    if not query_stems:
        return []
    item_stems = {item.id: content_stems(item.text) for item in pool}
    # A stem present in half the pool or more distinguishes nothing ("human", "agent", ...).
    ceiling = max(2, len(pool) // 2)
    df = {s: sum(1 for stems in item_stems.values() if s in stems) for s in query_stems}
    distinctive = {s for s in query_stems if 0 < df[s] < ceiling or (len(pool) < 4 and df[s] > 0)}
    if not distinctive:
        return []
    scored = [(len(distinctive & item_stems[item.id]), item) for item in pool if distinctive & item_stems[item.id]]
    scored.sort(key=lambda pair: (-pair[0], -pair[1].last_activated_at.timestamp(), pair[1].id))
    return [item for _, item in scored[:limit]]


def select_context(recent: Sequence[Retrievable], pool: Sequence[Retrievable], query: str | None, *,
                   budget: int, relevant_slots: int, exclude: frozenset[str] = frozenset()) -> list[Retrievable]:
    """The bundle's items: up to ``relevant_slots`` query-relevant ones, the rest filled by recency.

    ``recent`` is recency-ordered (most recent first). With no query, or nothing relevant beyond what
    recency already shows, this returns exactly ``recent[:budget]`` — unchanged behavior. An item that
    is both recent and relevant appears once; the total never exceeds ``budget``.
    """
    head = list(recent[:budget])
    if not query or relevant_slots <= 0:
        return head
    shown = {item.id for item in head}
    relevant = rank_relevant(query, pool, limit=relevant_slots + budget, exclude=exclude)
    extra = [item for item in relevant if item.id not in shown][:relevant_slots]
    if not extra:
        return head
    keep = head[: budget - len(extra)]
    return keep + extra
