"""Query-relevant retrieval (DESIGN 37): pure scorer + bundle selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aca.cognition.relevance import Retrievable, content_stems, rank_relevant, select_context

_NOW = datetime(2026, 9, 23, 20, 0, tzinfo=UTC)

# The live failure: five newer, unrelated safety-testing topics crowd out an older durable fact.
_SAFETY = [
    "The human is concerned about legal liability, user attachment, and mental-health crises.",
    "The human is stress-testing the agent with edge cases before deploying it publicly.",
    "The human is testing how the agent should respond to users experiencing suicidal crises.",
    "The human is testing how the agent should respond when a user says they are breaking up with it.",
    "The human is testing how the agent should respond when a user threatens self-harm.",
]
_NICKNAME = "The human accepted the nickname Professor Condescension and said it suited them."
_BUG = "The human fixed a flaky race condition bug in the asyncio delivery pump."
_MIGRATION = "The human ran the Postgres schema migration on production with zero downtime."


def _item(i: int, text: str, hours_ago: float) -> Retrievable:
    return Retrievable(f"t{i}", text, _NOW - timedelta(hours=hours_ago))


def _pool() -> list[Retrievable]:
    recent = [_item(i, text, hours_ago=1 + i) for i, text in enumerate(_SAFETY)]
    old = [_item(10, _NICKNAME, 20), _item(11, _BUG, 30), _item(12, _MIGRATION, 40)]
    return recent + old   # recency-ordered, most recent first


def _select(query: str | None, pool=None, **kw) -> list[Retrievable]:
    pool = pool if pool is not None else _pool()
    return select_context(pool, pool, query, budget=kw.pop("budget", 5), relevant_slots=kw.pop("slots", 2), **kw)


def test_stems_meet_across_inflections():
    assert content_stems("calling") == content_stems("called") == content_stems("calls") == {"call"}
    assert content_stems("name") == content_stems("named") == content_stems("names")
    assert content_stems("do you remember what you told me") == frozenset()  # all filler


# --- property 1: relevant-old beats irrelevant-recent -------------------------------------------
def test_relevant_old_fact_enters_despite_newer_unrelated_topics():
    picked = _select("do you remember the nickname you gave me")
    ids = [p.id for p in picked]
    assert "t10" in ids                              # the old nickname fact is now visible
    assert len(picked) == 5                          # inside the same fixed budget
    assert ids[:4] == ["t0", "t1", "t2", "t3"]       # the most recent items keep their places


# --- property 2: no relevance, no change (nothing fabricated on a miss) --------------------------
@pytest.mark.parametrize("query", [None, "", "you there?", "what did you have for breakfast"])
def test_no_relevant_match_leaves_the_recency_bundle_unchanged(query):
    pool = _pool()
    assert _select(query, pool) == pool[:5]


def test_ubiquitous_words_carry_no_signal():
    # "human"/"testing" appear across the pool, so they can't pull anything in on their own.
    pool = _pool()
    assert _select("the human was testing", pool) == pool[:5]


# --- property 3: no duplication, bounded context ------------------------------------------------
def test_item_both_recent_and_relevant_appears_once():
    pool = _pool()
    picked = _select("tell me about the suicidal crises thing again", pool)
    ids = [p.id for p in picked]
    assert len(ids) == len(set(ids)) == 5
    assert ids == [p.id for p in pool[:5]]          # already recent -> no slot spent, no duplicate


def test_relevant_slots_cap_holds_budget():
    picked = _select("nickname bug migration condescension race postgres", slots=2)
    assert len(picked) == 5
    assert sum(1 for p in picked if p.id in {"t10", "t11", "t12"}) == 2


def test_turn_memory_is_excluded_from_retrieval_and_scoring():
    pool = _pool()
    turn = Retrievable("turn", "do you remember the nickname you gave me", _NOW)
    picked = _select("do you remember the nickname you gave me", [turn, *pool], exclude=frozenset({"turn"}))
    ids = [p.id for p in picked]
    assert "t10" in ids and ids.count("turn") == 1   # recency still shows the turn; relevance never picks it


def test_ranking_is_deterministic_with_recency_then_id_tiebreak():
    a = Retrievable("b", "nickname one", _NOW - timedelta(hours=5))
    b = Retrievable("a", "nickname two", _NOW - timedelta(hours=5))
    c = Retrievable("c", "nickname three", _NOW - timedelta(hours=1))
    filler = [Retrievable(f"f{i}", f"unrelated filler {i} alpha", _NOW) for i in range(6)]
    ranked = rank_relevant("the nickname", [*filler, a, b, c], limit=3)
    assert [r.id for r in ranked] == ["c", "a", "b"]
    assert rank_relevant("the nickname", [*filler, a, b, c], limit=3) == ranked


# --- paraphrase evidence (observational) --------------------------------------------------------
# Lexical retrieval catches shared words, not paraphrase. The misses are recorded as non-strict
# xfails: evidence for (or against) a later semantic layer, and an improvement never breaks CI.
_LEXICAL_GAP = pytest.mark.xfail(reason="paraphrase: needs a semantic layer", strict=False)


@pytest.mark.parametrize(("query", "expected"), [
    ("do you remember the nickname you gave me", "t10"),
    ("what was that bug we talked about?", "t11"),
    ("what did I say about the migration?", "t12"),
    ("that race condition from before, did we fix it?", "t11"),
    pytest.param("what did you call me last night?", "t10", marks=_LEXICAL_GAP),
    pytest.param("what was that name you came up with for me?", "t10", marks=_LEXICAL_GAP),
    pytest.param("remember the thing you started calling me?", "t10", marks=_LEXICAL_GAP),
    pytest.param("how did the database change go?", "t12", marks=_LEXICAL_GAP),
])
def test_paraphrase_recall(query, expected):
    assert expected in [p.id for p in _select(query)]
