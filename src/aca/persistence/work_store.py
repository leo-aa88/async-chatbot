"""Work-item, obligation, action, and trace persistence (DESIGN 23.5, 26).

Work ownership is durable even though workers are ephemeral (DESIGN 23.5): a work item is
persisted before dispatch, leased while RUNNING, and reclaimable on crash. Result handling is
idempotent by ``work_id`` / ``result_event_id`` (invariant 24).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ..domain.enums import ObligationStatus, WorkKind, WorkStatus
from ..domain.runtime import Action, CognitionTrace, ResponseObligation, WorkItem
from .db import Database
from .mapping import as_bool, dt, dumps, loads, txt


class WorkStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- work items ----------------------------------------------------------------------
    def insert_work(self, w: WorkItem) -> None:
        self._db.execute(
            """INSERT INTO work_items (
                work_id, kind, cycle_id, basis_revision, source_event_id, status,
                attempt_count, lease_until, created_at, completed_at, result_event_id,
                last_error, snapshot
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                w.work_id, w.kind.value, w.cycle_id, w.basis_revision, w.source_event_id,
                w.status.value, w.attempt_count, txt(w.lease_until), txt(w.created_at),
                txt(w.completed_at), w.result_event_id, w.last_error, dumps(w.snapshot),
            ),
        )

    def get_work(self, work_id: str) -> WorkItem | None:
        row = self._db.query_one("SELECT * FROM work_items WHERE work_id=?", (work_id,))
        return None if row is None else self._to_work(row)

    def generative_exists_for_cycle(self, cycle_id: str) -> bool:
        """Whether a generative (LLM) work item already exists for this cycle (invariant 3)."""
        row = self._db.query_one(
            "SELECT 1 FROM work_items WHERE cycle_id=? AND kind IN (?, ?) LIMIT 1",
            (cycle_id, WorkKind.LLM_COGNITION.value, WorkKind.LLM_ENRICHMENT.value),
        )
        return row is not None

    def lease(self, work_id: str, lease_until: datetime) -> None:
        self._db.execute(
            """UPDATE work_items SET status=?, attempt_count=attempt_count+1, lease_until=?
            WHERE work_id=?""",
            (WorkStatus.RUNNING.value, txt(lease_until), work_id),
        )

    def complete(self, work_id: str, result_event_id: str, completed_at: datetime) -> None:
        self._db.execute(
            """UPDATE work_items SET status=?, result_event_id=?, completed_at=?, lease_until=NULL
            WHERE work_id=?""",
            (WorkStatus.COMPLETED.value, result_event_id, txt(completed_at), work_id),
        )

    def fail_terminal(self, work_id: str, error: str) -> None:
        self._db.execute(
            "UPDATE work_items SET status=?, last_error=? WHERE work_id=?",
            (WorkStatus.FAILED_TERMINAL.value, error, work_id),
        )

    def requeue(self, work_id: str, error: str | None = None) -> None:
        self._db.execute(
            "UPDATE work_items SET status=?, lease_until=NULL, last_error=? WHERE work_id=?",
            (WorkStatus.PENDING.value, error, work_id),
        )

    def pending(self) -> list[WorkItem]:
        rows = self._db.query_all(
            "SELECT * FROM work_items WHERE status=? ORDER BY created_at ASC",
            (WorkStatus.PENDING.value,),
        )
        return [self._to_work(r) for r in rows]

    def active_embedding_topic_ids(self) -> set[str]:
        """Topic ids of unfinished topic-summary embedding work (PENDING or RUNNING).

        For backfill dedup: a leased (RUNNING) job is still in flight, so ``pending()`` alone would
        miss it and let a restart or on-demand reconcile enqueue a duplicate. FAILED_TERMINAL is
        excluded so a permanently failed job stays eligible for a fresh attempt; COMPLETED is
        irrelevant (the topic then has an embedding and is no longer a backfill candidate).
        """
        rows = self._db.query_all(
            "SELECT snapshot FROM work_items WHERE kind=? AND status IN (?, ?)",
            (WorkKind.EMBEDDING.value, WorkStatus.PENDING.value, WorkStatus.RUNNING.value),
        )
        ids: set[str] = set()
        for r in rows:
            topic_id = loads(r["snapshot"], {}).get("topic_id")
            if topic_id:
                ids.add(topic_id)
        return ids

    def expired_running(self, now: datetime) -> list[WorkItem]:
        rows = self._db.query_all(
            "SELECT * FROM work_items WHERE status=? AND lease_until IS NOT NULL "
            "AND lease_until < ?",
            (WorkStatus.RUNNING.value, txt(now)),
        )
        return [self._to_work(r) for r in rows]

    # --- obligations ---------------------------------------------------------------------
    def insert_obligation(self, o: ResponseObligation) -> None:
        self._db.execute(
            """INSERT INTO response_obligations (
                id, source_event_id, created_at, status, work_id, satisfied_by_message_id,
                superseded_by_id, last_error
            ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                o.id, o.source_event_id, txt(o.created_at), o.status.value, o.work_id,
                o.satisfied_by_message_id, o.superseded_by_id, o.last_error,
            ),
        )

    def update_obligation(self, o: ResponseObligation) -> None:
        self._db.execute(
            """UPDATE response_obligations SET status=?, work_id=?, satisfied_by_message_id=?,
                superseded_by_id=?, last_error=? WHERE id=?""",
            (
                o.status.value, o.work_id, o.satisfied_by_message_id, o.superseded_by_id,
                o.last_error, o.id,
            ),
        )

    def get_obligation(self, obligation_id: str) -> ResponseObligation | None:
        row = self._db.query_one(
            "SELECT * FROM response_obligations WHERE id=?", (obligation_id,)
        )
        return None if row is None else self._to_obligation(row)

    def pending_obligations(self) -> list[ResponseObligation]:
        rows = self._db.query_all(
            "SELECT * FROM response_obligations WHERE status=? ORDER BY created_at ASC",
            (ObligationStatus.PENDING.value,),
        )
        return [self._to_obligation(r) for r in rows]

    def obligation_by_work(self, work_id: str) -> ResponseObligation | None:
        row = self._db.query_one(
            "SELECT * FROM response_obligations WHERE work_id=?", (work_id,)
        )
        return None if row is None else self._to_obligation(row)

    # --- actions & traces ----------------------------------------------------------------
    def insert_action(self, a: Action) -> None:
        self._db.execute(
            """INSERT INTO actions (id, kind, cycle_id, source_event_id, detail, created_at)
            VALUES (?,?,?,?,?,?)""",
            (a.id, a.kind.value, a.cycle_id, a.source_event_id, a.detail, txt(a.created_at)),
        )

    def insert_trace(self, t: CognitionTrace) -> None:
        self._db.execute(
            """INSERT OR REPLACE INTO cognition_traces (
                cycle_id, created_at, source_event_id, basis_revision, commit_revision,
                trigger, cycle_type, conversation_mode, candidate_kind, candidate_id,
                candidate_was_null, llm_called, action, useful_enrichment, pre_outbox_invalidated,
                notes, rng_seed_fragment, prompt_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                t.cycle_id, txt(t.created_at), t.source_event_id, t.basis_revision,
                t.commit_revision, t.trigger, t.cycle_type, t.conversation_mode, t.candidate_kind,
                t.candidate_id, int(t.candidate_was_null), int(t.llm_called), t.action,
                int(t.useful_enrichment), int(t.pre_outbox_invalidated), t.notes,
                t.rng_seed_fragment, t.prompt_hash,
            ),
        )

    def finalize_trace(
        self,
        cycle_id: str,
        *,
        action: str | None,
        useful_enrichment: bool,
        pre_outbox_invalidated: bool,
        note: str | None,
    ) -> None:
        """Update an existing cycle trace with its result (no-op if the cycle has no trace)."""
        self._db.execute(
            """UPDATE cognition_traces SET action=?, useful_enrichment=?,
                pre_outbox_invalidated=?, notes=COALESCE(?, notes) WHERE cycle_id=?""",
            (action, int(useful_enrichment), int(pre_outbox_invalidated), note, cycle_id),
        )

    def recent_traces(self, limit: int = 20) -> list[CognitionTrace]:
        rows = self._db.query_all(
            "SELECT * FROM cognition_traces ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [self._to_trace(r) for r in rows]

    def trace_metrics(self, *, repeated_since: datetime | None = None) -> dict[str, int]:
        """Mechanical cognition counts from the trace log (no model calls) — see DESIGN 26.

        A proactive cycle is a StochasticWake; a reactive/mandatory cycle is a HumanMessage, told
        apart by its ``notes`` tag. ``blocked`` proactive cycles never reached the model (budget,
        mode, quiet hours). These raw counts feed the derived rates the CLI prints.
        """
        # Classify by cycle_type — set once at dispatch and never overwritten — so a FAILED mandatory
        # or superseded reactive cycle still counts toward its total (finalize_trace rewrites the
        # free-text ``notes``, so matching on notes would silently drop failures from the ratio).
        # Proactive is derived from the always-set, never-overwritten ``trigger`` (so it covers
        # rows from before the cycle_type column existed). Reactive/mandatory need cycle_type, which
        # older rows lack — those human-triggered rows are surfaced as ``unclassified`` rather than
        # silently dropped, so proactive+reactive+mandatory+unclassified == total reconciles.
        row = self._db.query_one(
            """SELECT
                COUNT(*) AS total,
                COALESCE(SUM(trigger='StochasticWake'), 0) AS proactive_total,
                COALESCE(SUM(trigger='StochasticWake' AND llm_called=1), 0) AS proactive_dispatched,
                COALESCE(SUM(trigger='StochasticWake' AND action='speak'), 0) AS proactive_spoke,
                COALESCE(SUM(trigger='StochasticWake' AND llm_called=0), 0) AS proactive_blocked,
                COALESCE(SUM(cycle_type='reactive'), 0) AS reactive_total,
                COALESCE(SUM(cycle_type='reactive' AND action='speak'), 0) AS reactive_spoke,
                COALESCE(SUM(cycle_type='mandatory'), 0) AS mandatory_total,
                COALESCE(SUM(cycle_type='mandatory' AND action='speak'), 0) AS mandatory_spoke,
                COALESCE(SUM(trigger='HumanMessage' AND cycle_type IS NULL), 0) AS unclassified,
                COALESCE(SUM(action='speak'), 0) AS spoke,
                COALESCE(SUM(action='silence'), 0) AS silent,
                COALESCE(SUM(action='failed'), 0) AS failed,
                COALESCE(SUM(notes LIKE '%worker_failure%'), 0) AS worker_failures,
                COALESCE(SUM(notes LIKE '%enrichment_gated%'), 0) AS enrichment_gated
            FROM cognition_traces""",
        )
        metrics = {} if row is None else {k: int(row[k] or 0) for k in row.keys()}
        # Repeated-candidate rate: how often a proactive SPEAK re-voiced a candidate_id it had
        # already spoken before, WITHIN A RECENT WINDOW (``repeated_since``). Exact about candidate
        # identity, not semantics; a proxy for topic repetition. The window matters: without it,
        # legitimate long-gap resurfacing of a persistent topic (a design goal) would inflate the
        # count purely with uptime. Windowed, it means "nagging lately" — re-voicing the same thing
        # repeatedly in the recent period, not appropriately revisiting it much later (DESIGN 6.1,
        # 11.3, 13.5, 16). Unbounded (repeated_since=None) is a lifetime re-voicing count.
        params: tuple[str, ...] = ()
        clause = ""
        if repeated_since is not None:
            # Format the cutoff with the same helper used to store created_at (RFC-3339 'Z'), so the
            # lexicographic string comparison is valid.
            clause, params = " AND created_at >= ?", (txt(repeated_since),)
        rep = self._db.query_one(
            "SELECT COUNT(*) AS spoke_recent, "
            "COUNT(candidate_id) AS spoke_with_candidate, "  # COUNT(col) ignores NULLs
            "COUNT(DISTINCT candidate_id) AS distinct_candidates "
            "FROM cognition_traces "
            "WHERE trigger='StochasticWake' AND action='speak'" + clause,
            params,
        )
        spoke_recent = int(rep["spoke_recent"] or 0) if rep else 0
        spoke_c = int(rep["spoke_with_candidate"] or 0) if rep else 0
        distinct_c = int(rep["distinct_candidates"] or 0) if rep else 0
        # Proactive burst rate: pure count of spontaneous speeches in the window — deliberately
        # blunt (frequency, not value or theme). Wolfy's own point stands: high frequency can be
        # noise, so read this alongside initiation quality, not as a goal to maximize.
        metrics["proactive_spoke_recent"] = spoke_recent
        metrics["proactive_spoke_with_candidate"] = spoke_c
        metrics["proactive_repeated"] = spoke_c - distinct_c
        return metrics

    # Ordered proactive block-reason buckets (DESIGN 26): the point in the cognition pipeline where a
    # proactive cycle that did NOT speak was stopped. Pipeline order — wake-time pre-filter, then the
    # post-dispatch pre-outbox re-check (§34.6, §35.4), then the model's own silence. A single CASE
    # assigns each *resolved* non-speaking StochasticWake row to exactly one bucket (no double
    # counting), so the buckets partition the resolved-non-speaking proactive cycles exactly
    # (``other`` catches the rest). ``model_silence`` is the dispatched cycle the model let lapse:
    # the reducer overwrites its trace note with ``None`` (COALESCE keeps the ``proactive_dispatch``
    # tag) and only flips the action to ``silence`` — so the tag survives, and within this query
    # (which already requires a resolved silence/failure) a ``proactive_dispatch`` row *is* a model
    # silence. In-flight cycles (action still NULL) are excluded, so a pending dispatch is not
    # miscounted as a silence.
    _BLOCK_REASON_ORDER = (
        "nothing", "low_worth", "budget", "cooldown", "quiet_hours", "mode_suppressed",
        "enrichment_gated", "discourse_orphan_wake", "continuity_repeat_wake",
        "discourse_orphan_recheck", "continuity_repeat_recheck", "advancement", "superseded",
        "model_silence", "enrichment_only", "failed", "other",
    )
    # A priority-ordered CASE assigning each row exactly one bucket. Earlier WHENs win, so the
    # specific post-dispatch tags (``pre_outbox:*``, ``advancement_*``, supersession) are matched
    # before the keyword groups that would also match via LIKE. Both the wake-time ``blocked:{reason}``
    # and the pre-outbox ``pre_outbox:{reason}`` prefixes carry the same hard-gate ``reason`` literals
    # (``budget_exhausted``/``cooldown_active``/``mode_suppresses_initiative``/``quiet_hours``), plus
    # ``llm_budget`` at wake, so the keyword matches deliberately span both prefixes.
    _BLOCK_REASON_CASE = """CASE
        WHEN notes LIKE '%advancement\\_%' ESCAPE '\\' THEN 'advancement'
        WHEN notes LIKE '%superseded%' OR notes LIKE '%newer_human%'
             OR notes LIKE '%candidate_resolved%' THEN 'superseded'
        WHEN notes LIKE 'pre_outbox%discourse_orphan%' THEN 'discourse_orphan_recheck'
        WHEN notes LIKE 'pre_outbox%continuity_repeat%' THEN 'continuity_repeat_recheck'
        WHEN notes = 'discourse_orphan' THEN 'discourse_orphan_wake'
        WHEN notes = 'continuity_repeat' THEN 'continuity_repeat_wake'
        WHEN notes LIKE 'enrichment_gated%' THEN 'enrichment_gated'
        WHEN notes LIKE '%budget%' THEN 'budget'
        WHEN notes LIKE '%cooldown%' THEN 'cooldown'
        WHEN notes LIKE '%quiet_hours%' THEN 'quiet_hours'
        WHEN notes LIKE '%mode_suppresses%' OR notes LIKE '%capability_denied%' THEN 'mode_suppressed'
        WHEN notes = 'low_worth' THEN 'low_worth'
        WHEN notes = 'nothing' THEN 'nothing'
        WHEN notes LIKE 'enrichment_only%' THEN 'enrichment_only'
        WHEN notes = 'proactive_dispatch' THEN 'model_silence'
        WHEN COALESCE(action, '') = 'failed' THEN 'failed'
        ELSE 'other'
    END"""

    def proactive_block_reasons(self, *, since: datetime | None = None) -> dict[str, int]:
        """Why did proactive cycles NOT speak? Counts of non-speaking ``StochasticWake`` cycles,
        bucketed by where in the pipeline each was stopped (observability, DESIGN 26).

        Read-only decomposition over the trace ``notes`` the reducer already records — no model
        calls, no state. Returns every bucket in pipeline order (zeros included) so the shape is
        stable across runs. Scope is *resolved* non-speaking proactive cycles — those that reached a
        ``silence``/``failed`` outcome — so an in-flight dispatch (action still NULL) is not counted;
        the values therefore sum to ``proactive_total - proactive_spoke - (proactive still pending)``.
        This is the foundation for the offline counterfactual gating eval: it says which gate is
        doing the suppressing before asking whether that suppression was *right*.
        """
        clause, params = "", ()
        if since is not None:
            clause, params = " AND created_at >= ?", (txt(since),)
        rows = self._db.query_all(
            f"SELECT {self._BLOCK_REASON_CASE} AS bucket, COUNT(*) AS n FROM cognition_traces "
            "WHERE trigger='StochasticWake' AND action IN ('silence', 'failed')" + clause
            + " GROUP BY bucket",
            params,
        )
        counts = {r["bucket"]: int(r["n"] or 0) for r in rows}
        return {bucket: counts.get(bucket, 0) for bucket in self._BLOCK_REASON_ORDER}

    def proactive_spoken_candidates(self, *, since: datetime | None = None) -> list[tuple[str, str]]:
        """``(candidate_kind, candidate_id)`` of proactive SPEAK cycles, oldest first.

        Feeds the semantic-dominance metric. Returns the kind too because a spoken candidate is
        usually a TOPIC or DEFERRED_INTENT, not a raw memory (build_candidates drops enriched
        memories), so the caller must resolve each kind to an embedding differently. Order is stable
        (created_at) so downstream clustering is deterministic.
        """
        clause, params = "", ()
        if since is not None:
            clause, params = " AND created_at >= ?", (txt(since),)
        rows = self._db.query_all(
            "SELECT candidate_kind, candidate_id FROM cognition_traces "
            "WHERE trigger='StochasticWake' AND action='speak' AND candidate_id IS NOT NULL"
            + clause + " ORDER BY created_at ASC",
            params,
        )
        return [(r["candidate_kind"], r["candidate_id"]) for r in rows]

    # --- mappers -------------------------------------------------------------------------
    @staticmethod
    def _to_work(row: sqlite3.Row) -> WorkItem:
        return WorkItem(
            work_id=row["work_id"], kind=WorkKind(row["kind"]), cycle_id=row["cycle_id"],
            basis_revision=row["basis_revision"], source_event_id=row["source_event_id"],
            status=WorkStatus(row["status"]), attempt_count=row["attempt_count"],
            lease_until=dt(row["lease_until"]), created_at=dt(row["created_at"]),
            completed_at=dt(row["completed_at"]), result_event_id=row["result_event_id"],
            last_error=row["last_error"], snapshot=loads(row["snapshot"], {}),
        )

    @staticmethod
    def _to_obligation(row: sqlite3.Row) -> ResponseObligation:
        return ResponseObligation(
            id=row["id"], source_event_id=row["source_event_id"],
            created_at=dt(row["created_at"]), status=ObligationStatus(row["status"]),
            work_id=row["work_id"], satisfied_by_message_id=row["satisfied_by_message_id"],
            superseded_by_id=row["superseded_by_id"], last_error=row["last_error"],
        )

    @staticmethod
    def _to_trace(row: sqlite3.Row) -> CognitionTrace:
        return CognitionTrace(
            cycle_id=row["cycle_id"], created_at=dt(row["created_at"]),
            source_event_id=row["source_event_id"], basis_revision=row["basis_revision"],
            commit_revision=row["commit_revision"], trigger=row["trigger"],
            cycle_type=row["cycle_type"] if "cycle_type" in row.keys() else None,
            conversation_mode=row["conversation_mode"], candidate_kind=row["candidate_kind"],
            candidate_id=row["candidate_id"], candidate_was_null=as_bool(row["candidate_was_null"]),
            llm_called=as_bool(row["llm_called"]), action=row["action"],
            useful_enrichment=as_bool(row["useful_enrichment"]),
            pre_outbox_invalidated=as_bool(row["pre_outbox_invalidated"]), notes=row["notes"],
            rng_seed_fragment=row["rng_seed_fragment"], prompt_hash=row["prompt_hash"],
        )
