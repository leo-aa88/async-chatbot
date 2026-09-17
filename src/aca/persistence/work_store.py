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

    def trace_metrics(self) -> dict[str, int]:
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
        # already spoken before (a distinct-count can't live in the aggregate above). This is exact
        # about candidate identity, not semantics — two different ids on the same subject don't
        # count, and the same candidate resurfacing weeks later does; it's a proxy for topic
        # repetition. High = nagging / circling the same thought; low = fresh (DESIGN 6.1, 11.3).
        rep = self._db.query_one(
            """SELECT COUNT(*) AS spoke_with_candidate,
                      COUNT(DISTINCT candidate_id) AS distinct_candidates
            FROM cognition_traces
            WHERE trigger='StochasticWake' AND action='speak' AND candidate_id IS NOT NULL""",
        )
        spoke_c = int(rep["spoke_with_candidate"] or 0) if rep else 0
        distinct_c = int(rep["distinct_candidates"] or 0) if rep else 0
        metrics["proactive_spoke_with_candidate"] = spoke_c
        metrics["proactive_repeated"] = spoke_c - distinct_c
        return metrics

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
