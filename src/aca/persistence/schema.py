"""SQLite schema DDL (DESIGN 23).

One authoritative schema string, applied idempotently (``IF NOT EXISTS``). Timestamps are
RFC 3339 / UTC text (DESIGN 10.3). JSON blobs hold opaque payloads/snapshots. ``UNIQUE``
constraints enforce the idempotency invariants: ingress by ``event_id`` (invariant 14) and
delivery by ``delivery_key`` (invariant 38).
"""

from __future__ import annotations

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_identity (
    agent_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    current_runtime_session_id TEXT,
    last_started_at TEXT,
    last_active_at TEXT,
    last_clean_suspend_at TEXT,
    last_resume_at TEXT,
    last_heartbeat_at TEXT,
    total_active_seconds INTEGER NOT NULL DEFAULT 0,
    last_runtime_exit_kind TEXT
);

CREATE TABLE IF NOT EXISTS runtime_sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    scheduler_generation INTEGER NOT NULL,
    closed_at TEXT,
    exit_kind TEXT
);

-- Durable inbox / event log. event_id UNIQUE gives at-least-once ingress, once reduction.
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    payload TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    reduced_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_unreduced ON events (reduced_at, accepted_at);

-- Single-row agent state: monotonic revision + serialized self/conversation models.
CREATE TABLE IF NOT EXISTS agent_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_revision INTEGER NOT NULL,
    scheduler_generation INTEGER NOT NULL DEFAULT 0,
    self_model TEXT NOT NULL,
    conversation_state TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provisional_memories (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    text TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '[]',
    activation REAL NOT NULL,
    salience REAL NOT NULL,
    decay_rate_per_hour REAL NOT NULL,
    embedding_id TEXT,
    enrichment_status TEXT NOT NULL,
    enrichment_attempts INTEGER NOT NULL DEFAULT 0,
    next_enrichment_after TEXT,
    last_enrichment_error TEXT,
    created_at TEXT NOT NULL,
    last_activated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    provisional_memory_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    activation REAL NOT NULL,
    importance REAL NOT NULL,
    unfinished INTEGER NOT NULL DEFAULT 0,
    decay_rate_per_hour REAL NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL,
    last_activated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deferred_intents (
    id TEXT PRIMARY KEY,
    topic_id TEXT,
    provisional_memory_id TEXT,
    intent TEXT NOT NULL,
    activation REAL NOT NULL,
    decay_rate_per_hour REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_activated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS response_obligations (
    id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    work_id TEXT,
    satisfied_by_message_id TEXT,
    superseded_by_id TEXT,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS work_items (
    work_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    basis_revision INTEGER NOT NULL,
    source_event_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    lease_until TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    result_event_id TEXT,
    last_error TEXT,
    snapshot TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_work_status ON work_items (status, lease_until);

CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    cycle_id TEXT,
    source_event_id TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outbound_messages (
    message_id TEXT PRIMARY KEY,
    delivery_key TEXT NOT NULL UNIQUE,
    action_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    channel TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    delivered_at TEXT,
    superseded_by_id TEXT,
    last_delivery_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbound_status ON outbound_messages (status, kind);

CREATE TABLE IF NOT EXISTS delivery_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    delivery_key TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    delivered INTEGER NOT NULL,
    error TEXT
);

-- Durable conversation record. Agent turns appear only once DELIVERED (DESIGN 23.7).
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,               -- 'human' | 'agent'
    text TEXT NOT NULL,
    channel TEXT NOT NULL,
    event_id TEXT,
    message_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_created ON conversation_turns (created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_action_id TEXT,
    classification TEXT,
    weight REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

-- Proactive budget usage keyed by (window_kind, window_id). Missing rows => no banked credit.
CREATE TABLE IF NOT EXISTS budget_usage (
    window_kind TEXT NOT NULL,       -- 'llm_day' | 'llm_hour' | 'msg_hour'
    window_id TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (window_kind, window_id)
);

CREATE TABLE IF NOT EXISTS cognition_traces (
    cycle_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_event_id TEXT,
    basis_revision INTEGER,
    commit_revision INTEGER,
    trigger TEXT,
    conversation_mode TEXT,
    candidate_kind TEXT,
    candidate_id TEXT,
    candidate_was_null INTEGER NOT NULL DEFAULT 0,
    llm_called INTEGER NOT NULL DEFAULT 0,
    action TEXT,
    useful_enrichment INTEGER NOT NULL DEFAULT 0,
    pre_outbox_invalidated INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    rng_seed_fragment TEXT,
    prompt_hash TEXT
);
"""
