"""Agent-state, budget-usage, and feedback persistence.

``agent_state`` is a single row holding the monotonic ``state_revision`` (advanced only by the
reducer, DESIGN 8.1), the current ``scheduler_generation``, and the serialized self/conversation
models. Budget usage is keyed by wall-clock window identity so it never banks across downtime
(DESIGN 25, invariant 27).
"""

from __future__ import annotations

from ..domain.enums import ConversationMode
from ..domain.state import ConversationState, SelfModel
from .db import Database
from .mapping import dt, txt


class StateStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- revision / models ---------------------------------------------------------------
    def initialize(self, self_model: SelfModel, conversation: ConversationState) -> None:
        self._db.execute(
            """INSERT OR IGNORE INTO agent_state (
                id, state_revision, scheduler_generation, self_model, conversation_state
            ) VALUES (1, 0, 0, ?, ?)""",
            (self._dump_self(self_model), self._dump_conversation(conversation)),
        )

    def state_revision(self) -> int:
        row = self._db.query_one("SELECT state_revision FROM agent_state WHERE id=1")
        return 0 if row is None else int(row["state_revision"])

    def advance_revision(self) -> int:
        self._db.execute("UPDATE agent_state SET state_revision = state_revision + 1 WHERE id=1")
        return self.state_revision()

    def scheduler_generation(self) -> int:
        row = self._db.query_one("SELECT scheduler_generation FROM agent_state WHERE id=1")
        return 0 if row is None else int(row["scheduler_generation"])

    def set_scheduler_generation(self, generation: int) -> None:
        self._db.execute(
            "UPDATE agent_state SET scheduler_generation=? WHERE id=1", (generation,)
        )

    def load_self_model(self) -> SelfModel:
        row = self._db.query_one("SELECT self_model FROM agent_state WHERE id=1")
        return self._load_self(row["self_model"])

    def save_self_model(self, model: SelfModel) -> None:
        self._db.execute(
            "UPDATE agent_state SET self_model=? WHERE id=1", (self._dump_self(model),)
        )

    def load_conversation(self) -> ConversationState:
        row = self._db.query_one("SELECT conversation_state FROM agent_state WHERE id=1")
        return self._load_conversation(row["conversation_state"])

    def save_conversation(self, conversation: ConversationState) -> None:
        self._db.execute(
            "UPDATE agent_state SET conversation_state=? WHERE id=1",
            (self._dump_conversation(conversation),),
        )

    # --- budgets -------------------------------------------------------------------------
    def budget_count(self, window_kind: str, window_id: str) -> int:
        row = self._db.query_one(
            "SELECT count FROM budget_usage WHERE window_kind=? AND window_id=?",
            (window_kind, window_id),
        )
        return 0 if row is None else int(row["count"])

    def increment_budget(self, window_kind: str, window_id: str, amount: int = 1) -> None:
        self._db.execute(
            """INSERT INTO budget_usage (window_kind, window_id, count) VALUES (?,?,?)
            ON CONFLICT(window_kind, window_id) DO UPDATE SET count = count + excluded.count""",
            (window_kind, window_id, amount),
        )

    # --- feedback ------------------------------------------------------------------------
    def add_feedback(self, target_action_id, classification, weight, created_at) -> None:
        self._db.execute(
            """INSERT INTO feedback (target_action_id, classification, weight, created_at)
            VALUES (?,?,?,?)""",
            (target_action_id, classification, weight, txt(created_at)),
        )

    # --- (de)serialization ---------------------------------------------------------------
    @staticmethod
    def _dump_self(m: SelfModel) -> str:
        import json

        return json.dumps(
            {
                "initiative": m.initiative,
                "inhibition": m.inhibition,
                "persistence": m.persistence,
                "recent_proactive_messages": m.recent_proactive_messages,
                "last_outward_action_at": txt(m.last_outward_action_at),
                "last_delivered_proactive_at": txt(m.last_delivered_proactive_at),
                "dominant_topic": m.dominant_topic,
            }
        )

    @staticmethod
    def _load_self(text: str) -> SelfModel:
        import json

        d = json.loads(text)
        return SelfModel(
            initiative=d["initiative"],
            inhibition=d["inhibition"],
            persistence=d["persistence"],
            recent_proactive_messages=d.get("recent_proactive_messages", 0),
            last_outward_action_at=dt(d.get("last_outward_action_at")),
            last_delivered_proactive_at=dt(d.get("last_delivered_proactive_at")),
            dominant_topic=d.get("dominant_topic"),
        )

    @staticmethod
    def _dump_conversation(c: ConversationState) -> str:
        import json

        return json.dumps(
            {
                "mode": c.mode.value,
                "last_human_message_at": txt(c.last_human_message_at),
                "last_agent_delivered_at": txt(c.last_agent_delivered_at),
                "active_observed_silence_seconds": c.active_observed_silence_seconds,
                "recent_human_turn_timestamps": [
                    txt(ts) for ts in c.recent_human_turn_timestamps
                ],
            }
        )

    @staticmethod
    def _load_conversation(text: str) -> ConversationState:
        import json

        d = json.loads(text)
        return ConversationState(
            mode=ConversationMode(d["mode"]),
            last_human_message_at=dt(d.get("last_human_message_at")),
            last_agent_delivered_at=dt(d.get("last_agent_delivered_at")),
            active_observed_silence_seconds=d.get("active_observed_silence_seconds", 0.0),
            recent_human_turn_timestamps=tuple(
                dt(ts) for ts in d.get("recent_human_turn_timestamps", []) if ts
            ),
        )
