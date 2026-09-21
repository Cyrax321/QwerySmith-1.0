#!/usr/bin/env python3
"""
harness/conversation/state_tracker.py -- Dialogue State Tracking (DST) for Multi-Turn SQL Conversations
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


ANAPHORIC_PRONOUNS = re.compile(
    r"\b(they|them|those|that|these|it|their|the same|above|earlier|previous|former|latter)\b",
    re.IGNORECASE,
)

FOLLOWUP_CONJUNCTIONS = re.compile(
    r"^(and|or|also|then|what about|how about|which of them|which one|filter by|sort by|only)\b",
    re.IGNORECASE,
)


@dataclass
class DialogueTurn:
    turn_id: int
    question: str
    sql: str
    columns: List[str] = field(default_factory=list)
    rows_sample: List[Tuple[Any, ...]] = field(default_factory=list)
    human_summary: str = ""
    is_followup: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    db_name: str
    turns: List[DialogueTurn] = field(default_factory=list)
    active_tables: Set[str] = field(default_factory=set)
    active_filters: List[str] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)


class DialogueStateTracker:
    """
    Maintains conversational state across multi-turn user queries, detecting anaphoric
    references, tracking active query filters, and compacting context windows.
    """

    def __init__(self, max_context_turns: int = 5):
        self.max_context_turns = max_context_turns
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str, db_name: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id, db_name=db_name)
        return self.sessions[session_id]

    def is_followup(self, text: str, session_id: str) -> bool:
        """Determines if the question depends on earlier conversation context."""
        session = self.sessions.get(session_id)
        if not session or not session.turns:
            return False

        clean = text.strip()
        if FOLLOWUP_CONJUNCTIONS.search(clean) or ANAPHORIC_PRONOUNS.search(clean):
            return True

        # Elliptical questions (short phrases without a verb, e.g. "and in 2024?", "in Japan?")
        if len(clean.split()) <= 4 and ("?" in clean or clean.startswith("in ") or clean.startswith("for ")):
            return True

        return False

    def record_turn(
        self,
        session_id: str,
        db_name: str,
        question: str,
        sql: str,
        columns: List[str],
        rows: List[Tuple[Any, ...]],
        human_summary: str = "",
    ) -> DialogueTurn:
        """Records a completed turn and extracts dialogue entities."""
        session = self.get_or_create_session(session_id, db_name)
        turn_num = len(session.turns) + 1

        is_fup = self.is_followup(question, session_id)

        # Extract sample entities from rows for context injection
        sample_entities = []
        for r in rows[:3]:
            for val in r:
                if isinstance(val, str) and 2 <= len(val) <= 40:
                    sample_entities.append(val)

        turn = DialogueTurn(
            turn_id=turn_num,
            question=question,
            sql=sql,
            columns=columns,
            rows_sample=rows[:5],
            human_summary=human_summary,
            is_followup=is_fup,
        )
        session.turns.append(turn)

        # Update active entities
        if sample_entities:
            session.active_entities = sample_entities[:10]

        # Extract table references from SQL
        tbl_matches = re.findall(r"\bFROM\s+([a-zA-Z0-9_]+)|\bJOIN\s+([a-zA-Z0-9_]+)", sql, re.IGNORECASE)
        for m in tbl_matches:
            tbl = m[0] or m[1]
            if tbl:
                session.active_tables.add(tbl)

        return turn

    def build_context_prompt(self, session_id: str, current_question: str) -> str:
        """
        Constructs a focused multi-turn context block for prompt synthesis.
        """
        session = self.sessions.get(session_id)
        if not session or not session.turns:
            return ""

        is_fup = self.is_followup(current_question, session_id)
        if not is_fup:
            # If not an explicit follow-up, do not contaminate prompt with past SQL
            return ""

        recent_turns = session.turns[-self.max_context_turns:]
        prompt_parts = ["[Multi-Turn Conversational Dialogue History]"]

        for t in recent_turns:
            prompt_parts.append(f"Turn #{t.turn_id}: User asked: \"{t.question}\"")
            prompt_parts.append(f"Generated SQL: {t.sql}")
            if t.rows_sample:
                sample_str = ", ".join(str(r[0]) for r in t.rows_sample[:3] if r)
                prompt_parts.append(f"Returned Data (Sample): [{sample_str}]")

        prompt_parts.append(
            f"\nCurrent Follow-up Question: \"{current_question}\"\n"
            "Directive: This is an anaphoric follow-up refining or extending the previous query. "
            "Maintain relevant tables, joins, and context while applying the new constraint."
        )

        return "\n".join(prompt_parts)

    def clear_session(self, session_id: str) -> None:
        """Resets the conversational state for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Exports session state into a serializable dictionary."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id": session.session_id,
            "db_name": session.db_name,
            "active_tables": list(session.active_tables),
            "active_filters": session.active_filters,
            "active_entities": session.active_entities,
            "turns": [asdict(t) for t in session.turns],
        }

    def restore_session(self, data: Dict[str, Any]) -> SessionState:
        """Restores a session state from a serialized dictionary."""
        session = SessionState(
            session_id=data["session_id"],
            db_name=data["db_name"],
            active_tables=set(data.get("active_tables", [])),
            active_filters=data.get("active_filters", []),
            active_entities=data.get("active_entities", []),
        )
        for t_dict in data.get("turns", []):
            session.turns.append(DialogueTurn(**t_dict))
        self.sessions[session.session_id] = session
        return session
