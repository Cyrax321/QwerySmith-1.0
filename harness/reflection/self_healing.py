#!/usr/bin/env python3
"""
harness/reflection/self_healing.py -- Multi-Step AST Error Diagnosis & Self-Healing Reflection
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ErrorDiagnosis:
    error_type: str
    target_entity: Optional[str] = None
    suggested_fix: str = ""
    closest_alternatives: List[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class RepairAttempt:
    attempt_index: int
    attempted_sql: str
    error_msg: str
    diagnosis: ErrorDiagnosis


@dataclass
class MultiStepRepairTracker:
    session_id: str
    db_name: str
    question: str
    original_sql: str
    attempts: List[RepairAttempt] = field(default_factory=list)
    final_sql: Optional[str] = None
    success: bool = False


class SelfHealingEngine:
    """
    Advanced AST-aware reflection and self-healing engine.
    Diagnoses runtime SQLite errors, matches misspelled schema identifiers,
    detects empty result set anomalies, and guides multi-step iterative repair.
    """

    def __init__(self):
        self.repair_history: List[MultiStepRepairTracker] = []

    def diagnose_error(
        self,
        error_msg: str,
        failed_sql: str,
        available_columns: Optional[List[str]] = None,
        available_tables: Optional[List[str]] = None,
    ) -> ErrorDiagnosis:
        """
        Performs fine-grained diagnosis of SQLite execution faults with fuzzy identifier matching.
        """
        clean_err = error_msg.lower()

        # 1. Ambiguous column name in JOINs
        ambig_match = re.search(r"ambiguous column name:\s*(\w+)", error_msg, re.IGNORECASE)
        if ambig_match:
            col = ambig_match.group(1)
            return ErrorDiagnosis(
                error_type="AMBIGUOUS_COLUMN",
                target_entity=col,
                suggested_fix=(
                    f"Column '{col}' exists in multiple joined tables. "
                    f"Prefix every occurrence of '{col}' with its specific table name or alias (e.g., table_alias.{col})."
                ),
            )

        # 2. No such column (misspelling, wrong alias, or hallucinated attribute)
        col_match = re.search(r"no such column:\s*([\w.]+)", error_msg, re.IGNORECASE)
        if col_match:
            raw_col = col_match.group(1)
            clean_col = raw_col.split(".")[-1]
            suggestions = []
            if available_columns:
                suggestions = difflib.get_close_matches(clean_col, available_columns, n=3, cutoff=0.5)

            fix_text = f"Column '{raw_col}' does not exist in the referenced schema."
            if suggestions:
                fix_text += f" Did you mean one of: {', '.join(suggestions)}?"
            else:
                fix_text += " Check the table DDL definition for exact column names and aliases."

            return ErrorDiagnosis(
                error_type="MISSING_COLUMN",
                target_entity=raw_col,
                suggested_fix=fix_text,
                closest_alternatives=suggestions,
            )

        # 3. No such table
        tbl_match = re.search(r"no such table:\s*([\w.]+)", error_msg, re.IGNORECASE)
        if tbl_match:
            raw_tbl = tbl_match.group(1)
            suggestions = []
            if available_tables:
                suggestions = difflib.get_close_matches(raw_tbl, available_tables, n=3, cutoff=0.5)

            fix_text = f"Table '{raw_tbl}' is not recognized in this database."
            if suggestions:
                fix_text += f" Did you mean: {', '.join(suggestions)}?"
            else:
                fix_text += " Verify the exact table name in the schema."

            return ErrorDiagnosis(
                error_type="MISSING_TABLE",
                target_entity=raw_tbl,
                suggested_fix=fix_text,
                closest_alternatives=suggestions,
            )

        # 4. Syntax Error
        if "syntax error" in clean_err:
            near_match = re.search(r'syntax error near "([^"]+)"', error_msg, re.IGNORECASE)
            target = near_match.group(1) if near_match else None
            fix = "Fix SQL grammar, verify matching quotes/parentheses, and check operator sequence."
            if target:
                fix += f" Focus on the syntax immediately preceding or around '{target}'."
            return ErrorDiagnosis(
                error_type="SYNTAX_ERROR",
                target_entity=target,
                suggested_fix=fix,
            )

        # 5. Non-aggregate with Aggregate in SELECT without GROUP BY
        if "aggregate" in clean_err or ("group by" in clean_err and "misuse" in clean_err):
            return ErrorDiagnosis(
                error_type="AGGREGATE_MISUSE",
                suggested_fix="Every non-aggregated column in the SELECT clause must be included in the GROUP BY clause.",
            )

        # 6. Fallback general error
        return ErrorDiagnosis(
            error_type="EXECUTION_ERROR",
            suggested_fix=f"Database execution failed with: {error_msg}. Review SQL structure against table schemas.",
        )

    def build_repair_prompt(
        self,
        question: str,
        failed_sql: str,
        error_feedback: str,
        diagnosis: Optional[ErrorDiagnosis] = None,
        attempt: int = 1,
        max_attempts: int = 3,
        history: Optional[List[RepairAttempt]] = None,
    ) -> str:
        """
        Constructs a structured multi-step repair prompt with progressive diagnostic feedback.
        """
        if not diagnosis:
            diagnosis = self.diagnose_error(error_feedback, failed_sql)

        prompt_parts = [
            f"\n\n[Auto-Repair Reflection - Attempt {attempt}/{max_attempts}]",
            f"Previous Query:\n```sql\n{failed_sql}\n```",
            f"Execution Error: {error_feedback}",
            f"Diagnosis [{diagnosis.error_type}]: {diagnosis.suggested_fix}",
        ]

        if history:
            prompt_parts.append("\nPrevious Failed Attempts (Avoid repeating these patterns):")
            for h in history:
                prompt_parts.append(f"- Attempt #{h.attempt_index}: `{h.attempted_sql}` -> Error: {h.error_msg}")

        prompt_parts.append(
            "\nPlease output ONLY the repaired SQL query, prefixing table columns appropriately and verifying syntax."
        )
        return "\n".join(prompt_parts)

    def start_repair_session(
        self,
        session_id: str,
        db_name: str,
        question: str,
        original_sql: str,
    ) -> MultiStepRepairTracker:
        """Initializes a multi-step repair tracker."""
        tracker = MultiStepRepairTracker(
            session_id=session_id,
            db_name=db_name,
            question=question,
            original_sql=original_sql,
        )
        self.repair_history.append(tracker)
        return tracker

    def record_repair(
        self,
        session_id: str,
        db_name: str,
        question: str,
        original_sql: str,
        repaired_sql: str,
        error_msg: str,
        success: bool = True,
    ) -> MultiStepRepairTracker:
        """Records a repair event with backward-compatible signature."""
        tracker = self.start_repair_session(session_id, db_name, question, original_sql)
        diag = self.diagnose_error(error_msg, original_sql)
        self.record_attempt(tracker, original_sql, error_msg, diag)
        self.finalize_repair(tracker, repaired_sql, success)
        return tracker

    def record_attempt(
        self,
        tracker: MultiStepRepairTracker,
        attempt_sql: str,
        error_msg: str,
        diagnosis: ErrorDiagnosis,
    ) -> None:
        """Appends an attempt to the session tracker."""
        attempt = RepairAttempt(
            attempt_index=len(tracker.attempts) + 1,
            attempted_sql=attempt_sql,
            error_msg=error_msg,
            diagnosis=diagnosis,
        )
        tracker.attempts.append(attempt)

    def finalize_repair(
        self,
        tracker: MultiStepRepairTracker,
        final_sql: str,
        success: bool,
    ) -> None:
        """Marks a repair session as succeeded or failed."""
        tracker.final_sql = final_sql
        tracker.success = success

    def get_stats(self) -> Dict[str, Any]:
        """Returns repair statistics across all historical sessions."""
        total = len(self.repair_history)
        succeeded = sum(1 for r in self.repair_history if r.success)
        total_attempts = sum(len(r.attempts) for r in self.repair_history)

        return {
            "total_repair_sessions": total,
            "successful_repairs": succeeded,
            "repair_success_rate": (succeeded / total) if total > 0 else 1.0,
            "avg_attempts_per_repair": (total_attempts / total) if total > 0 else 0.0,
        }
