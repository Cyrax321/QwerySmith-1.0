#!/usr/bin/env python3
"""
harness/self_healing.py -- AST & Reflection Self-Healing Engine for QwerySmith

Diagnoses SQLite execution tracebacks, analyzes ambiguous table/column references,
and constructs structured repair prompts for real-time model reflection.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepairRecord:
    """Tracks a single repair event for evaluation telemetry."""
    session_id: str
    db_name: str
    question: str
    original_sql: str
    repaired_sql: str
    error_msg: str
    attempt_count: int = 1
    success: bool = True


class SelfHealingEngine:
    """
    Reflection and self-healing engine.
    Analyzes SQL syntax errors, ambiguous column names, missing tables,
    and constructs high-precision repair instructions for LLM re-generation.
    """

    def __init__(self):
        self.repair_history: List[RepairRecord] = []

    def diagnose_error(self, error_msg: str, failed_sql: str) -> Dict[str, Any]:
        """Classifies the SQLite error and suggests repair tactics."""
        clean_err = error_msg.lower()
        diagnosis = {
            "error_type": "UNKNOWN",
            "suggested_fix": "Verify query syntax and match against schema.",
            "target_entity": None,
        }

        # Check 1: Ambiguous column name in JOINs
        ambig_match = re.search(r"ambiguous column name:\s*(\w+)", error_msg, re.IGNORECASE)
        if ambig_match:
            col = ambig_match.group(1)
            diagnosis["error_type"] = "AMBIGUOUS_COLUMN"
            diagnosis["target_entity"] = col
            diagnosis["suggested_fix"] = (
                f"Column '{col}' exists in multiple joined tables. "
                f"Prefix every occurrence of '{col}' with its specific table name or alias (e.g. table.{col})."
            )
            return diagnosis

        # Check 2: No such column
        col_match = re.search(r"no such column:\s*([\w.]+)", error_msg, re.IGNORECASE)
        if col_match:
            col = col_match.group(1)
            diagnosis["error_type"] = "MISSING_COLUMN"
            diagnosis["target_entity"] = col
            diagnosis["suggested_fix"] = (
                f"Column '{col}' does not exist. Check the DDL schema for the exact column spelling or alias."
            )
            return diagnosis

        # Check 3: No such table
        tbl_match = re.search(r"no such table:\s*([\w.]+)", error_msg, re.IGNORECASE)
        if tbl_match:
            tbl = tbl_match.group(1)
            diagnosis["error_type"] = "MISSING_TABLE"
            diagnosis["target_entity"] = tbl
            diagnosis["suggested_fix"] = (
                f"Table '{tbl}' does not exist in this database. Check the available tables in the schema."
            )
            return diagnosis

        # Check 4: Syntax error
        if "syntax error" in clean_err:
            diagnosis["error_type"] = "SYNTAX_ERROR"
            diagnosis["suggested_fix"] = "Fix SQL syntax near the indicated clause, matching parentheses and quotes."
            return diagnosis

        return diagnosis

    def build_repair_prompt(
        self,
        question: str,
        failed_sql: str,
        error_feedback: str,
        diagnosis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Constructs an actionable user prompt instructing the model how to repair its query."""
        if not diagnosis:
            diagnosis = self.diagnose_error(error_feedback, failed_sql)

        return (
            f"\n\nPrevious attempted query:\n```sql\n{failed_sql}\n```\n\n"
            f"Execution failed with SQLite error:\n{error_feedback}\n\n"
            f"Diagnosis & Repair Instruction:\n{diagnosis['suggested_fix']}\n"
            "Please output the corrected SQL query. Ensure table aliases are correct, verify column names against the schema, "
            "and prefix every selected or grouped column with its table alias."
        )

    def record_repair(
        self,
        session_id: str,
        db_name: str,
        question: str,
        original_sql: str,
        repaired_sql: str,
        error_msg: str,
        success: bool = True,
    ) -> RepairRecord:
        """Records a repair event."""
        rec = RepairRecord(
            session_id=session_id,
            db_name=db_name,
            question=question,
            original_sql=original_sql,
            repaired_sql=repaired_sql,
            error_msg=error_msg,
            success=success,
        )
        self.repair_history.append(rec)
        return rec

    def get_stats(self) -> Dict[str, Any]:
        """Returns repair statistics for evaluation harnesses."""
        total = len(self.repair_history)
        successful = sum(1 for r in self.repair_history if r.success)
        return {
            "total_repairs_attempted": total,
            "repairs_succeeded": successful,
            "repair_success_rate": (successful / total) if total > 0 else 1.0,
            "repair_history": [asdict(r) for r in self.repair_history[-20:]],
        }
