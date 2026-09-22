#!/usr/bin/env python3
"""
harness/config.py -- Central Configuration for QwerySmith Agent Harness
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class HarnessConfig:
    """Configuration parameters governing execution, safety, and reasoning pipelines."""
    # Safety & Sandbox
    read_only: bool = True
    timeout_sec: float = 3.0
    max_rows: int = 100
    block_mutations: bool = True

    # Schema & Value Grounding
    enable_schema_pruning: bool = True
    schema_max_tables: int = 8
    enable_value_grounding: bool = True
    max_distinct_values_per_col: int = 25

    # Candidate Generation & Execution-Guided Selection
    candidate_count: int = 1
    candidate_temperature: float = 0.4
    enable_execution_voting: bool = True

    # Multi-Step Reflection & Healing
    max_repair_attempts: int = 3
    diagnose_empty_results: bool = True

    # Conversational Dialogue State Tracking
    enable_dst: bool = True
    max_dialogue_turns_context: int = 6

    # Dialect & Storage
    adapter_type: str = "sqlite"
    memory_path: Optional[Union[str, Path]] = None

    # Telemetry
    verbose_telemetry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessConfig":
        """Instantiate configuration from a dictionary, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)
